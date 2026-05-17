#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <list>
#include <mutex>

// =========================================================================
// Forward Kernel: Pad and Roll in one go
// =========================================================================
// Number of threads equals the number of PSF elements, which is usually small.
// Map each PSF coordinate (y, x) directly to the padded output coordinate.
template <typename scalar_t>
__global__ void psf_pad_roll_forward_kernel(
    const scalar_t* __restrict__ psf,
    scalar_t* __restrict__ otf,
    const int count,      // psf.numel()
    const int kh, const int kw,
    const int H, const int W,
    const int shift_h, const int shift_w) { // shift_h = -kh/2
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // 1. Decode the PSF coordinates
    // PSF layout: [B, C, kh, kw]
    int x = idx % kw;
    int temp = idx / kw;
    int y = temp % kh;
    int bc = temp / kh; // Batch * Channel

    // 2. Compute the target coordinates after roll
    // The target is the large canvas (H, W), so handle negative modulo values.
    // target_y = (y + shift_h) % H
    int target_y = (y + shift_h) % H;
    if (target_y < 0) target_y += H;

    int target_x = (x + shift_w) % W;
    if (target_x < 0) target_x += W;

    // 3. Compute the linear OTF index
    // OTF layout: [B, C, H, W]
    // stride_h = W, stride_bc = H * W
    int out_idx = bc * (H * W) + target_y * W + target_x;

    // 4. Write the value
    otf[out_idx] = psf[idx];
}

// =========================================================================
// Backward Kernel: Unroll and Crop
// =========================================================================
// Number of threads equals the number of PSF elements.
// Gather gradients back from the corresponding large-canvas positions.
template <typename scalar_t>
__global__ void psf_pad_roll_backward_kernel(
    const scalar_t* __restrict__ grad_otf,
    scalar_t* __restrict__ grad_psf,
    const int count,
    const int kh, const int kw,
    const int H, const int W,
    const int shift_h, const int shift_w) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    int x = idx % kw;
    int temp = idx / kw;
    int y = temp % kh;
    int bc = temp / kh;

    int target_y = (y + shift_h) % H;
    if (target_y < 0) target_y += H;

    int target_x = (x + shift_w) % W;
    if (target_x < 0) target_x += W;

    int src_idx = bc * (H * W) + target_y * W + target_x;

    grad_psf[idx] = grad_otf[src_idx];
}

// =========================================================================
// Launchers
// =========================================================================

at::Tensor p2o_cuda_forward(const at::Tensor &psf, int64_t H, int64_t W) {
    // 1. Allocate the OTF canvas initialized to zero
    // It must be zero-initialized because the kernel only writes nonzero positions.
    at::Tensor otf = at::zeros({psf.size(0), psf.size(1), H, W}, psf.options());
    
    int64_t kh = psf.size(2);
    int64_t kw = psf.size(3);
    
    // Integer division truncates in C++; sizes are positive here, so this is safe.
    int shift_h = -H / 2;
    int shift_w = -W / 2;

    const int count = psf.numel();
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(psf.scalar_type(), "psf_pad_roll_forward", ([&] {
        psf_pad_roll_forward_kernel<scalar_t><<<blocks, threads>>>(
            psf.data_ptr<scalar_t>(),
            otf.data_ptr<scalar_t>(),
            count, kh, kw, H, W, shift_h, shift_w
        );
    }));

    // 2. Only handle the spatial-domain transform here; leave FFT to ATen.
    return otf;
}

at::Tensor p2o_cuda_backward(const at::Tensor &grad_otf, at::IntArrayRef psf_size) {
    at::Tensor grad_psf = at::empty(psf_size, grad_otf.options());
    
    int64_t kh = psf_size[2];
    int64_t kw = psf_size[3];
    int64_t H = grad_otf.size(2);
    int64_t W = grad_otf.size(3);

    int shift_h = -H / 2;
    int shift_w = -W / 2;

    const int count = grad_psf.numel();
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_otf.scalar_type(), "psf_pad_roll_backward", ([&] {
        psf_pad_roll_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_otf.data_ptr<scalar_t>(),
            grad_psf.data_ptr<scalar_t>(),
            count, kh, kw, H, W, shift_h, shift_w
        );
    }));

    return grad_psf;
}