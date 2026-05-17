#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <list>
#include <mutex>

// CUDA Kernel
// Template supports float, double, half, and similar scalar types.
template <typename scalar_t>
__global__ void sfold_upsample_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int count,      // Total number of input elements
    const int in_h,       // Input height
    const int in_w,       // Input width
    const int out_h,      // Output height (in_h * s)
    const int out_w,      // Output width (in_w * s)
    const int s           // scale
) {
    // 1D grid index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (idx >= count) return;

    // Core logic: decode (batch_channel, y, x) from the linear index.
    // Batch and channel are merged into one dimension for simpler indexing.
    
    int w = idx % in_w;
    int temp = idx / in_w;
    int h = temp % in_h;
    int bc = temp / in_h; // batch * channel index

    // Compute the position in the output tensor
    int h_out = h * s;
    int w_out = w * s;

    // Compute the linear output index
    // output layout: [BC, out_h, out_w]
    int out_idx = bc * (out_h * out_w) + h_out * out_w + w_out;

    // Copy the value
    output[out_idx] = input[idx];
}

// ================= Backward Kernel =================
// Each thread computes one grad_input value.
// It gathers the value from the corresponding grad_output position.
template <typename scalar_t>
__global__ void sfold_upsample_backward_kernel(
    const scalar_t* __restrict__ grad_output, // Gradient from the next layer (large map)
    scalar_t* __restrict__ grad_input,        // Gradient to compute (small map)
    const int count,      // Number of elements in the small map
    const int in_h,       // Small-map height
    const int in_w,       // Small-map width
    const int out_h,      // Large-map height
    const int out_w,      // Large-map width
    const int s) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // 1. Decode the small-map coordinate for this thread
    int w = idx % in_w;
    int temp = idx / in_w;
    int h = temp % in_h;
    int bc = temp / in_h;

    // 2. Map to the large-map coordinate
    int h_out = h * s;
    int w_out = w * s;
    int out_idx = bc * (out_h * out_w) + h_out * out_w + w_out;

    // 3. Gather the gradient from the corresponding large-map position
    grad_input[idx] = grad_output[out_idx];
}

// ================= C++ Launcher Functions =================

// Forward Launcher
at::Tensor sfold_upsample_cuda_forward(const at::Tensor &x, int64_t s) {
    auto sizes = x.sizes().vec();
    int64_t H = sizes[2];
    int64_t W = sizes[3];
    sizes[2] *= s;
    sizes[3] *= s;
    
    at::Tensor z = at::zeros(sizes, x.options()); // Must be initialized to zero
    
    const int count = x.numel();
    const int threads = 512;
    const int blocks = (count + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "sfold_upsample_forward", ([&] {
        sfold_upsample_forward_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            z.data_ptr<scalar_t>(),
            count, H, W, H * s, W * s, s
        );
    }));
    return z;
}

// Backward Launcher
at::Tensor sfold_upsample_cuda_backward(const at::Tensor &grad_output, int64_t s) {
    auto sizes = grad_output.sizes().vec();
    // Compute the inverse output size
    sizes[2] /= s;
    sizes[3] /= s;
    int64_t H_in = sizes[2];
    int64_t W_in = sizes[3];
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);

    // empty is enough because every position is written.
    at::Tensor grad_input = at::empty(sizes, grad_output.options());
    
    const int count = grad_input.numel();
    const int threads = 512;
    const int blocks = (count + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "sfold_upsample_backward", ([&] {
        sfold_upsample_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            count, H_in, W_in, H_out, W_out, s
        );
    }));
    return grad_input;
}