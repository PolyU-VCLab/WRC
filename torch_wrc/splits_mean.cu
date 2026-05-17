#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/complex.h> // Ensure complex-number support
#include <vector>
#include <unordered_map>
#include <tuple>
#include <list>
#include <mutex>

// =========================================================================
// Forward Kernel
// =========================================================================
template <typename scalar_t>
__global__ void splits_mean_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int count,
    const int W_s,
    const int H_s,
    const int s) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Decode output coordinates [BC, W_s, H_s]
    int h_out = idx % H_s;
    int temp = idx / H_s;
    int w_out = temp % W_s;
    int bc = temp / W_s;

    // Compute base offset
    int input_spatial_size = (s * W_s) * (s * H_s);
    int input_base = bc * input_spatial_size;

    int stride_0 = W_s * s * H_s; 
    int stride_1 = s * H_s;       
    int stride_2 = H_s;           

    int spatial_start = w_out * stride_1 + h_out;
    int current_base = input_base + spatial_start;

    // 4. Accumulate the sum
    // Initialize to zero; complex types construct complex(0, 0) automatically.
    scalar_t sum = static_cast<scalar_t>(0);

    for (int i = 0; i < s; ++i) {     
        for (int j = 0; j < s; ++j) { 
            int offset = i * stride_0 + j * stride_2;
            // PyTorch c10::complex supports the overloaded + operator.
            sum += input[current_base + offset];
        }
    }

    // 5. Write the output
    // Complex values can be divided by the real scalar (s * s).
    output[idx] = sum / static_cast<scalar_t>(s * s);
}

// =========================================================================
// Backward Kernel
// =========================================================================
template <typename scalar_t>
__global__ void splits_mean_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    scalar_t* __restrict__ grad_input,
    const int count,        
    const int W_s,
    const int H_s,
    const int s) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Input View Layout: [BC, s, W_s, s, H_s]
    int d3_size = H_s;
    int d2_size = s;
    int d1_size = W_s;
    int d0_size = s;
    
    int temp = idx;
    int i3 = temp % d3_size; 
    temp /= d3_size;
    temp /= d2_size;
    int i1 = temp % d1_size; 
    temp /= d1_size;
    temp /= d0_size;
    int bc = temp;           

    int out_idx = bc * (W_s * H_s) + i1 * H_s + i3;

    // Divide by the averaging factor.
    grad_input[idx] = grad_output[out_idx] / static_cast<scalar_t>(s * s);
}

// =========================================================================
// C++ Launchers
// =========================================================================

at::Tensor splits_mean_cuda_forward(const at::Tensor &a, int64_t s) {
    const auto sizes = a.sizes();
    int64_t dims = a.dim();
    int64_t W = sizes[dims - 2];
    int64_t H = sizes[dims - 1];
    
    int64_t W_s = W / s;
    int64_t H_s = H / s;

    std::vector<int64_t> out_shape = sizes.vec();
    out_shape[dims - 2] = W_s;
    out_shape[dims - 1] = H_s;
    
    at::Tensor out = at::empty(out_shape, a.options());

    const int count = out.numel();
    const int threads = 512;
    const int blocks = (count + threads - 1) / threads;

    // Use AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES.
    // This supports both real tensors (WH2) and complex tensors (x1).
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(a.scalar_type(), "splits_mean_forward", ([&] {
        splits_mean_forward_kernel<scalar_t><<<blocks, threads>>>(
            a.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            count, W_s, H_s, s
        );
    }));

    return out;
}

at::Tensor splits_mean_cuda_backward(const at::Tensor &grad_output, int64_t s, at::IntArrayRef input_sizes) {
    at::Tensor grad_input = at::empty(input_sizes, grad_output.options());
    
    int64_t dims = grad_output.dim();
    int64_t W_s = grad_output.size(dims - 2);
    int64_t H_s = grad_output.size(dims - 1);

    const int count = grad_input.numel(); 
    const int threads = 512;
    const int blocks = (count + threads - 1) / threads;

    // Use the complex-aware dispatch macro here as well.
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(grad_output.scalar_type(), "splits_mean_backward", ([&] {
        splits_mean_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            count, W_s, H_s, s
        );
    }));

    return grad_input;
}