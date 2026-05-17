#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

// Declare launcher functions implemented in .cu files
at::Tensor sfold_upsample_cuda_forward(const at::Tensor &x, int64_t s);
at::Tensor sfold_upsample_cuda_backward(const at::Tensor &grad_output, int64_t s);
at::Tensor splits_mean_cuda_forward(const at::Tensor &a, int64_t s);
at::Tensor splits_mean_cuda_backward(const at::Tensor &grad_output, int64_t s, at::IntArrayRef input_sizes);
at::Tensor p2o_cuda_forward(const at::Tensor &psf, int64_t H, int64_t W);
at::Tensor p2o_cuda_backward(const at::Tensor &grad_otf, at::IntArrayRef psf_size);

// Define the autograd function
class SFoldUpsampleFunction : public torch::autograd::Function<SFoldUpsampleFunction> {
public:
    // Forward function
    // ctx stores information needed for backward propagation.
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor x, int64_t s) {
        // Save the scale parameter for backward.
        ctx->saved_data["s"] = s; 
        
        // Ensure the input is contiguous, which is important for custom kernels.
        x = x.contiguous(); 
        
        return sfold_upsample_cuda_forward(x, s);
    }

    // Backward function
    // grad_outputs contains output gradients; there is only one output here.
    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
        auto grad_output = grad_outputs[0].contiguous();
        int64_t s = ctx->saved_data["s"].toInt();

        auto grad_input = sfold_upsample_cuda_backward(grad_output, s);

        // Return values must match the number of forward arguments.
        // forward has two arguments: (x, s).
        // x requires gradients, so return grad_input.
        // s is an integer and non-differentiable, so return torch::Tensor() (None).
        return {grad_input, torch::Tensor()};
    }
};

// Wrap the operation as a simple C++ function call
at::Tensor sfold_upsample(const at::Tensor &x, int64_t s) {
    // Use the custom operator when the input is on CUDA.
    if (x.is_cuda()) {
        return SFoldUpsampleFunction::apply(x, s);
    } 
    // CPU fallback (keeps the original CPU implementation path)
    else {
        // ... original sfold_upsample_zero_insertion code ...
        // For simplicity, this assumes the original implementation still exists.
        TORCH_CHECK(false, "CPU version not implemented in this snippet"); 
    }
}



// Autograd Function
class SplitsMeanFunction : public torch::autograd::Function<SplitsMeanFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor input, int64_t s) {
        ctx->saved_data["s"] = s;
        ctx->saved_data["input_shape"] = input.sizes();
        
        if (input.is_cuda()) {
             return splits_mean_cuda_forward(input.contiguous(), s);
        } else {
             // Fallback to CPU implementation (the original code)
             // ...
             TORCH_CHECK(false, "CPU not implemented in this snippet");
        }
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
        auto grad_output = grad_outputs[0].contiguous();
        int64_t s = ctx->saved_data["s"].toInt();
        auto input_shape = ctx->saved_data["input_shape"].toIntVector();

        if (grad_output.is_cuda()) {
             return {splits_mean_cuda_backward(grad_output, s, input_shape), torch::Tensor()};
        } else {
             TORCH_CHECK(false, "CPU backward not implemented");
        }
    }
};

// Wrapper function
at::Tensor splits_mean(const at::Tensor &a, int64_t s) {
    if (s == 1) return a;
    return SplitsMeanFunction::apply(a, s);
}


// Declare CUDA functions


// Define the autograd function for pad and roll
class PadRollFunction : public torch::autograd::Function<PadRollFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor psf, int64_t H, int64_t W) {
        ctx->saved_data["psf_shape"] = psf.sizes();
        // Call the CUDA kernel.
        return p2o_cuda_forward(psf, H, W);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
        auto grad_otf = grad_outputs[0].contiguous();
        auto psf_shape = ctx->saved_data["psf_shape"].toIntVector();
        
        // Call the gather kernel.
        auto grad_psf = p2o_cuda_backward(grad_otf, psf_shape);
        
        return {grad_psf, torch::Tensor(), torch::Tensor()};
    }
};

// Wrap the final p2o function
at::Tensor p2o(const at::Tensor &psf, int64_t H, int64_t W) {
    // 1. Get the spatial-domain tensor after pad and roll
    at::Tensor otf_spatial;
    
    if (psf.is_cuda()) {
        otf_spatial = PadRollFunction::apply(psf, H, W);
    } else {
        // CPU fallback (original logic)
        otf_spatial = at::zeros({psf.size(0), psf.size(1), H, W}, psf.options());
        int64_t kh = psf.size(2), kw = psf.size(3);
        otf_spatial.index_put_(
            {at::indexing::Slice(), at::indexing::Slice(), at::indexing::Slice(0, kh), at::indexing::Slice(0, kw)},
            psf);
        otf_spatial = at::roll(otf_spatial, {-(int64_t)(H / 2), -(int64_t)(W / 2)}, {-2, -1});
    }

    // 2. Run FFT; PyTorch fft_fftn uses cuFFT underneath and handles gradients.
    std::vector<int64_t> fft_dims = {-2, -1};
    return at::fft_fftn(otf_spatial, c10::nullopt, c10::optional<c10::IntArrayRef>(fft_dims), c10::nullopt);
}



at::Tensor weighted_reverse_convolution_forward(
    at::Tensor x,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor w_conv_low_weight,
    c10::optional<at::Tensor> w_conv_low_bias,   // Optional bias
    at::Tensor w_conv_high_weight,
    c10::optional<at::Tensor> w_conv_high_bias,  // Optional bias
    int64_t scale,
    int64_t padding,
    double eps)
{
    TORCH_CHECK(x.dim() == 4, "x must be (B,C,H,W)");
    TORCH_CHECK(weight.dim() == 4 && weight.size(0) == 1, "weight must be (1,C,kh,kw)");
    TORCH_CHECK(bias.dim() == 4 && bias.size(0) == 1 && bias.size(2) == 1 && bias.size(3) == 1, "bias must be (1,C,1,1)");
    TORCH_CHECK(scale >= 1, "scale must be >= 1");

    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();
    w_conv_low_weight = w_conv_low_weight.contiguous();
    w_conv_high_weight = w_conv_high_weight.contiguous();
    if (w_conv_low_bias.has_value() && w_conv_low_bias->defined())
        w_conv_low_bias = w_conv_low_bias->contiguous();
    if (w_conv_high_bias.has_value() && w_conv_high_bias->defined())
        w_conv_high_bias = w_conv_high_bias->contiguous();

    // // NaN diagnostics: check all input tensors
    // #define CHECK_NAN(name, tensor) \
    //     if (at::isnan(tensor).any().item<bool>()) \
    //         printf("[NaN] %s contains NaN!\n", name); \
    //     if (at::isinf(tensor).any().item<bool>()) \
    //         printf("[Inf] %s contains Inf!\n", name);
    // CHECK_NAN("x", x);
    // CHECK_NAN("weight", weight);
    // CHECK_NAN("bias", bias);
    // CHECK_NAN("w_conv_low_weight", w_conv_low_weight);
    // CHECK_NAN("w_conv_high_weight", w_conv_high_weight);

    if (padding > 0)
    {
        std::vector<int64_t> pad = {padding, padding, padding, padding};
        x = at::pad(x, pad, "circular", 0);
    }

    at::Tensor WL = at::conv2d(x, w_conv_low_weight, w_conv_low_bias, {1, 1}, {1, 1}, {1, 1}, 1);
    at::Tensor WL2 = at::log1p(at::pow(at::clamp(WL, -1e4, 1e4), 2));
    // CHECK_NAN("WL2", WL2);

    at::Tensor biaseps = at::softplus(bias) + eps;
    // CHECK_NAN("biaseps", biaseps);

    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);

    at::Tensor STy = sfold_upsample(WL2 * x, scale);

    if (scale != 1)
    {
        x = at::upsample_bilinear2d(x, {H * scale, W * scale}, /*align_corners=*/true, c10::nullopt, c10::nullopt);
    }

    at::Tensor WH = at::conv2d(x, w_conv_high_weight, w_conv_high_bias, {1, 1}, {1, 1}, {1, 1}, 1);
    at::Tensor WH2 = at::log1p(at::pow(at::clamp(WH, -1e4, 1e4), 2));
    // CHECK_NAN("WH2", WH2);

    const int64_t Hs = H * scale;
    const int64_t Ws = W * scale;

    at::Tensor FB = p2o(weight, Hs, Ws);
    // CHECK_NAN("FB_real", at::real(FB));

    at::Tensor FBC = FB.conj();
    at::Tensor F2B = at::abs(FB).pow(2);

    std::vector<int64_t> fft_dims = {-2, -1};

    at::Tensor F_STy = at::fft_fftn(STy, c10::nullopt, c10::optional<c10::IntArrayRef>(fft_dims), c10::nullopt);
    at::Tensor FBFy = FBC * F_STy;
    // CHECK_NAN("FBFy_real", at::real(FBFy));

    at::Tensor WH2_plus_biaseps_x = (WH2 + biaseps) * x;
    // CHECK_NAN("WH2_plus_biaseps_x", WH2_plus_biaseps_x);

    at::Tensor F_term = at::fft_fftn(WH2_plus_biaseps_x, c10::nullopt, c10::optional<c10::IntArrayRef>(fft_dims), c10::nullopt);

    at::Tensor FR = FBFy + F_term;
    // CHECK_NAN("FR_real", at::real(FR));

    at::Tensor x1 = FB * FR;
    at::Tensor FBR = WL2 * splits_mean(x1, scale);
    // CHECK_NAN("FBR_real", at::real(FBR));

    at::Tensor invW = WL2 * splits_mean(F2B, scale);
    at::Tensor WH2_splits_mean = splits_mean(WH2, scale);

    at::Tensor denom1 = at::clamp_min(invW + WH2_splits_mean + biaseps, 1e-3);
    at::Tensor invWBR = FBR / denom1;
    // CHECK_NAN("invWBR_real", at::real(invWBR));

    at::Tensor invWBR_rep = invWBR.repeat({1, 1, scale, scale});
    at::Tensor FCBinvWBR = FBC * invWBR_rep;

    at::Tensor denom2 = at::clamp_min(WH2 + biaseps, 1e-3);
    at::Tensor FX = (FR - FCBinvWBR) / denom2;
    // CHECK_NAN("FX_real", at::real(FX));

    at::Tensor out_c = at::fft_ifftn(FX, c10::nullopt, c10::optional<c10::IntArrayRef>(fft_dims), c10::nullopt);
    at::Tensor out = at::real(out_c);
    // CHECK_NAN("out", out);

    if (padding > 0)
    {
        int64_t pad_scaled = padding * scale;
        using at::indexing::Slice;
        out = out.index({Slice(), Slice(), Slice(pad_scaled, -pad_scaled), Slice(pad_scaled, -pad_scaled)});
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &weighted_reverse_convolution_forward,
          "Weighted Reverse Convolution Forward",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("w_conv_low_weight"),
          pybind11::arg("w_conv_low_bias"),
          pybind11::arg("w_conv_high_weight"),
          pybind11::arg("w_conv_high_bias"),
          pybind11::arg("scale"),
          pybind11::arg("padding"),
          pybind11::arg("eps") = 1e-5
    );
}