import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA_PROJECT_PATH = Path(__file__).resolve().parents[2] / "torch_wrc"
if CUDA_PROJECT_PATH.exists():
    sys.path.insert(0, str(CUDA_PROJECT_PATH))

use_cuda = False
try:
    from load import wrc as wrc_cuda
    use_cuda = True
except Exception:
    wrc_cuda = None


class WRC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=1, padding=2, padding_mode='circular', eps=1e-5):
        super(WRC, self).__init__()
        """
        WRC Operator.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the operation.
            kernel_size (int): Size of the kernel.
            scale (int): Upsampling factor. For example, `scale=2` doubles the resolution.
            padding (int): Padding size. Recommended value is `kernel_size - 1`.
            padding_mode (str, optional): Padding method. One of {'reflect', 'replicate', 'circular', 'constant'}.
                                        Default is `circular`.
            eps (float, optional): Small value added to denominators for numerical stability.
                                Default is a small value like 1e-5.

        Returns:
            Tensor: Output tensor of shape (N, out_channels, H * scale, W * scale), where spatial dimensions
                    are upsampled by the given scale factor.
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size =  kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps

        # ensure depthwise
        assert self.out_channels == self.in_channels
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.weight.data = nn.functional.softmax(self.weight.data.view(1,self.in_channels,-1), dim=-1).view(1, self.in_channels, self.kernel_size, self.kernel_size)

        self.w_conv_low = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.w_conv_high = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        if use_cuda:
            out = wrc_cuda.forward(
                x,
                self.weight,
                self.bias,
                self.w_conv_low.weight,
                None,
                self.w_conv_high.weight,
                None,
                self.scale,
                self.padding,
                self.eps
            )
        else:
            if self.padding > 0:
                x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode, value=0)
            
            WL2 = torch.log1p(torch.pow(self.w_conv_low(x), 2))
            biaseps = F.softplus(self.bias) + self.eps
            
            _, _, h, w = x.shape
            STy = self.upsample(WL2*x, scale=self.scale)
            if self.scale != 1:
                x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
            WH2 = torch.log1p(torch.pow(self.w_conv_high(x), 2)) # (b, 1, h*scale, w*scale)

            FB = self.p2o(self.weight, (h*self.scale, w*self.scale))
            FBC = torch.conj(FB)
            F2B = torch.pow(torch.abs(FB), 2)
            FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
            
            FR = FBFy + torch.fft.fftn((WH2+biaseps)*x, dim=(-2,-1))
            x1 = FB.mul(FR)
            FBR = WL2 * torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
            invW = WL2 * torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
            invWBR = FBR.div(invW + torch.mean(self.splits(WH2, self.scale), dim=-1, keepdim=False) + biaseps)
            FCBinvWBR = FBC*invWBR.repeat(1, 1, self.scale, self.scale)
            FX = (FR-FCBinvWBR).div(WH2+biaseps)
            out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

            if self.padding > 0:
                out = out[..., self.padding*self.scale:-self.padding*self.scale, self.padding*self.scale:-self.padding*self.scale]

        return out

    def splits(self, a, scale):
        '''
        Split tensor `a` into `scale x scale` distinct blocks.
        Args:
            a: Tensor of shape (..., W, H)
            scale: Split factor
        Returns:
            b: Tensor of shape (..., W/scale, H/scale, scale^2)
        '''
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        # Reshape to separate the scale factors
        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        # Generate the permutation order
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3, len(leading_dims), len(leading_dims) + 2]
        b = b.permute(*permute_order).contiguous()

        # Combine the scale dimensions
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b


    def p2o(self, psf, shape):
        '''
        Convert point-spread function to optical transfer function.
        otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
        point-spread function (PSF) array and creates the optical transfer
        function (OTF) array that is not influenced by the PSF off-centering.
        Args:
            psf: NxCxhxw
            shape: [H, W]
        Returns:
            otf: NxCxHxWx2
        '''
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[...,:psf.shape[-2],:psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(shape[0]/2), -int(shape[1]/2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2,-1))

        return otf

    def upsample(self, x, scale=3):
        '''s-fold upsampler
        Upsampling the spatial size by filling the new entries with zeros
        x: tensor image, NxCxWxH
        '''
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*scale, x.shape[3]*scale)).type_as(x)
        z[..., st::scale, st::scale].copy_(x)
        return z
