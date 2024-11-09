import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


### single layers

class CustomConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, gain = 2**0.5, lrmul=1, custom_weight=None, bias=False, requires_grad=True, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.lrmul = lrmul
        
        he_std = gain * (input_channels//groups * kernel_size ** 2) ** (-0.5)  # He init
        init_std = he_std / lrmul

        if transpose: 
            if custom_weight is not None: self.weight = nn.Parameter(custom_weight[None, None, ...].expand(input_channels, output_channels//groups, -1, -1), requires_grad=requires_grad)
            else:                         self.weight = nn.Parameter(torch.randn(input_channels, output_channels//groups, kernel_size, kernel_size) * init_std, requires_grad=requires_grad)
            self.conv   = nn.functional.conv_transpose2d
        else:
            if custom_weight is not None: self.weight = nn.Parameter(custom_weight[None, None, ...].expand(output_channels, input_channels//groups, -1, -1), requires_grad=requires_grad)        
            else:                         self.weight = nn.Parameter(torch.randn(output_channels, input_channels//groups, kernel_size, kernel_size) * init_std, requires_grad=requires_grad)
            self.conv   = nn.functional.conv2d
            
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_channels), requires_grad=requires_grad)
            self.b_mul = lrmul
        else: self.bias = None

    def forward(self, x):
        x = self.conv(x, self.weight * self.lrmul, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        if self.bias is not None:
            bias = self.bias * self.b_mul
            x = x + bias.view(1, -1, 1, 1)
            
        return x

def conv2d(*args, **kwargs):
    return spectral_norm(CustomConv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(CustomConv2d(*args, **kwargs, transpose=True))

# def NormLayer(c, mode='group'):
#     if mode == 'group':
#         return nn.GroupNorm(c//4, c)
#     elif mode == 'batch':
#         return nn.BatchNorm2d(c)

### Downblocks

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = conv2d(in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, bias=bias, padding=1)
        self.pointwise = conv2d(in_channels, out_channels,
            kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        if not separable:
            self.main = nn.Sequential(
                conv2d(in_planes, out_planes, 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.main = nn.Sequential(
                SeparableConv2d(in_planes, out_planes, 3),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2, 2),
            )

    def forward(self, feat):
        return self.main(feat)

class DownBlockPatch(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes, separable),
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)

### CSM

class ResidualConvUnit(nn.Module):
    def __init__(self, cin, activation, bn):
        super().__init__()
        self.conv = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_add.add(self.conv(x), x)

class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, lowest=False):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output