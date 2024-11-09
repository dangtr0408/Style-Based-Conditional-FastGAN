import torch
from torch import nn

#--------------------LAYER--------------------

def linear(*args, **kwargs):
    return CustomLinear(*args, **kwargs)
def conv2d(*args, **kwargs):
    return CustomConv2d(*args, **kwargs)
def convTranspose2d(*args, **kwargs):
    return CustomConv2d(*args, **kwargs, transpose=True)

def get_haar_wavelet():
    haar_wav_l = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h = 1 / (2 ** 0.5) * torch.ones(1, 2)
    haar_wav_h[0, 0] = -1 * haar_wav_h[0, 0]

    haar_wav_ll = haar_wav_l.T * haar_wav_l
    haar_wav_lh = haar_wav_h.T * haar_wav_l
    haar_wav_hl = haar_wav_l.T * haar_wav_h
    haar_wav_hh = haar_wav_h.T * haar_wav_h

    return haar_wav_ll, haar_wav_lh, haar_wav_hl, haar_wav_hh

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

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

class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size, gain = 2**0.5, lrmul=1, bias=True):
        super().__init__()
        self.lrmul = lrmul
        he_std = gain * input_size ** (-0.5)  # He init
        init_std = he_std / lrmul
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * init_std)

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else: self.bias = None

    def forward(self, x):
        if self.bias is not None:
            bias = self.bias * self.b_mul
        return nn.functional.linear(x, self.weight * self.lrmul, bias)
    
class UpSample(nn.Module):
    def __init__(self, up_sample_mode='bilinear'):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode=up_sample_mode)
    def forward(self, x):
        return self.up_sample(x)

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.nn.functional.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)

class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels), requires_grad=True)
    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)
        return feat + self.weight[None,:,None, None] * noise

class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 3, 1, 0),
                        PixelNorm(), 
                        nn.GLU(dim=1),)
    def forward(self, noise):
        return self.init(noise)
    
class AdaIN(nn.Module):
    def __init__(self, input_size, output_size):
        super(AdaIN, self).__init__()
        self.gamma_fc = linear(input_size, output_size)
        self.beta_fc  = linear(input_size, output_size)
        self.instance_norm = nn.InstanceNorm2d(output_size)
    def forward(self, x, w):
        gamma = self.gamma_fc(w).view(x.size(0), -1, 1, 1)
        beta  = self.beta_fc(w).view(x.size(0), -1, 1, 1)

        x_normalized = self.instance_norm(x)
        out = gamma * x_normalized + beta
        
        return out

class HaarTransform(nn.Module):
    def __init__(self):
        super(HaarTransform, self).__init__()
        self.requires_grad = False
    def dwt_init(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        x_LH = -x1 + x2 - x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_HH =  x1 - x2 - x3 + x4

        return torch.cat((x_LH, x_HL, x_HH), 1)
    def forward(self, x):
        return self.dwt_init(x)

class InverseHaarTransform(nn.Module):
    def __init__(self, in_channels=3):
        super(InverseHaarTransform, self).__init__()

        ll, lh, hl, hh = get_haar_wavelet()

        self.conv2d_lh = CustomConv2d(in_channels, in_channels, lh.shape[-1], stride=1, groups=3, custom_weight=-lh, requires_grad=False)
        self.conv2d_hl = CustomConv2d(in_channels, in_channels, hl.shape[-1], stride=1, groups=3, custom_weight=-hl, requires_grad=False)
        self.conv2d_hh = CustomConv2d(in_channels, in_channels, hh.shape[-1], stride=1, groups=3, custom_weight=hh, requires_grad=False)
        self.up        = UpSample(up_sample_mode='bilinear')

    def forward(self, input):
        lh, hl, hh = input.chunk(3, 1)

        lh = self.conv2d_lh(nn.functional.pad(self.up(lh), (0, 1, 0, 1)))
        hl = self.conv2d_hl(nn.functional.pad(self.up(hl), (0, 1, 0, 1)))
        hh = self.conv2d_hh(nn.functional.pad(self.up(hh), (0, 1, 0, 1)))

        return lh + hl + hh

class ToWavelets(nn.Module):
    def __init__(self, in_channel, out_planes):
        super().__init__()
        rgb_channels = 3
        self.sample = UpSample()

        self.iwt = InverseHaarTransform()
        self.dwt = HaarTransform()
        
        self.conv_1 = conv2d(in_channel, rgb_channels*3, 1, 1, 0)
        self.conv_2 = conv2d(rgb_channels*3, out_planes, 1, 1, 0)

    def forward(self, input):
        out = self.conv_1(input)#To wavelets of lh, hl, hh

        #To RGB (iwt) -> Upsample -> wavelets again (dwt)
        out = self.iwt(out)
        out = self.sample(out)
        out = self.dwt(out)

        out = self.conv_2(out)
            
        return out
    
#--------------------BLOCK--------------------

class UpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, w_dim, up_sample_mode='bilinear', use_wavelets=True):
        super().__init__()
        self.use_wavelets = use_wavelets
        self.up_sample    = UpSample(up_sample_mode)
        self.conv2d_1     = conv2d(in_planes, out_planes*2, 3, 1, 1)
        self.adain1       = AdaIN(w_dim, out_planes*2)
        self.act          = nn.GLU(dim=1)
        if self.use_wavelets: self.to_wavelets = ToWavelets(in_planes, out_planes*2)
            
    def forward(self, x, w_noise):
        if self.use_wavelets: wavelets = self.to_wavelets(x)
        else:                 wavelets = 0

        x = self.up_sample(x)

        x = self.conv2d_1(x)
        x = x + wavelets
        x = self.adain1(x, w_noise)
        x = self.act(x)
        
        return x
    
class UpBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes, w_dim, up_sample_mode='bilinear', use_wavelets=True):
        super().__init__()
        self.use_wavelets = use_wavelets
        self.up_sample = UpSample(up_sample_mode)
        self.conv2d_1  = conv2d(in_planes, out_planes*2, 3, 1, 1)
        self.conv2d_2  = conv2d(out_planes, out_planes*2, 3, 1, 1)
        self.adain1    = AdaIN(w_dim, out_planes*2)
        self.adain2    = AdaIN(w_dim, out_planes*2)
        self.noise1    = NoiseInjection(out_planes*2)
        self.noise2    = NoiseInjection(out_planes*2)
        self.act       = nn.GLU(dim=1)
        if self.use_wavelets: self.to_wavelets = ToWavelets(in_planes, out_planes*2)
            
    def forward(self, x, w_noise):
        if self.use_wavelets: wavelets = self.to_wavelets(x)
        else:                 wavelets = 0

        x = self.up_sample(x)

        x = self.conv2d_1(x)
        x = x + wavelets
        x = self.noise1(x)
        x = self.adain1(x, w_noise)
        x = self.act(x)
        
        x = self.conv2d_2(x)
        x = x + wavelets
        x = self.noise2(x)
        x = self.adain2(x, w_noise)
        x = self.act(x)
        
        return x

class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()
        self.conv2d_1 = conv2d(in_planes, out_planes, 4, 2, 1)
        self.conv2d_2 = conv2d(out_planes, out_planes, 3, 1, 1)
        self.direct   = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),)
        self.scale = torch.rsqrt(torch.tensor(2.0))
    def forward(self, feat):
        main = self.conv2d_1(feat)
        main = nn.LeakyReLU(0.2, inplace=True)(main)
        main = self.conv2d_2(main)
        main = nn.LeakyReLU(0.2, inplace=True)(main)

        return (main + self.direct(feat)) * self.scale
    
class SimpleDecoder(nn.Module):
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()
        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)
        def Block(in_planes, out_planes, up_sample_mode='nearest'):
            block = nn.Sequential(
                UpSample(up_sample_mode),
                conv2d(in_planes, out_planes, 3, 1, 1),
                PixelNorm(),
                nn.LeakyReLU(0.2, inplace=True),)
            return block
        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    Block(nfc_in, nfc[16]) ,
                                    Block(nfc[16], nfc[32]),
                                    Block(nfc[32], nfc[64]),
                                    Block(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1),
                                    nn.Tanh() )

    def forward(self, input):
        return self.main(input)

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=128, w_dim=512):
        super().__init__()
        self.mapping = nn.Sequential(
            linear(z_dim, z_dim, lrmul=0.01),
            nn.LeakyReLU(0.2, inplace=True), 
            linear(z_dim, w_dim, lrmul=0.01),
            nn.LeakyReLU(0.2, inplace=True), 
            linear(w_dim, w_dim, lrmul=0.01)
        )
    def forward(self, noise):
        mapping = self.mapping(noise)
        return mapping

#--------------------GAN--------------------

class Generator(nn.Module):
    def __init__(self, ngf=64, z_dim=128, w_dim=512, nc=3, im_size=1024):
        super(Generator, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)
        
        self.im_size = im_size
        self.mapping_network = MappingNetwork(z_dim, w_dim)
        
        self.init_constant = nn.Parameter(torch.randn((1, z_dim, 2, 2)))
        self.init          = InitLayer(z_dim, channel=nfc[4])
        
        self.feat_8        = UpBlockComp(nfc[4], nfc[8], w_dim)
        self.feat_16       = UpBlock(nfc[8], nfc[16], w_dim)
        self.feat_32       = UpBlockComp(nfc[16], nfc[32], w_dim)
        self.feat_64       = UpBlock(nfc[32], nfc[64], w_dim)
        self.feat_128      = UpBlockComp(nfc[64], nfc[128], w_dim)  
        self.feat_256      = UpBlock(nfc[128], nfc[256], w_dim)
        if im_size > 256:
            self.feat_512  = UpBlockComp(nfc[256], nfc[512], w_dim) 
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024], w_dim)

        self.conv_rgb   = conv2d(nfc[im_size], nc, 3, 1, 1)
        self.pixel_norm = PixelNorm()
        
    def forward(self, noise):
        mapped          = self.mapping_network(noise)

        init_constant   = self.pixel_norm(self.init_constant.expand(noise.shape[0], -1, -1, -1))

        feat_4          = self.init(init_constant)

        feat_8          = self.feat_8(feat_4, mapped)
        
        feat_16         = self.feat_16(feat_8, mapped)

        feat_32         = self.feat_32(feat_16, mapped)
        
        feat_64         = self.feat_64(feat_32, mapped)
        
        feat_128        = self.feat_128(feat_64, mapped)

        feat_256        = self.feat_256(feat_128, mapped)
        if self.im_size == 256:
            return torch.tanh(self.conv_rgb(feat_256))
        
        feat_512        = self.feat_512(feat_256, mapped)
        if self.im_size == 512:
            return torch.tanh(self.conv_rgb(feat_512))
            
        feat_1024       = self.feat_1024(feat_512, mapped)
        return torch.tanh(self.conv_rgb(feat_1024))
    


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:1, 128:0.5, 256:0.25, 512:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        minibatch_std_offset = 1
            
        if im_size == 1024:
            self.down_from_big = nn.Sequential( DownBlockComp(nc, nfc[512]),
                                                DownBlockComp(nfc[512], nfc[256]))
        elif im_size == 512:
            self.down_from_big = nn.Sequential( DownBlockComp(nc, nfc[256]))
        elif im_size == 256:
            self.down_from_big = nn.Sequential( conv2d(nc, nfc[256], 3, 1, 1),
                                                nn.LeakyReLU(0.2, inplace=True),)

        self.down_4  = DownBlockComp(nfc[256],nfc[128])
        self.down_8  = DownBlockComp(nfc[128],nfc[64])
        self.down_16 = DownBlockComp(nfc[64] ,nfc[32])
        self.down_32 = DownBlockComp(nfc[32] ,nfc[16])
        self.down_64 = DownBlockComp(nfc[16] ,nfc[8])

        self.out = nn.Sequential(conv2d(nfc[8]+minibatch_std_offset , nfc[4], 1, 1, 0), 
                                    nn.LeakyReLU(0.2, inplace=True),
                                    conv2d(nfc[4], 20, 4, 1, 0), 
                                    nn.LeakyReLU(0.2, inplace=True),
                                    nn.Flatten(start_dim=1),
                                    linear(500, 100),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    linear(100, 1))

        self.decoder_big = SimpleDecoder(nfc[8], nc)
        self.decoder_part = SimpleDecoder(nfc[16], nc)

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)
        
    def forward(self, imgs, get_aux=False):
        feat_2          = self.down_from_big(imgs)
        
        feat_4          = self.down_4(feat_2)

        feat_8          = self.down_8(feat_4)
        
        feat_16         = self.down_16(feat_8)

        feat_32         = self.down_32(feat_16)
        
        feat_last       = self.down_64(feat_32)

        feat_last_stats = self.minibatch_std(feat_last)

        out = self.out(feat_last_stats).view(-1)

        if get_aux:
            rec_img_big = self.decoder_big(feat_last)

            part = torch.randint(0, 4, (1,)).item()
            if part==0: rec_img_part        = self.decoder_part(feat_32[:,:,:8,:8])
            if part==1: rec_img_part        = self.decoder_part(feat_32[:,:,:8,8:])
            if part==2: rec_img_part        = self.decoder_part(feat_32[:,:,8:,:8])
            if part==3: rec_img_part        = self.decoder_part(feat_32[:,:,8:,8:])
            return out, part, [rec_img_big, rec_img_part]
        
        return out