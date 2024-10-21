import numpy as np

import torch
from torch import nn, einsum
from torch.nn.utils import spectral_norm

#Layer

def conv2d(*args, **kwargs):
    return spectral_norm(CustomConv2d(*args, **kwargs))
def linear(*args, **kwargs):
    return spectral_norm(CustomLinear(*args, **kwargs))
def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))
def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)
def groupNorm2d(c):
    return nn.GroupNorm(c//4, c)

class CustomConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, gain = 2**0.5, lrmul=1, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.w_mul = lrmul
        he_std = gain * (input_channels//groups * kernel_size ** 2) ** (-0.5)  # He init
        init_std = he_std / lrmul
        
        self.weight = nn.Parameter(torch.randn(output_channels, input_channels//groups, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else: self.bias = None
    def forward(self, x):
        x = nn.functional.conv2d(x, self.weight * self.w_mul, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        if self.bias is not None:
            bias = self.bias * self.b_mul
            x = x + bias.view(1, -1, 1, 1)
        return x

class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size, gain = 2**0.5, lrmul=1, bias=True):
        super().__init__()
        self.w_mul = lrmul
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
        return nn.functional.linear(x, self.weight * self.w_mul, bias)

class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 3, 1, 0),
                        groupNorm2d(channel*2), 
                        nn.GLU(dim=1),)
    def forward(self, noise):
        return self.init(noise)

class NoiseInjection(nn.Module):
    def __init__(self, channels, init_noise_w=0, requires_grad=True):
        super().__init__()
        self.weight = nn.Parameter(torch.full((channels,), init_noise_w, dtype=torch.float), requires_grad=requires_grad)
    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)
        return feat + self.weight[None,:,None, None] * noise
    
class AdaIN(nn.Module):
    def __init__(self, output_size, input_size):
        super(AdaIN, self).__init__()
        self.gamma_fc = linear(input_size, output_size)
        self.beta_fc = linear(input_size, output_size)
        self.instance_norm = nn.InstanceNorm2d(output_size)
    def forward(self, x, w):
        gamma = self.gamma_fc(w).view(x.size(0), -1, 1, 1)
        beta = self.beta_fc(w).view(x.size(0), -1, 1, 1)

        x_normalized = self.instance_norm(x)
        out = gamma * x_normalized + beta
        
        return out
    
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

#Block

class SEBlockCond(nn.Module):
    def __init__(self, ch_in, ch_out, nz=None):
        super().__init__()
        self.AdaAvgPool = nn.AdaptiveAvgPool2d(4)
        self.conv2d_2 = conv2d(ch_out, ch_out, 2, 1, 0)
        self.conv2d_3 = conv2d(ch_in, ch_out, 3, 1, 0)
        self.instance_norm = AdaIN(ch_out, nz)
    def forward(self, feat_small, feat_big, c, feature_injection=None):
        if feature_injection == None:

            x = self.AdaAvgPool(feat_small)
            x = self.conv2d_3(x)
            x = nn.SiLU()(x)
            
            x = self.instance_norm(x, c)
            x = self.conv2d_2(x)
            x = nn.Sigmoid()(x)

            return feat_big * x
        return feat_big * nn.Sigmoid()(feature_injection)

class UpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, nz, up_sample_mode='bilinear'):
        super().__init__()
        self.up_sample = UpSample(up_sample_mode)
        self.conv2d_1 = conv2d(in_planes, out_planes*2, 3, 1, 1)
        self.adain1 = AdaIN(out_planes*2, nz)
        self.noise1 = NoiseInjection(out_planes*2)
            
    def forward(self, x, w_noise, noise_injection=None):
        x = self.up_sample(x)

        x = self.conv2d_1(x)
        x = self.noise1(x, noise_injection)
        x = self.adain1(x, w_noise)
        x = nn.GLU(dim=1)(x)
        
        return x
    
class UpBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes, nz, up_sample_mode='bilinear'):
        super().__init__()
        self.up_sample = UpSample(up_sample_mode)
        self.conv2d_1 = conv2d(in_planes, out_planes*2, 3, 1, 1)
        self.conv2d_2 = conv2d(out_planes, out_planes*2, 3, 1, 1)
        self.adain1 = AdaIN(out_planes*2, nz)
        self.adain2 = AdaIN(out_planes*2, nz)
        self.noise1 = NoiseInjection(out_planes*2)
        self.noise2 = NoiseInjection(out_planes*2)
            
    def forward(self, x, w_noise, noise_injection=None):
        x = self.up_sample(x)

        x = self.conv2d_1(x)
        x = self.noise1(x, noise_injection)
        x = self.adain1(x, w_noise)
        x = nn.GLU(dim=1)(x)
        
        x = self.conv2d_2(x)
        x = self.noise2(x)
        x = self.adain2(x, w_noise)
        x = nn.GLU(dim=1)(x)
        
        return x

class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            groupNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1),
            groupNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

        self.res = nn.Sequential(
            DownSample(),
            conv2d(in_planes, out_planes, 1, 1, 0),
            groupNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True))
        self.scale = torch.rsqrt(torch.tensor(2.0))
    def forward(self, feat):
        return (self.main(feat) + self.res(feat)) * self.scale

# class DownBlockComp(nn.Module):
#     def __init__(self, in_planes, out_planes):
#         super(DownBlockComp, self).__init__()
#         self.main = nn.Sequential(
#             conv2d(in_planes, out_planes, 4, 2, 1),
#             groupNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
#             conv2d(out_planes, out_planes, 3, 1, 1),
#             groupNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.scale = torch.rsqrt(torch.tensor(2.0))

#         self.to_k     = conv2d(in_planes, 1, 1, 1, 0)
#         self.conv2d_1 = conv2d(in_planes, out_planes, 1, 1, 0)
#         self.conv2d_2 = conv2d(out_planes, out_planes, 1, 1, 0)
#         self.norm     = groupNorm2d(out_planes)
        
#     def forward(self, feat):
#         attention = self.to_k(feat).flatten(2).softmax(dim = -1)
#         global_context = einsum('b i n, b c n -> b c i', attention, feat.flatten(2)).unsqueeze(-1)
#         res = self.conv2d_1(global_context)
#         res = self.norm(res)
#         res = nn.LeakyReLU(0.2, inplace=True)(res)
#         res = self.conv2d_2(res)

#         return (self.main(feat) + res) * self.scale

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
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.mapping = nn.Sequential(
            linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2, inplace=True), 
            linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.2, inplace=True), 
            linear(embedding_dim, embedding_dim)
        )
    def forward(self, noise):
        mapping = self.mapping(noise)
        return mapping

class CondEmbedding(nn.Module):
    def __init__(self, n_classes, embedding_dim=128):
        super().__init__()
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes, embedding_dim)
        self.label_weight = nn.Parameter(torch.zeros(1), requires_grad=True)
    def forward(self, noise, c=None, c_training=False):
        #c = None for random conditions || c = (int) one condition for the batch || c = (tensor) for pre-defined conditions
        assert c is None or isinstance(c, int) or isinstance(c, torch.Tensor), "Invalid type for c"
        if c is None: c = torch.LongTensor(np.random.randint(0, self.n_classes, noise.shape[0]))
        elif type(c) is int: c = torch.full((noise.shape[0],), c, dtype=torch.long)
        c = c.to(noise.device)
        embeded = self.embed(c)

        if c_training: noisy_embedding = embeded * nn.Sigmoid()(self.label_weight) + noise
        else         : noisy_embedding = noise

        noisy_embedding = (noisy_embedding - noisy_embedding.mean())/noisy_embedding.std()
        return noisy_embedding

#GAN

class Generator(nn.Module):
    def __init__(self, n_classes=1, ngf=64, nz=512, nc=3, im_size=1024, cond_training=False):
        super(Generator, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)
        
        self.im_size = im_size
        self.n_classes = n_classes
        self.cond_training = cond_training
        self.emb = CondEmbedding(n_classes, embedding_dim=nz)
        self.mapping_network = MappingNetwork(embedding_dim=nz)

        self.init_constant = nn.Parameter(torch.randn((1, nz, 2, 2)))
        self.init     = InitLayer(nz, channel=nfc[4])

        self.feat_8   = UpBlockComp(nfc[4], nfc[8], nz)
        self.feat_16  = UpBlock(nfc[8], nfc[16], nz)
        self.feat_32  = UpBlockComp(nfc[16], nfc[32], nz)

        self.feat_64  = UpBlock(nfc[32], nfc[64], nz)
        self.feat_128 = UpBlockComp(nfc[64], nfc[128], nz)  
        self.feat_256 = UpBlock(nfc[128], nfc[256], nz)

        self.se_64  = SEBlockCond(nfc[4], nfc[64], nz)
        self.se_128 = SEBlockCond(nfc[8], nfc[128], nz)
        self.se_256 = SEBlockCond(nfc[16], nfc[256], nz)

        self.to_small = conv2d(nfc[128], nc, 1, 1, 0) 
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1)
        
        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512], nz) 
            self.se_512 = SEBlockCond(nfc[32], nfc[512], nz)
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024], nz)
        
    def forward(self, noise, c=None, layer_noise_injection=None, feature_injection=None):
        injection1 = {'noise_8':None,'noise_16':None,'noise_32':None,'noise_64':None,'noise_128':None,'noise_256':None}
        injection2 = {'se_64':None,'se_128':None,'se_256':None,'se_512':None, 'se_1024':None}
        if layer_noise_injection != None:
            for k in layer_noise_injection:
                injection1[k] = layer_noise_injection[k]
        if feature_injection != None:
            for k in feature_injection:
                injection2[k] = feature_injection[k]

        embed           = self.emb(noise, c, c_training=self.cond_training)
        mapped          = self.mapping_network(noise)
        init_constant   = self.init_constant.expand(noise.shape[0], -1, -1, -1) #(batch_size, c, h, w)

        feat_4          = self.init(init_constant)
        feat_8          = self.feat_8(feat_4, mapped, noise_injection=injection1['noise_8'])
        feat_16         = self.feat_16(feat_8, mapped, noise_injection=injection1['noise_16'])
        feat_32         = self.feat_32(feat_16, mapped, noise_injection=injection1['noise_32'])
        
        feat_64         = self.se_64 (feat_4, self.feat_64(feat_32, mapped, noise_injection=injection1['noise_64']), 
                                      embed, feature_injection=injection2['se_64'])
        
        feat_128        = self.se_128(feat_8, self.feat_128(feat_64, mapped, noise_injection=injection1['noise_128']), 
                                      embed, feature_injection=injection2['se_128'])
        
        feat_256        = self.se_256(feat_16, self.feat_256(feat_128, mapped, noise_injection=injection1['noise_256']), 
                                      embed, feature_injection=injection2['se_256'])
        if self.im_size == 256:
            return torch.tanh(self.to_big(feat_256))
            
        feat_512 = self.se_512( feat_32, self.feat_512(feat_256, mapped), 
                               embed, feature_injection=injection2['se_512'])
        if self.im_size == 512:
            return torch.tanh(self.to_big(feat_512))
            
        feat_1024 = self.feat_1024(feat_512, mapped)
        return torch.tanh(self.to_big(feat_1024))
 


class Discriminator(nn.Module):
    def __init__(self, n_classes, ndf=64, nz=100, nc=3, im_size=512, cond_training=False):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size
        self.nz = nz
        self.cond_training = cond_training

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        minibatch_std_offset = 1
        if im_size == 1024:
            self.down_from_big = nn.Sequential( DownBlockComp(nc, nfc[1024]),
                                                DownBlockComp(nfc[1024], nfc[512]))
        elif im_size == 512:
            self.down_from_big = nn.Sequential( DownBlockComp(nc, nfc[512]))
        elif im_size == 256:
            self.down_from_big = nn.Sequential( conv2d(nc, nfc[512], 3, 1, 1),
                                                nn.LeakyReLU(0.2, inplace=True),)

        self.emb = CondEmbedding(n_classes, embedding_dim=nz)

        self.down_4  = DownBlockComp(nfc[512], nfc[256])
        self.down_8  = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(conv2d(nfc[16]+minibatch_std_offset , nfc[8], 1, 1, 0),
                                    nn.LeakyReLU(0.2, inplace=True), 
                                    conv2d(nfc[8], 1, 4, 1, 0))

        self.se_2_16 = SEBlockCond(nfc[512], nfc[64], nz)
        self.se_4_32 = SEBlockCond(nfc[256], nfc[32], nz)
        self.se_8_64 = SEBlockCond(nfc[128], nfc[16], nz)
        
        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)

    def minibatch_std(self, x):
        batch_statistics = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        return torch.cat([x, batch_statistics], dim=1)
        
    def forward(self, imgs, c, get_aux=False):

        embed  = self.emb(torch.Tensor(imgs.shape[0], self.nz).normal_(0, 1).to(imgs.device), c, c_training=self.cond_training)

        feat_2 = self.down_from_big(imgs) 
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)
        
        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16, embed)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32, embed)
        
        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last, embed)
        feat_last_stats = self.minibatch_std(feat_last)
        
        out = self.rf_big(feat_last_stats).view(-1)

        if get_aux:
            rec_img_big = self.decoder_big(feat_last)

            part = np.random.randint(0, 4)
            if part==0: rec_img_part        = self.decoder_part(feat_32[:,:,:8,:8])
            if part==1: rec_img_part        = self.decoder_part(feat_32[:,:,:8,8:])
            if part==2: rec_img_part        = self.decoder_part(feat_32[:,:,8:,:8])
            if part==3: rec_img_part        = self.decoder_part(feat_32[:,:,8:,8:])
            return out, part, [rec_img_big, rec_img_part]
        
        return out