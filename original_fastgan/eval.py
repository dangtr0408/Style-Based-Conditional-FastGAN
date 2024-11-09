import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm

from model import Generator


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img,size=256):
    return F.interpolate(img, size=size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--end_iter', type=int, default=4)

    parser.add_argument('--dist', type=str, default='./eval_images')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=16, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=512)
    parser.add_argument('--multiplier', type=int, default=5000, help='multiplier for model number')
    parser.set_defaults(big=False)
    args = parser.parse_args()
    
    if not os.path.exists(args.dist):
        os.makedirs(args.dist)

    noise_dim = 256
    device = torch.device('cuda:%d'%(args.cuda))
    
    net_ig = Generator( ngf=64, nz=noise_dim, nc=3, im_size=args.im_size)#, big=args.big )
    net_ig.to(device)

    for n in [args.multiplier*i for i in range(args.start_iter, args.end_iter+1)]:
        ckpt = f"./saved_model_weights/checkpoint_fastgan_iter_{n}.pt"
        checkpoint = torch.load(ckpt, map_location=lambda a,b: a)
        checkpoint['gen'] = {k.replace('module.', ''): v for k, v in checkpoint['gen'].items()}
        net_ig.load_state_dict(checkpoint['gen'])
        #load_params(net_ig, checkpoint['g_ema'])

        #net_ig.eval()
        print('load checkpoint success, iter %d'%n)

        net_ig.to(device)

        del checkpoint

        dist = f'./{args.dist}/eval_%d'%(n)
        dist = os.path.join(dist, 'img')
        os.makedirs(dist, exist_ok=True)

        with torch.no_grad():
            for i in tqdm(range(args.n_sample//args.batch)):
                noise = torch.randn(args.batch, noise_dim).to(device)
                g_imgs = net_ig(noise)
                g_imgs = resize(g_imgs,args.im_size) # resize the image using given dimension
                for j, g_img in enumerate( g_imgs ):
                    vutils.save_image(g_img.add(1).mul(0.5), 
                        os.path.join(dist, '%d.png'%(i*args.batch+j)))#, normalize=True, range=(-1,1))
