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
import re

from models import Generator


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img,size=256):
    return F.interpolate(img, size=size)

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='generate images'
    )
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')

    parser.add_argument('--dist', type=str, default='./eval_images')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', default=64, type=int, help='batch size')
    parser.add_argument('--n_sample', type=int, default=10000)
    parser.add_argument('--big', action='store_true')
    parser.add_argument('--im_size', type=int, default=256)
    parser.set_defaults(big=False)
    args = parser.parse_args()
    
    if not os.path.exists(args.dist):
        os.makedirs(args.dist)

    noise_dim = 128
    device = torch.device('cuda:%d'%(args.cuda))
    
    net_ig = Generator( ngf=64, z_dim=noise_dim, nc=3, im_size=args.im_size)
    net_ig.to(device)

    checkpoint_dir = f"./saved_model_weights/"
    n=0
    for i, filename in enumerate(sorted(os.listdir(checkpoint_dir), key=extract_number)[n:]):#start from n checkpoint
        checkpoint = torch.load(checkpoint_dir+filename, map_location=lambda a,b: a)
        checkpoint['gen_ema'] = {k.replace('module.', ''): v for k, v in checkpoint['gen'].items()}
        net_ig.load_state_dict(checkpoint['gen_ema'], strict=False)
        #load_params(net_ig, checkpoint['g_ema'])

        #net_ig.eval()
        print(f'loaded checkpoint {filename}')

        net_ig.to(device)

        del checkpoint

        dist = f'./{args.dist}/eval_{filename}'
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
