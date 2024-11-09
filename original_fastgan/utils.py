import os
import numpy as np
from scipy.stats import truncnorm

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from copy import deepcopy
import config

#Utils
def update_average(model_tgt, model_src, beta):
    toggle_grad(model_tgt, False)
    toggle_grad(model_src, False)

    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert (p_src is not p_tgt)
        p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

    toggle_grad(model_tgt, True)
    toggle_grad(model_src, True)

def get_copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def crop_image_by_part(image, part):
    height, width = image.shape[2], image.shape[3]
    hw = height // 2

    if part == 0:
        return image[:, :, :hw, :hw]
    elif part == 1:
        return image[:, :, :hw, hw:]
    elif part == 2:
        return image[:, :, hw:, :hw]
    elif part == 3:
        return image[:, :, hw:, hw:]

def get_truncated_noise(shape, truncation):
    truncated_noise = truncnorm.rvs(-truncation, truncation, size=shape)
    return torch.Tensor(truncated_noise)

def get_noise():
    return torch.Tensor(config.BATCH_SIZE, config.Z_DIM).normal_(0, 1).to(config.DEVICE)

def generate_examples(gen, iter, dir=None, n=0, batch_size=1, trunc=2, nrow=0, seed=None, cus_name=None, c_in=None, isEpoch=False):
    if dir is None:
        if isEpoch: dir = f"./saved_epoch_images/epoch_{iter}"
        else: dir = f"./saved_images/iter_{iter}"
    if not os.path.exists(dir):
        os.makedirs(dir)
    for i in range(n+1):
        with torch.no_grad():
            if seed is not None : noise = torch.FloatTensor(batch_size, config.Z_DIM).normal_(0, 1, generator=torch.manual_seed(seed)).to(config.DEVICE)
            else: noise = get_truncated_noise((batch_size, config.Z_DIM), trunc).to(config.DEVICE)
                
            img = gen(noise, c_in)*0.5+0.5
            filename = f"/{cus_name}.png" if cus_name else f"/img_{i}.png"
            if nrow == 0: save_image(img, dir + filename)
            else        : save_image(img, dir + filename, nrow=nrow)
def blur_schedule(img, blur_sigma, kernel=5):
    blur_size = np.floor(blur_sigma * 3)
    if blur_size > 0:
        img = transforms.GaussianBlur(kernel, blur_sigma)(img)
    return img.to(config.DEVICE)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_model(epoch, gen, gen_ema, critic, opt_gen, opt_critic, dir):
    checkpoint = {
        'epoch': epoch-1,
        'gen_ema': gen_ema.state_dict(),
        'gen': gen.state_dict(),
        'critic':critic.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'opt_critic': opt_critic.state_dict(),
    }
    torch.save(checkpoint, dir)