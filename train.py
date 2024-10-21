import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
from collections import OrderedDict
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from diffaug import DiffAugment
import lpips
from copy import deepcopy

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import current_process

import config
from models import Generator, Discriminator
from dataloader import DATASET, Multi_DataLoader
from utils import *

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument('--dir', type=str, help='Directory path')
parser.add_argument('--n_workers', type=int, help='Number of workers', default='3')
parser.add_argument('--master', type=int, help='Master GPU', default='0')
parser.add_argument('--n_gpus', type=int, help='Number of GPUs', default='1')

parser.add_argument('--z_dim', type=int, help='Latent size', default='512')
parser.add_argument('--im_size', type=int, help='Output images size', default='512')
parser.add_argument('--batch_size', type=int, help='Batch size for each GPU', default='8')
parser.add_argument('--epoch', type=int, help='Number of epochs', default='800')
parser.add_argument('--gen_lr', type=float, help='Generator learning rate', default='2e-4')
parser.add_argument('--disc_lr', type=float, help='Discriminator learning rate', default='2e-4')
parser.add_argument('--disc_update', type=int, help='Update discriminator more', default='1')
parser.add_argument('--cond', action='store_true', help='Start conditional training')

parser.add_argument('--snap', type=int, help='Number of batches before save the weights. This option will overwrite old "checkpoint_fastgan.pt"', default='200')
parser.add_argument('--snap_epoch', type=int, help='Number of epochs before save the weights. This option will create new saved file called "checkpoint_fastgan_epoch_n.pt"', default='2')
parser.add_argument('--snap_iter', type=int, help='Use with inf_sampler. Save a new file in format "checkpoint_fastgan_iter_n.pt"', default='5000')
parser.add_argument('--inf', action='store_true', help='No epoch! Just training. Recommend for small dataset. Use snap_iter to save weights.')#Note that using this option will not resample the undersampling classes. Check line "loader = Multi_DataLoader"
parser.add_argument('--start_iter', type=int, help='Use with inf_sampler. Continue at iter n.', default='0')

args = parser.parse_args()

config.DEVICE = None
config.MASTER_DEVICE = args.master
config.IMG_SIZE = args.im_size
config.Z_DIM = args.z_dim
config.BATCH_SIZE = args.batch_size
config.LEARNING_RATE_GENERATOR = args.gen_lr
config.LEARNING_RATE_CRITIC = args.disc_lr
config.NUM_WORKERS = args.n_workers
config.NUM_EPOCHS = args.epoch
config.NUM_GPU = args.n_gpus
config.DISC_UPDATE = args.disc_update
config.SAVE_MODEL = True
config.SNAP = args.snap
config.SNAP_EPOCH = args.snap_epoch
config.SNAP_ITER = args.snap_iter
config.CONDITIONAL_TRAINING = args.cond
config.INF_SAMPLER = args.inf

#Setup multi GPUs

def Multi_GPU_Setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1245'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #dist.init_process_group("nccl", rank=rank, world_size=world_size)

#--------------------TRAIN FUNCTION--------------------

def train_fn(critic, gen, gen_ema, percept,
             loader, policy,
             opt_critic, opt_gen,
             total_images_processed, epoch,
             blur_init_sigma=1, blur_fade_kimg=20, 
             n_critic_updates=2, lazy_reg=16, ema_decay=0.999):
    current = current_process()
    loop = tqdm(loader, leave=True, position=current._identity[0] - 1, initial=args.start_iter)
    cum_loss_critic = 0
    cum_loss_gen = 0

    for batch_idx, (real, c) in enumerate(loop):
        batch_idx += args.start_iter

        real = real.to(config.DEVICE)
        c = c.to(config.DEVICE)
        cur_batch_size = real.shape[0]
        total_images_processed += cur_batch_size
        blur_sigma = max(1 - total_images_processed / (blur_fade_kimg * 1e3), 0) * blur_init_sigma if blur_fade_kimg > 1 else 0

        #Train critic
        critic.zero_grad()
        for i in range(n_critic_updates):
            if i+1 == n_critic_updates:#last loop for generator update
                noise = get_noise()
                fake  = gen(noise, c)
            else:
                with torch.no_grad():
                    noise = get_noise()
                    fake  = gen(noise, c)         
            fake_image    = DiffAugment(fake, types=policy)
            real_image    = DiffAugment(real, types=policy)

            #Fake
            critic_fake = critic(blur_schedule(fake_image.detach(), blur_sigma), c)#
            loss_critic_fake = (F.relu(torch.ones_like(critic_fake) + critic_fake)).mean()
            loss_critic_fake.backward()
            
            if batch_idx % lazy_reg == 0 and i+1 == n_critic_updates:
                real_image.requires_grad_()

            #Real
            critic_real, part, [rec_all, rec_part] = critic(blur_schedule(real_image, blur_sigma), c, get_aux=True)#
            real_lowres = F.interpolate(real_image, rec_all.shape[2])
            real_lowres_crop = F.interpolate(crop_image_by_part(real_image, part), rec_part.shape[2])

            #Gradient penalty
            if batch_idx % lazy_reg == 0 and i+1 == n_critic_updates:
                gradients, *_ = torch.autograd.grad(outputs=critic_real,
                                                    inputs=real_image,
                                                    grad_outputs=critic_real.new_ones(critic_real.shape),
                                                    create_graph=True)
                gradients = gradients.reshape(cur_batch_size, -1)
                norm = gradients.norm(2, dim=-1)
                gradient_penalty = torch.mean(norm ** 2)
            else: gradient_penalty = 0

            mse_loss = torch.nn.MSELoss()
            loss_critic_real =  (F.relu(torch.ones_like(critic_real) - critic_real)).mean()                                 +\
                                mse_loss(rec_all, real_lowres).sum()                                                        +\
                                mse_loss(rec_part, real_lowres_crop).sum()                                                  +\
                                percept(rec_all, real_lowres).sum()                                                         +\
                                percept(rec_part, real_lowres_crop).sum()                                                   +\
                                gradient_penalty                                                                          *4
                                                                                                 
            loss_critic_real.backward()
            opt_critic.step()

        #Train gen
        critic_fake = critic(blur_schedule(fake_image, blur_sigma), c)#
        loss_gen =  -torch.mean(critic_fake)
        loss_gen.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=10.)
        opt_gen.step()
        gen.zero_grad()

        #Update gen_ema
        update_average(gen_ema, gen, ema_decay)

        if config.INF_SAMPLER:
            if config.DEVICE == config.MASTER_DEVICE and batch_idx % config.SNAP_ITER == 0:
                true_params = get_copy_params(gen)
                update_average(gen, gen_ema, beta=0)#swap true params to avg params
                generate_examples(gen, batch_idx, seed=482002, batch_size=8, nrow=4, cus_name=f'img_{batch_idx}', dir='./saved_images/')
                save_image( torch.cat([
                            F.interpolate(real_image, rec_all.shape[-1]),
                            rec_all,
                            rec_part], dim=2)[:8].add(1).mul(0.5),
                            f'./saved_recs/rec_{batch_idx}.jpg')
                load_params(gen, true_params)#back to true params
            if config.DEVICE == config.MASTER_DEVICE and config.SAVE_MODEL and batch_idx % config.SNAP_ITER == 0 and batch_idx != 0:
                checkpoint = {
                        'epoch': epoch-1,
                        'gen_ema': gen_ema.state_dict(),
                        'gen': gen.state_dict(),
                        'critic':critic.state_dict(),
                        'opt_gen': opt_gen.state_dict(),
                        'opt_critic': opt_critic.state_dict(),
                    }
                torch.save(checkpoint, f'./saved_model_weights/checkpoint_fastgan_iter_{batch_idx}.pt')
        else:
            if config.DEVICE == config.MASTER_DEVICE and batch_idx % 100 == 0:
                generate_examples(gen, batch_idx, n=16)
                save_image( torch.cat([
                            F.interpolate(real_image, rec_all.shape[-1]),
                            rec_all,
                            rec_part], dim=2)[:8].add(1).mul(0.5),
                            f'./saved_recs/rec_{batch_idx}.jpg')
        if config.DEVICE == config.MASTER_DEVICE and config.SAVE_MODEL and (batch_idx+1) % config.SNAP == 0:
            checkpoint = {
                    'epoch': epoch-1,
                    'gen_ema': gen_ema.state_dict(),
                    'gen': gen.state_dict(),
                    'critic':critic.state_dict(),
                    'opt_gen': opt_gen.state_dict(),
                    'opt_critic': opt_critic.state_dict(),
                }
            torch.save(checkpoint, './saved_model_weights/checkpoint_fastgan.pt')

        loss_critic = loss_critic_fake + loss_critic_real
        cum_loss_critic     += loss_critic.item()
        cum_loss_gen        += loss_gen.item()
        loop.set_postfix(
            l_critic=cum_loss_critic/(batch_idx+1),
            l_gen   =cum_loss_gen/(batch_idx+1),
        )

#--------------------MAIN--------------------

def main(rank, world_size):
    config.DEVICE = rank
    print(f"Rank: {config.DEVICE}, Name: {torch.cuda.get_device_name(config.DEVICE)}")
    if config.DEVICE == config.MASTER_DEVICE: print(f"Master device: {torch.cuda.get_device_name(config.DEVICE)}")
    Multi_GPU_Setup(rank=rank, world_size=world_size)

    data = DATASET(directory=args.dir, augment=True)

    num_classes = data.get_num_classes()
    data_size   = data.__len__()

    gen                 = Generator(num_classes, ngf=64, nz=config.Z_DIM, im_size=config.IMG_SIZE, cond_training=config.CONDITIONAL_TRAINING).to(config.DEVICE)
    critic              = Discriminator(num_classes, ndf=64, nz=config.Z_DIM, im_size=config.IMG_SIZE).to(config.DEVICE)
    
    #diff lr for mapping network
    parameters = []
    match_name = ['mapping_network','mapping']
    for idx, (name, param) in enumerate(gen.named_parameters()):
        if param.requires_grad:
            if name.split(".")[0] in match_name:
                #print(f"found {name}")
                parameters += [{'params': [param], 'lr': 0.001}]
            else: parameters += [{'params': [param]}]
    # for idx, (name, param) in enumerate(critic.named_parameters()):
    #     if param.requires_grad:
    #         if name.split(".")[0] in match_name:
    #             #print(f"found {name}")
    #             parameters += [{'params': [param], 'lr': 0.001}]
    #         else: parameters += [{'params': [param]}]

    opt_gen             = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5, 0.99), eps=1e-8)
    opt_critic          = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE_CRITIC, betas=(0.5, 0.99), eps=1e-8)
    percept = lpips.LPIPS(net='vgg').to(config.DEVICE)
    policy = ['color', 'offset_h', 'cutout', 'translation']

    gen     = DDP(gen, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    critic  = DDP(critic, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    gen_ema = deepcopy(gen)
    # initialize the gen_ema weights equal to the weights of gen
    update_average(gen_ema, gen, beta=0)

    load_epoch = 0
    dir = './saved_model_weights/checkpoint_fastgan.pt'
    if os.path.exists(dir):
        checkpoint = torch.load(dir, map_location="cpu")
        # Add "module" prefix for non-ddp saved files
        model_dict = {'gen':OrderedDict(),'gen_ema':OrderedDict(),'critic':OrderedDict()}
        for key in checkpoint.keys():
            if key in ['gen', 'gen_ema', 'critic']:
                for k,v in checkpoint[key].items():
                    if not k.startswith('module.'):
                        new_key = 'module.' + k
                        model_dict[key][new_key] = v
                    else:
                        model_dict[key][k] = v
        gen.load_state_dict(model_dict['gen'], strict=False)
        gen_ema.load_state_dict(model_dict['gen_ema'], strict=False)
        critic.load_state_dict(model_dict['critic'], strict=False)
        load_epoch = checkpoint['epoch']
        opt_gen.load_state_dict(checkpoint['opt_gen'])
        opt_critic.load_state_dict(checkpoint['opt_critic'])
        del checkpoint
        del model_dict
        torch.cuda.empty_cache()
    else:
        if config.DEVICE == config.MASTER_DEVICE: print(f"\nThere is no saved file at {dir}. Start training a new model...")

    gen.train()
    gen_ema.train()
    critic.train()

    if config.DEVICE == config.MASTER_DEVICE:
        gen_total_params = sum(p.numel() for p in gen.parameters())
        critic_total_params = sum(p.numel() for p in critic.parameters())
        print("\n")
        print(f"Gen parameters: {gen_total_params}")
        print(f"Critic parameters: {critic_total_params}")
        print("\n")

    for epoch in range(load_epoch+1, config.NUM_EPOCHS):
        if config.DEVICE == config.MASTER_DEVICE: print(f"\nEpoch: {epoch}")
        # re-sample for under-sampling classes
        loader = Multi_DataLoader(data, rank=rank, world_size=world_size, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)
        if not config.INF_SAMPLER:
            loader.sampler.set_epoch(epoch)
        total_images_processed = (epoch-1)*data_size

        train_fn(critic, gen, gen_ema, percept,
                loader, policy,
                opt_critic, opt_gen,
                total_images_processed, epoch, 
                n_critic_updates=config.DISC_UPDATE)
        # Gen examples
        if config.DEVICE == config.MASTER_DEVICE:
            true_params = get_copy_params(gen)
            update_average(gen, gen_ema, beta=0)#swap true params to avg params
            for i in range(num_classes):
                generate_examples(gen, epoch, dir=f"./saved_epoch_images/{data.class_names[i]}", c_in=i, batch_size=8, nrow=4, seed=482002, isEpoch=True, cus_name=f"{data.class_names[i]}_{epoch}")
            load_params(gen, true_params)#back to true params
        # Save model
        if config.DEVICE == config.MASTER_DEVICE and config.SAVE_MODEL:
            checkpoint = {
                'epoch': epoch,
                'gen_ema': gen_ema.state_dict(),
                'gen': gen.state_dict(),
                'critic':critic.state_dict(),
                'opt_gen': opt_gen.state_dict(),
                'opt_critic': opt_critic.state_dict(),
            }
            torch.save(checkpoint, './saved_model_weights/checkpoint_fastgan.pt')
            #Backup
            if epoch % config.SNAP_EPOCH == 0: 
                torch.save(checkpoint, f'./saved_model_weights/checkpoint_fastgan_epoch_{epoch}.pt')












if __name__ == '__main__':
    print("CONDITIONAL TRAINING:",config.CONDITIONAL_TRAINING)
    if torch.cuda.is_available():   print("Using GPU")
    else:                           print("No GPU")
    create_dir_list = ["saved_model_weights","saved_images","saved_recs"]
    for dir in create_dir_list:
        if not os.path.exists(dir):
            os.makedirs(dir)

    world_size = config.NUM_GPU

    mp.spawn(main, args=(world_size,), nprocs=world_size)
