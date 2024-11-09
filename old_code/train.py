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
from copy import deepcopy
from pytorch_msssim import SSIM

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import current_process

import config
from models import Generator, Discriminator
from dataloader import DATASET, Multi_DataLoader
from diffaug import DiffAugment
from utils import *

parser = argparse.ArgumentParser(description="Process some integers.")

parser.add_argument('--dir'         , type=str           , help='Directory path', default='../../Data/fewshot/art-painting')
parser.add_argument('--n_workers'   , type=int           , help='Number of workers', default='2')
parser.add_argument('--master'      , type=int           , help='Master GPU', default='1')
parser.add_argument('--n_gpus'      , type=int           , help='Number of GPUs', default='2')

parser.add_argument('--z_dim'       , type=int           , help='Latent size', default='128')
parser.add_argument('--im_size'     , type=int           , help='Output images size', default='256')
parser.add_argument('--batch_size'  , type=int           , help='Batch size for each GPU', default='26')
parser.add_argument('--epoch'       , type=int           , help='Number of epochs', default='800')
parser.add_argument('--gen_lr'      , type=float         , help='Generator learning rate', default='2e-4')
parser.add_argument('--disc_lr'     , type=float         , help='Discriminator learning rate', default='2e-4')
parser.add_argument('--disc_update' , type=int           , help='Update discriminator more', default='1')
parser.add_argument('--cond'        , action='store_true', help='Start conditional training')

parser.add_argument('--snap'        , type=int           , help='Number of batches before save the weights. This option will overwrite old "checkpoint_fastgan.pt"', default='200')
parser.add_argument('--snap_epoch'  , type=int           , help='Number of epochs before save the weights. This option will create new saved file called "checkpoint_fastgan_epoch_n.pt"', default='2')
parser.add_argument('--snap_iter'   , type=int           , help='Use with inf_sampler. Save a new file in format "checkpoint_fastgan_iter_n.pt"', default='500')
parser.add_argument('--inf'         , action='store_true', help='No epoch! Just training. Recommend for small dataset. Use snap_iter to save weights.')#Note that using this option will not resample the undersampling classes. Check line "loader = Multi_DataLoader"
parser.add_argument('--start_iter'  , type=int           , help='Use with inf_sampler. Continue from iter n.', default='0')

args = parser.parse_args()

config.DEVICE                  = None
config.MASTER_DEVICE           = args.master
config.IMG_SIZE                = args.im_size
config.Z_DIM                   = args.z_dim
config.BATCH_SIZE              = args.batch_size
config.LEARNING_RATE_GENERATOR = args.gen_lr
config.LEARNING_RATE_CRITIC    = args.disc_lr
config.NUM_WORKERS             = args.n_workers
config.NUM_EPOCHS              = args.epoch
config.NUM_GPU                 = args.n_gpus
config.DISC_UPDATE             = args.disc_update
config.SNAP                    = args.snap
config.SNAP_EPOCH              = args.snap_epoch
config.SNAP_ITER               = args.snap_iter
config.CONDITIONAL_TRAINING    = args.cond
config.INF_SAMPLER             = args.inf
config.SAVE_MODEL              = True

#Setup multi GPUs

def Multi_GPU_Setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1245'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)  #For windows
    #dist.init_process_group("nccl", rank=rank, world_size=world_size) #For linux

#--------------------TRAIN FUNCTION--------------------

def train_fn(critic, gen, gen_ema, rec_ssim,
             loader, policy,
             opt_critic, opt_gen,
             total_images_processed, epoch,
             blur_init_sigma=2, blur_fade_kimg=100, 
             n_critic_updates=1, lazy_reg=16, ema_decay=0.999):
    current = current_process()
    loop = tqdm(loader, leave=True, position=current._identity[0] - 1, initial=args.start_iter)
    cum_l_critic = 0
    cum_l_gen    = 0
    cum_gp       = 0
    cum_l_rec    = 0
    t1           = 10*lazy_reg  #Gradient penalty multiplier

    for batch_idx, (real, c) in enumerate(loop):
        batch_idx += args.start_iter

        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]
        total_images_processed += cur_batch_size
        blur_sigma = max(1 - total_images_processed / (blur_fade_kimg * 1e3), 0) * blur_init_sigma if blur_fade_kimg > 1 else 0

        #Train critic
        critic.zero_grad()
        for i in range(n_critic_updates):
            if i+1 == n_critic_updates:#last loop for generator update
                noise = get_noise()
                fake  = gen(noise)
            else:
                with torch.no_grad():
                    noise = get_noise()
                    fake  = gen(noise)         
            fake_image = DiffAugment(fake, types=policy)
            real_image = DiffAugment(real, types=policy)

            #Fake loss
            critic_fake   = critic(blur_schedule(fake_image.detach(), blur_sigma))#
            l_critic_fake = torch.mean(F.softplus(critic_fake))

            l_critic_fake.backward()
            
            #Lazy reg
            if batch_idx % lazy_reg == 0 and i+1 == n_critic_updates:
                real_image.requires_grad_()
                critic_real, part, [rec_all, rec_part] = critic(blur_schedule(real_image, blur_sigma), get_aux=True)#
                #Gradient penalty
                gradients,    = torch.autograd.grad(outputs=critic_real,
                                                    inputs=real_image,
                                                    grad_outputs=critic_real.new_ones(critic_real.shape),
                                                    create_graph=True)
                gradient_penalty = gradients.reshape(cur_batch_size, -1).norm(2, dim=-1).pow(2).mean() * t1
            else:
                critic_real, part, [rec_all, rec_part] = critic(blur_schedule(real_image, blur_sigma), get_aux=True)#
                gradient_penalty = torch.zeros(1,device=config.DEVICE)
            
            #Reconstruction loss
            norm_rec_all     = (rec_all+ 1) / 2 #ssim loss requires img in range 0,1
            norm_rec_part    = (rec_part+ 1) / 2
            real_lowres      = (F.interpolate(real_image, rec_all.shape[2])+ 1) / 2
            real_lowres_crop = (F.interpolate(crop_image_by_part(real_image, part), rec_part.shape[2])+ 1) / 2

            l_rec =  1-rec_ssim(norm_rec_part, real_lowres_crop)    + 1-rec_ssim(norm_rec_all, real_lowres)
            
            #Real loss
            l_critic_real =  torch.mean(F.softplus(-critic_real)) +\
                             l_rec                                +\
                             gradient_penalty                                                     
                                                                                           
            l_critic_real.backward()
            opt_critic.step()

        #Train gen
        critic_fake = critic(blur_schedule(fake_image, blur_sigma))#
        l_gen       =  torch.mean(F.softplus(-critic_fake))

        l_gen.backward()
        opt_gen.step()
        gen.zero_grad()

        #Update gen_ema
        update_average(gen_ema, gen, ema_decay)

        #Gen examples and save models
        if config.INF_SAMPLER:
            if config.DEVICE == config.MASTER_DEVICE and batch_idx % config.SNAP_ITER == 0:
                true_params = get_copy_params(gen)
                update_average(gen, gen_ema, beta=0)#swap true params to avg params
                generate_examples(gen, batch_idx, seed=482002, batch_size=8, nrow=4, cus_name=f'img_{batch_idx}', dir='./saved_images/')
                load_params(gen, true_params)#back to true params
                save_image( torch.cat([
                            F.interpolate(real_image, rec_all.shape[-1]),
                            rec_all,
                            rec_part], dim=2)[:8].add(1).mul(0.5),
                            f'./saved_recs/rec_{batch_idx}.jpg')
            if config.DEVICE == config.MASTER_DEVICE and config.SAVE_MODEL and batch_idx % config.SNAP_ITER == 0 and batch_idx != 0:
                save_model(epoch, gen, gen_ema, critic, opt_gen, opt_critic, dir=f'./saved_model_weights/checkpoint_fastgan_iter_{batch_idx}.pt')
        else:
            if config.DEVICE == config.MASTER_DEVICE and batch_idx % 500 == 0:
                true_params = get_copy_params(gen)
                update_average(gen, gen_ema, beta=0)#swap true params to avg params
                generate_examples(gen, batch_idx, n=16)
                load_params(gen, true_params)#back to true params
                save_image( torch.cat([
                            F.interpolate(real_image, rec_all.shape[-1]),
                            rec_all,
                            rec_part], dim=2)[:8].add(1).mul(0.5),
                            f'./saved_recs/rec_{batch_idx}.jpg')
                
        if config.DEVICE == config.MASTER_DEVICE and config.SAVE_MODEL and (batch_idx+1) % config.SNAP == 0:
            save_model(epoch, gen, gen_ema, critic, opt_gen, opt_critic, dir='./saved_model_weights/checkpoint_fastgan.pt')

        #Display losses
        l_critic = l_critic_fake + (l_critic_real-l_rec-gradient_penalty)
        cum_l_critic     += l_critic.item()
        cum_l_gen        += l_gen.item()
        cum_l_rec        += l_rec.item()
        cum_gp           += gradient_penalty.item()
        
        loop.set_postfix(OrderedDict([
            ('l_gen', cum_l_gen/(batch_idx-args.start_iter+1)),
            ('l_critic', cum_l_critic/(batch_idx-args.start_iter+1)),
            ('l_rec', cum_l_rec/(batch_idx-args.start_iter+1)),
            ('gp', cum_gp/(batch_idx-args.start_iter+1)),
        ]))

#--------------------MAIN--------------------

def main(rank, world_size):
    config.DEVICE = rank
    print(f"Rank: {config.DEVICE}, Name: {torch.cuda.get_device_name(config.DEVICE)}")
    if config.DEVICE == config.MASTER_DEVICE: print(f"Master device: {torch.cuda.get_device_name(config.DEVICE)}")
    Multi_GPU_Setup(rank=rank, world_size=world_size)

    data = DATASET(directory=args.dir, augment=True)

    #num_classes = data.get_num_classes()
    data_size   = data.__len__()

    gen                 = Generator(ngf=64, z_dim=config.Z_DIM, im_size=config.IMG_SIZE).to(config.DEVICE)
    critic              = Discriminator(ndf=64, im_size=config.IMG_SIZE).to(config.DEVICE)

    opt_gen    = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE_GENERATOR, betas=(0.5, 0.999), eps=1e-8)
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE_CRITIC, betas=(0.5, 0.999), eps=1e-8)

    rec_ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)

    #Augmentation options
    policy = ['color', 'cutout', 'translation']#, 'offset_h'
    
    gen     = DDP(gen, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    critic  = DDP(critic, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # copy and initialize the gen_ema weights equal to the weights of gen
    gen_ema = deepcopy(gen)
    update_average(gen_ema, gen, beta=0) 

    #Load weights
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
        opt_gen.load_state_dict(checkpoint['opt_gen'])
        opt_critic.load_state_dict(checkpoint['opt_critic'])
        load_epoch = checkpoint['epoch']
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

        train_fn(critic, gen, gen_ema, rec_ssim,
                loader, policy,
                opt_critic, opt_gen,
                total_images_processed, epoch, 
                n_critic_updates=config.DISC_UPDATE)
        
        # Gen examples
        if config.DEVICE == config.MASTER_DEVICE:
            true_params = get_copy_params(gen)
            update_average(gen, gen_ema, beta=0)#swap true params to avg params
            generate_examples(gen, epoch, dir="./saved_epoch_images/", batch_size=8, nrow=4, seed=482002, isEpoch=True, cus_name=f"epoch_{epoch}")
            load_params(gen, true_params)#back to true params
        # Save model
        if config.DEVICE == config.MASTER_DEVICE and config.SAVE_MODEL:
            save_model(epoch, gen, gen_ema, critic, opt_gen, opt_critic, dir='./saved_model_weights/checkpoint_fastgan.pt')
            #Backup
            if epoch % config.SNAP_EPOCH == 0:
                save_model(epoch, gen, gen_ema, critic, opt_gen, opt_critic, dir=f'./saved_model_weights/checkpoint_fastgan_epoch_{epoch}.pt')

#----------------------------------------

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