import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
from torch import optim
import numpy as np
import torch.nn.functional as F
import wandb
from time import time


def train(vq_gen: nn.Module, 
            patch_disc: nn.Module,
            epochs: int, 
            lr: float,
            train_loader: DataLoader, 
            # val_loader: DataLoader,
            data_variance: float, 
            wandb_log= False):
    
    classification_loss = nn.BCELoss()

    gen_optim = optim.Adam(vq_gen,lr = lr)
    disc_optim = optim.Adam(patch_disc,lr = lr)

    train_loader = iter(train_loader)

    steps_per_epoch = len(train_loader) //train_loader.batch_size
    for epoch in range(epochs):
        start_time = time()
        for _ in tqdm(range(steps_per_epoch)):

            data,_ = next(iter(train_loader))

            ###Patch Disc loss ###
            _, data_recon, _ = vq_gen(data)

            disc_optim.zero_grad()

            fake_pred = patch_disc(data_recon)
            real_pred = patch_disc(data)

            fake_loss = classification_loss(fake_pred,torch.zeros(fake_pred.shape))
            real_loss = classification_loss(real_pred,torch.ones(real_pred.shape))

            disc_loss = fake_loss+real_loss
            disc_loss.backward()
            disc_loss.step()


            ###Generator Loss###
            data,_ = next(iter(train_loader))

            gen_optim.zero_grad()

            vq_loss, data_recon, perplexity = vq_gen(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            vae_loss = recon_error + vq_loss
            
            gan_loss = classification_loss(data_recon,torch.zeros(data_recon.shape))

            gen_loss = vae_loss + gan_loss
            gen_loss.backward()
            gen_optim.step()

            if wandb_log:
                wandb.log({
                    'reconstruction' : recon_error, 
                    'vq loss' : vq_loss,
                    'disc loss': disc_loss,
                    'fake disc loss': fake_loss,
                    'real disc loss': real_loss,
                    'gan loss': gan_loss, 
                    'perplexity': perplexity,
                    })

        end_time = time()
        print(f'Epoch {epoch} completed in {end_time-start_time}')
        