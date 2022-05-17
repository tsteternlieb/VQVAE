import torch
from torch import nn
from torch.utils.data import DataLoader
import tqdm
from torch import optim
import numpy as np
import torch.nn.functional as F
import wandb
from time import time


def train(model: nn.Module, 
            epochs: int, 
            lr: float,
            train_loader: DataLoader, 
            val_loader: DataLoader,
            data_variance: float, 
            wandb_log= False):

    optimizer = optim.Adam(model,lr = lr)
    train_loader = iter(train_loader)

    steps_per_epoch = len(train_loader) //train_loader.batch_size
    for epoch in range(epochs):
        start_time = time()
        for _ in tqdm(range(steps_per_epoch)):
            data,_ = next(iter(train_loader))

            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            loss = recon_error + vq_loss
            loss.backward()
            optimizer.step()

            if wandb_log:
                wandb.log({'reconstruction' : recon_error, 'vq_loss' : vq_loss, 'perplexity': perplexity})

        end_time = time()
        val = 100
        print(f'Epoch {epoch} completed in {end_time-start_time}')
        