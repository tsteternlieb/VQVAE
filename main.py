
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from VQ_VAE_layers import VQ_VAE
from train import train

num_hidden = 1,
num_residual_hiddens = 1,
num_embeddings = 1,
embedding_dim = 1,
commitment_cost = 1

epochs = 1
lr = 1
train_loader = 1
val_loader = 1
data_variance = 1

vq_vae = VQ_VAE(num_hidden, num_residual_hiddens,num_embeddings,embedding_dim,commitment_cost)


train(vq_vae,epochs,lr,train_loader,val_loader,data_variance,wandb_log=False)