import torch
# from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import time
import h5py
from pathlib import Path
import pandas as pd
import os
import sys

import numpy as np
from pathlib import Path
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import wandb

from dreamsdm.models import ResNet18
from dreamsdm.datasets.wdmgal import WDMGalaxies, Galaxies
from dreamsdm.utils import train_all, validate_all, load_config

config_path = sys.argv[1]
config = load_config(config_path)
seed = config['seed']
name = config['name']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(seed)

#dataset
dataset_name = config['dataset']['type']
mu = config['dataset']['mean']
sigma = config['dataset']['std']
img_dir_train = config['dataset']['path']+ "/train/File_"
img_dir_val =  config['dataset']['path']+ "/val/File_"
train_n_files = config['dataset']['train_n_files']
train_n_gal = config['dataset']['train_n_galaxies']
val_n_files = config['dataset']['val_n_files']
val_n_gal = config['dataset']['val_n_galaxies']

#training
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
num_workers = config['training']['num_workers']
prefetch_factor = config['training']['prefetch_factor']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
patience = config['training']['patience']
factor = config['training']['factor']

#model
input_shape = config['model']['input_shape']
output_size = config['model']['output_size']

#results
results_path =  config['results']['path']

wandb.init(
    project= "dreams",
    config = {
        "model": config['model']['architecture'],
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "factor": factor,
        "epochs": epochs,
        "dataset": dataset_name,
        "seed": seed,
    },
    name=name,
)

transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = mu, std = sigma)
                                ])

trainset = Galaxies(img_dir_train, train_n_files, train_n_gal, transforms)
valset = Galaxies(img_dir_val, val_n_files, val_n_gal, transforms)
print(len(trainset), len(valset))

train_dataloader = DataLoader(trainset, batch_size, shuffle=True, num_workers = num_workers, prefetch_factor= prefetch_factor, drop_last = True)
val_dataloader = DataLoader(valset, batch_size, shuffle=False, num_workers = num_workers, prefetch_factor= prefetch_factor, drop_last = True)

model = ResNet18(img_dim = input_shape, output_size = output_size).to(device)
criterion = nn.MSELoss()
learning_rate = learning_rate
weight_decay = weight_decay
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience = patience, factor = factor)


best_vloss = 1_000_000_000.0


for epoch in range(epochs):
    model.train(True)
    avg_loss_train, r_sq_train = train_all(model, optimizer, criterion, train_dataloader, device)

    model.train(False)
    avg_loss_val, r_sq_val = validate_all(model, criterion, val_dataloader, device)

    if(epoch%10==0):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training loss': avg_loss_train,
                    'val loss': avg_loss_val
                    }, results_path+'modelckpt'+str(epoch)+'.pt')
    
    scheduler.step(avg_loss_val)
    # last_lr = scheduler.get_last_lr()[0]
    if avg_loss_val < best_vloss:
        best_vloss = avg_loss_val
        best_epoch = epoch
        torch.save(model.state_dict(), results_path +'best_model.pt')


    wandb.log({"train_loss":avg_loss_train, "validation_loss":avg_loss_val, 
    "R_sq_train_WDM":r_sq_train[0],
    "R_sq_train_AGN":r_sq_train[1],
    "R_sq_train_SN1":r_sq_train[2],
    "R_sq_train_SN2":r_sq_train[3], 
    "R_sq_val_WDM":r_sq_val[0],
    "R_sq_val_AGN":r_sq_val[1],
    "R_sq_val_SN1":r_sq_val[2],
    "R_sq_val_SN2":r_sq_val[3], 
   })

print("Finished training")
wandb.log({"best_vloss_epoch": best_epoch, "best_vloss": best_vloss})
