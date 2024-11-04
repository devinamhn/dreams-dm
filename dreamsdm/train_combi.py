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
from dreamsdm.datasets.wdmgal import WDMGalaxiesCombined
from dreamsdm.utils import train, validate, load_config

config_path = sys.argv[1]
config = load_config(config_path)
seed = config['seed']
name = config['name']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(seed)

#dataset
dataset_name = config['dataset']['type']
mu = eval(config['dataset']['mean'])
sigma = eval(config['dataset']['std'])
img_dir_train = config['dataset']['path']+ "/train/File_"
img_dir_val =  config['dataset']['path']+ "/val/File_"
train_n_files = config['dataset']['train_n_files']
train_n_gal = config['dataset']['train_n_galaxies']
val_n_files = config['dataset']['val_n_files']
val_n_gal = config['dataset']['val_n_galaxies']
pred_param = sys.argv[3] #config['dataset']['pred_param']

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
results_path = sys.argv[2] #config['results']['path']

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
    name=name+pred_param,
)

transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = mu, std = sigma)
                                ])

trainset = WDMGalaxiesCombined(img_dir_train, train_n_files, train_n_gal, transforms, pred_param)
valset = WDMGalaxiesCombined(img_dir_val, val_n_files, val_n_gal, transforms, pred_param)
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
    avg_loss_train, r_sq_train = train(model, optimizer, criterion, train_dataloader, device)

    model.train(False)
    avg_loss_val, r_sq_val = validate(model, criterion, val_dataloader, device)

    if(epoch%10==0):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training loss': avg_loss_train,
                    'val loss': avg_loss_val
                    }, results_path+'modelckpt'+str(epoch)+'.pt')
    
    scheduler.step(avg_loss_val)
    last_lr = scheduler.get_last_lr()[0]
    if avg_loss_val < best_vloss:
        best_vloss = avg_loss_val
        best_epoch = epoch
        torch.save(model.state_dict(), results_path +'best_model.pt')


    wandb.log({"train_loss":avg_loss_train, "validation_loss":avg_loss_val, "R_sq_train":r_sq_train, 
    "R_sq_val":r_sq_val, "best_vloss":best_vloss, "best_epoch":best_epoch})

print("Finished training")
wandb.log({"best_vloss_epoch": best_epoch, "best_vloss": best_vloss})
