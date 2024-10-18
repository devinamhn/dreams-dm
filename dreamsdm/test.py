import torch
# from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import time
import h5py
from pathlib import Path
import pandas as pd
import os

import numpy as np
from pathlib import Path
from torchvision import transforms

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import wandb

from dreamsdm.models import ResNet18
from dreamsdm.datasets.wdmgal import WDMGalaxiesDataset, WDMGalaxies
from dreamsdm.utils import train, validate


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
# mu = 498244.
# sigma = 1235061.2500
img_dir_train =  "/mnt/ceph/users/dmohan/dreams/data/dreams/train/File_"
img_dir_val =  "/mnt/ceph/users/dmohan/dreams/data/dreams/val/File_"

# img_dir = "/tmp/dmtmp/train/File_"
transforms = transforms.Compose([transforms.ToTensor(),
                                # transforms.Normalize(mean = mu, std = sigma)
                                ])

trainset = WDMGalaxies(img_dir_train, n_files = 700, n_gal = 20) #700, 20
valset = WDMGalaxies(img_dir_val, n_files = 100, n_gal = 2) #100, 2
# testset = WDMGalaxiesDataset(img_dir, 'Test', transforms)
print(len(trainset), len(valset))

train_dataloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers = 64, prefetch_factor= 4, drop_last = True)
val_dataloader = DataLoader(valset, batch_size=32, shuffle=True, num_workers = 64, prefetch_factor= 4, drop_last = True)

model = ResNet18(img_dim = (1, 512, 512), output_size =1).to(device)
print(model)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

#training loop
epochs = 200
best_vloss = 1_000_000_000.
train_loss = np.zeros(epochs)
val_loss = np.zeros(epochs)
results_path = '/mnt/ceph/users/dmohan/dreams/results/one/' #'/mnt/ceph/users/dmohan/dreams/data/dreams/results'

wandb.init(
    project= "dreams",
    config = {
        "model": "ResNet18",
        "learning_rate":1e-3,
        "data%": "100%",
        # "weight_decay": weight_decay,
        "epochs": epochs,
    },
    name='WDMGal',
)

for epoch in range(epochs):
    model.train(True)
    avg_loss_train = train(model, optimizer, criterion, train_dataloader, device)
    train_loss[epoch] = avg_loss_train

    model.train(False)
    avg_loss_val = validate(model, criterion, val_dataloader, device)
    val_loss[epoch] = avg_loss_val

    if(epoch%10==0):
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'training loss': avg_loss_train,
                    'val loss': avg_loss_val
                    }, results_path+'modelckpt'+str(epoch)+'.pt')

    if avg_loss_val < best_vloss:
        best_vloss = avg_loss_val
        best_epoch = epoch
        torch.save(model.state_dict(), results_path +'best_model.pt')

    r_sq_train = 1 - (avg_loss_train/0.227)
    r_sq_val = 1 - (avg_loss_val/0.227)
    wandb.log({"train_loss":avg_loss_train, "validation_loss":avg_loss_val, "R_sq_train":r_sq_train, "R_sq_val":r_sq_val})

print("Finished training")
wandb.log({"best_vloss_epoch": best_epoch, "best_vloss": best_vloss})
# torch.save(train_loss, results_path+'/train_loss.pt')
# torch.save(val_loss, results_path +'/val_loss.pt')

print(train_loss)
print(val_loss)

plt.plot(np.arange(epochs), train_loss)
plt.plot(np.arange(epochs), val_loss)
plt.savefig(results_path + '/train_loss.png')