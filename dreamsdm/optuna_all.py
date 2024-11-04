import torch
import torch.nn as nn
import sys

import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dreamsdm.models import ResNet18
from dreamsdm.datasets.wdmgal import GalaxiesCombined
from dreamsdm.utils import train_all, validate_all, load_config

import optuna 
from optuna.integration.wandb import WeightsAndBiasesCallback

config_path = sys.argv[1]
config = load_config(config_path)
seed = config['seed']
name = 'Optim' + config['name']
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

#training
epochs = 200 #config['training']['epochs']
batch_size = config['training']['batch_size']
num_workers = config['training']['num_workers']
prefetch_factor = config['training']['prefetch_factor']

#FIND BY OPTIMIZATION
# learning_rate = config['training']['learning_rate']
# weight_decay = config['training']['weight_decay']
# patience = config['training']['patience']
# factor = config['training']['factor']

#model
input_shape = config['model']['input_shape']
output_size = config['model']['output_size']

results_path =  config['results']['path']


transforms = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = mu, std = sigma)
                                ])

trainset = GalaxiesCombined(img_dir_train, train_n_files, train_n_gal, transforms)
valset = GalaxiesCombined(img_dir_val, val_n_files, val_n_gal, transforms)
print(len(trainset), len(valset))


def objective(trial):

    model = ResNet18(img_dim = input_shape, output_size = output_size).to(device)
    criterion = nn.MSELoss()

    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("wd", 1e-5, 1e-2, log=True)
    patience = trial.suggest_int("patience", 2, 10)
    factor = trial.suggest_float("factor", 0.1, 0.9)

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience = patience, factor = factor)


    train_dataloader = DataLoader(trainset, batch_size, shuffle=True, num_workers = num_workers, prefetch_factor= prefetch_factor, drop_last = True)
    val_dataloader = DataLoader(valset, batch_size, shuffle=False, num_workers = num_workers, prefetch_factor= prefetch_factor, drop_last = True)  
    
    for epoch in range(epochs):
        model.train(True)
        avg_loss_train, r_sq_train = train_all(model, optimizer, criterion, train_dataloader, device)

        model.train(False)
        avg_loss_val, r_sq_val = validate_all(model, criterion, val_dataloader, device)

        scheduler.step(avg_loss_val)

        trial.report(avg_loss_val, epoch)
        print(f"Saving a checkpoint in epoch {epoch}.")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss_train,
                "r_sq_train": r_sq_train,
                "val_loss": avg_loss_val,
                "r_sq_val": r_sq_val,
            },
            results_path+f"tmp_model_{trial.number}.pt",
        )
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_loss_val


if __name__ == "__main__":
        
    study = optuna.create_study(direction="minimize")
    # wandb_kwargs = {"project": "dreams",
    #                 "dataset": dataset_name,
    #                 "name": name
    # }
    # wandbc = WeightsAndBiasesCallback(wandb_kwargs)
    study.optimize(objective, n_trials=100) #, callbacks = wandbc)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))