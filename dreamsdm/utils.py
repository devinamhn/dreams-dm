from pathlib import Path
import torch
# from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

import h5py
from pathlib import Path
import pandas as pd
import os
import yaml
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

class Path_Handler:
    """Handle and generate paths in project directory"""

    def __init__(
        self, **kwargs
    ):  # use defaults except where specified in kwargs e.g. Path_Handler(data=some_alternative_dir)
        path_dict = {}
        path_dict["root"] = kwargs.get("root", Path(__file__).resolve().parent.parent.parent)
        path_dict["project"] = kwargs.get(
            "project", Path(__file__).resolve().parent.parent
        )  # i.e. this repo

        # separate data paths / make local copies into a single data folder
        # path_dict["data"] = kwargs.get("data", path_dict["project"] / "data")
        
        # path_dict["models"] = kwargs.get("inference", path_dict["project"] / "models")
        # path_dict["eval"] = kwargs.get("eval", path_dict["project"] / "eval")

        for key, path_str in path_dict.copy().items():
            path_dict[key] = Path(path_str)

        self.path_dict = path_dict

    def fill_dict(self):
        """Create dictionary of required paths"""

        # self.path_dict["rgz"] = self.path_dict["data"] / "rgz"
        # self.path_dict["mb"] = self.path_dict["data"] / "mb"
        # self.path_dict["mightee"] = self.path_dict["data"] / "MIGHTEE"

    def create_paths(self):
        """Create missing directories"""
        for path in self.path_dict.values():
            create_path(path)

    def _dict(self):
        """Generate path dictionary, create any missing directories and return dictionary"""
        self.fill_dict()
        self.create_paths()
        return self.path_dict


def create_path(path):
    if not Path.exists(path):
        Path.mkdir(path)

def train(model, optimizer, criterion, train_loader, device):

    running_loss = 0.0
    y_pred = []
    y_true = []
    for i, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        outputs = model(x_train).flatten()
        y_pred.append(outputs.to(torch.double))
        y_true.append(y_train)
        loss = criterion(outputs.to(torch.double), y_train)
        # print(outputs.to(torch.double).dtype, y_train.dtype)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    r2_ = r2_score(torch.cat(y_true).cpu().detach().numpy(), torch.cat(y_pred).cpu().detach().numpy())
 
    return running_loss/(i+1), r2_

def validate(model, criterion, validation_loader, device):
    running_loss = 0.0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(validation_loader):
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val).flatten()
            y_pred.append(outputs.to(torch.double))
            y_true.append(y_val)
            loss = criterion(outputs.to(torch.double), y_val)
            running_loss += loss.item()
    r2_ = r2_score(torch.cat(y_true).cpu().detach().numpy(), torch.cat(y_pred).cpu().detach().numpy())
    return running_loss/(i+1), r2_

# Load configuration from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_all(model, optimizer, criterion, train_loader, device):

    running_loss = 0.0
    y_pred = []
    y_true = []
    for i, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        outputs = model(x_train)
        y_pred.append(outputs.to(torch.double))
        y_true.append(y_train)
        loss = criterion(outputs.to(torch.double), y_train)
        # print(outputs.to(torch.double).dtype, y_train.dtype)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    r2_ = r2_score(torch.cat(y_true).cpu().detach().numpy(), torch.cat(y_pred).cpu().detach().numpy(), multioutput = 'raw_values' )
 
    return running_loss/(i+1), r2_

def validate_all(model, criterion, validation_loader, device):
    running_loss = 0.0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for i, (x_val, y_val) in enumerate(validation_loader):
            x_val, y_val = x_val.to(device), y_val.to(device)
            outputs = model(x_val)
            y_pred.append(outputs.to(torch.double))
            y_true.append(y_val)
            loss = criterion(outputs.to(torch.double), y_val)
            running_loss += loss.item()
    r2_ = r2_score(torch.cat(y_true).cpu().detach().numpy(), torch.cat(y_pred).cpu().detach().numpy(), multioutput = 'raw_values')
    return running_loss/(i+1), r2_

