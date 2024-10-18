import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image 
from pathlib import Path
import h5py
import numpy as np
import torch
import h5py
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



class WDMGalaxies(Dataset):
    """
    file_path: path to hdf file
    dataset_name: specify train or test datasets
    transform, target_transforms: optional transforms
    """
    
    def __init__(self, file_path, n_files, n_gal,
                 # dataset_name, 
                 transform = None,
                 target_transform = None):
        # self.file_path = file_path
        self.transform = transform 
        self.target_transform = target_transform

        length_total = 0
        data = np.zeros((n_files*n_gal, 1, 512, 512), dtype=np.float32)
        wdm = np.zeros((n_files*n_gal))
        
        for n_file in range(n_files):
            
            file_path_ = file_path + str(n_file) + '.hdf5'
            _hf = h5py.File(file_path_, 'r')
            
            hf_keys = list(_hf.keys())
            length= len(hf_keys)
            
            for i in range(length):
                data[i+n_file*n_gal::] = _hf[hf_keys[i]]
                wdm[i+n_file*n_gal::] =  _hf[hf_keys[i]].attrs['WDM']
                length_total+=1
                if(i== n_gal-1):
                    break
            _hf.close()

        self.data = torch.tensor(data)
        self.labels = torch.tensor(wdm)
        self.length = length_total
        
    def __len__(self):
        assert self.length is not None
        return self.length

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(data)
        # if self.target_transform:
            # label = self.target_transform(label)
        
        return image, label


