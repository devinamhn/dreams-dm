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
        data = np.zeros((n_files*n_gal, 512, 512), dtype=np.float32)
        wdm = np.zeros((n_files*n_gal))
        
        for n_file in range(n_files):
            
            file_path_ = file_path + str(n_file) + '.hdf5'
            _hf = h5py.File(file_path_, 'r')
            
            hf_keys = list(_hf.keys())
            length= len(hf_keys)
            
            for i in range(length):
                data[i+n_file*n_gal,:,:] = _hf[hf_keys[i]]
                wdm[i+n_file*n_gal] =  _hf[hf_keys[i]].attrs['WDM']
                length_total+=1
                if(i== n_gal-1):
                    break
            _hf.close()
        data = data/255.0
        self.data = data#torch.tensor(data)
        self.labels = torch.tensor(wdm)
        self.length = length_total
        
    def __len__(self):
        assert self.length is not None
        return self.length

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
            # label = self.target_transform(label)
        
        return image, label


class WDMGalaxiesCombined(Dataset):
    """
    file_path: path to hdf file
    dataset_name: specify train or test datasets
    transform, target_transforms: optional transforms
    """
    
    def __init__(self, file_path, n_files, n_gal,
                 # dataset_name, 
                 transform = None,
                 target_transform = None, pred_param: str = 'WDM'):
        # self.file_path = file_path
        self.transform = transform 
        self.target_transform = target_transform

        length_total = 0
        data = np.zeros((n_files*n_gal, 2, 512, 512), dtype=np.float32) #np.zeros((n_files*n_gal, 1, 512, 512), dtype=np.float32)
        wdm = np.zeros((n_files*n_gal))
        
        for n_file in range(n_files):
            
            file_path_ = file_path + str(n_file) + '.hdf5'
            _hf = h5py.File(file_path_, 'r')
            
            hf_keys = list(_hf.keys())
            length= len(hf_keys)
            
            for i in range(length):
                # data[i+n_file*n_gal,:,:,:] = _hf[hf_keys[i]]
                data[i+n_file*n_gal,:,:,:] = _hf[hf_keys[i]]
                wdm[i+n_file*n_gal] =  _hf[hf_keys[i]].attrs[pred_param] #_hf[hf_keys[i]].attrs['WDM']
                length_total+=1
                if(i== n_gal-1):
                    break
            _hf.close()
            
        data = data/255.0
        self.data = data #torch.tensor(data)
        self.labels = torch.tensor(wdm)
        self.length = length_total
        
        
    def __len__(self):
        assert self.length is not None
        return self.length

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = np.reshape(image, (512, 512, -1))
            image = self.transform(image)
        # if self.target_transform:
            # label = self.target_transform(label)
        
        return image, label


class Galaxies(Dataset):
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
        data = np.zeros((n_files*n_gal, 512, 512), dtype=np.float32)
        wdm = np.zeros((n_files*n_gal, 4))
        
        for n_file in range(n_files):
            
            file_path_ = file_path + str(n_file) + '.hdf5'
            _hf = h5py.File(file_path_, 'r')
            
            hf_keys = list(_hf.keys())
            length= len(hf_keys)
            
            for i in range(length):
                data[i+n_file*n_gal,:,:] = _hf[hf_keys[i]]
                wdm[i+n_file*n_gal][0] =  _hf[hf_keys[i]].attrs['WDM']
                wdm[i+n_file*n_gal][1] =  _hf[hf_keys[i]].attrs['AGN']
                wdm[i+n_file*n_gal][2] =  _hf[hf_keys[i]].attrs['SN1']
                wdm[i+n_file*n_gal][3] =  _hf[hf_keys[i]].attrs['SN2']    
                length_total+=1
                if(i== n_gal-1):
                    break
            _hf.close()
        data = data/255.0
        self.data = data#torch.tensor(data)
        self.labels = torch.tensor(wdm)
        self.length = length_total
        
    def __len__(self):
        assert self.length is not None
        return self.length

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
            # label = self.target_transform(label)
        
        return image, label

class GalaxiesCombined(Dataset):
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
        data = np.zeros((n_files*n_gal, 2, 512, 512), dtype=np.float32) #np.zeros((n_files*n_gal, 1, 512, 512), dtype=np.float32)
        wdm = wdm = np.zeros((n_files*n_gal, 4))
        
        for n_file in range(n_files):
            
            file_path_ = file_path + str(n_file) + '.hdf5'
            _hf = h5py.File(file_path_, 'r')
            
            hf_keys = list(_hf.keys())
            length= len(hf_keys)
            
            for i in range(length):
                # data[i+n_file*n_gal,:,:,:] = _hf[hf_keys[i]]
                data[i+n_file*n_gal,:,:,:] = _hf[hf_keys[i]]
                wdm[i+n_file*n_gal][0] =  _hf[hf_keys[i]].attrs['WDM']
                wdm[i+n_file*n_gal][1] =  _hf[hf_keys[i]].attrs['AGN']
                wdm[i+n_file*n_gal][2] =  _hf[hf_keys[i]].attrs['SN1']
                wdm[i+n_file*n_gal][3] =  _hf[hf_keys[i]].attrs['SN2']   

                length_total+=1
                if(i== n_gal-1):
                    break
            _hf.close()
            
        data = data/255.0
        self.data = data #torch.tensor(data)
        self.labels = torch.tensor(wdm)
        self.length = length_total
        
        
    def __len__(self):
        assert self.length is not None
        return self.length

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            image = np.reshape(image, (512, 512, -1))
            image = self.transform(image)
        # if self.target_transform:
            # label = self.target_transform(label)
        
        return image, label


