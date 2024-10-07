import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image 
from pathlib import Path

class WDMGalaxiesDataset(Dataset):
    """
    file_path: path to hdf file
    dataset_name: specify train or test datasets
    transform, target_transforms: optional transforms
    """
    
    def __init__(self, file_path, dataset_name, transform = None,
                 target_transform = None):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.transform = transform 
        self.target_transform = target_transform
        self.length = None
        self._idx_to_name = {} #data_dict 

        with h5py.File(self.file_path, 'r') as hf:
            for gname, group in hf.items():
                if gname == self.dataset_name:
                    sample_id_idx = 0
                    for sim_id, dd in (group.items()):
                        for Mgas_id, ee in enumerate(dd.items()):
                            self._idx_to_name[sample_id_idx] = [sim_id, ee[0]]
                            sample_id_idx+=1 
                    self.length = sample_id_idx
        print(self._idx_to_name)

    def __len__(self):
        assert self.length is not None
        return self.length
    
    def _open_hdf5(self):
        self._hf = h5py.File(self.file_path, 'r')

    def __getitem__(self, idx):
        # if (torch.is_tensor(idx)):
        #     idx = idx.tolist()
        if not hasattr(self, '_hf'):
            self._open_hdf5()

        sim_id, Mgas_id = self._idx_to_name[idx]
        data = self._hf[self.dataset_name][sim_id][Mgas_id]
        image = np.array(data)
        label = torch.tensor(data.attrs['WDM'])

        if self.transform:
            image = self.transform(image)

        # if self.target_transform:
            # label = self.target_transform(label)

        return image, label