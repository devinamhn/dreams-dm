import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image 
from pathlib import Path

class WDMGalaxiesDataset(Dataset):
    """
    
    """
    
    def __init__(self, csv_file, img_dir, transform = None,
                 target_transform = None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform 
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label
