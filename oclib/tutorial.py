import os
from glob import glob
from collections import deque
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.io import read_image


class MyDataset2(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform
        
        self.img_files = glob(f'{os.path.abspath(img_path)}/*')
        self.dname = os.path.basename(img_path)
            
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self,idx):
        img = read_image(self.img_files[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.dname

class MyDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform
        
        img_dirs = glob(f'{os.path.abspath(img_path)}/*')
        self.dsets = deque()
        for path in img_dirs:
            dname = path.split('/')[-1]
            self.dsets.extend([(img_path, dname) for img_path in path])
            
    def __len__(self):
        return len(self.dsets)
    
    def __getitem__(self,idx):
        img_path, label = self.dsets[idx]
        img = read_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label