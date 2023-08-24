import os
from glob import glob
from collections import deque

import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.io import read_image

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

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