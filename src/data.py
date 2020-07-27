from __future__ import print_function
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn


class LoadDataset(Dataset):
    """Loads the dataset and applies the relevant transforms"""
    def __init__(self, csv_file, root_dir, transform=None):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()

        img_name = os.path.join(self.root_dir,self.dataframe.iloc[idx, 3])
        image = io.imread(
