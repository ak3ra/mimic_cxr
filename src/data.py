from __future__ import print_function
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        #img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 4])
        #print(img_path)
        img_path = self.annotations.iloc[index, 5]
        #print(img_path)
        #print(self.annotations.iloc[index,5])
        image =io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,6]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
