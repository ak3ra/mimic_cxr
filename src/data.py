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

## transformations to dataset
my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    ])


class PneumoniaDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, nrows=100)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 5]
        image =io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,6]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


dataset = PneumoniaDataset(csv_file="output/pneumonia_images_and_labels_modified.csv", 
                        root_dir = "/scratch/akera/mmic_data/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
                        transform = my_transforms)


batch_size = 8

train_set, test_set = torch.utils.data.random_split(dataset, [50,50])
train_loader = DataLoader(dataset=train_set,batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size, shuffle=True)


