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
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 4])
        image =io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,5]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)

my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.0,0.0,0.0], std=[1.0, 1.0, 1.0])
    ])




pneumonia_dataset = PneumoniaDataset(csv_file="output/pneumonia_images_and_labels.csv", root_dir = "/scratch/akera/mmic_data/physionet.org/files/mimic-cxr-jpg/2.0.0/files", transform = my_transforms)

dataloader = torch.utils.data.DataLoader(dataset=pneumonia_dataset,batch_size=1, shuffle=False)

#for i, (images, labels) in enumerate(dataloader):
#    print(images.shape)
#    print(labels.shape)

img_num = 0
for _ in range(10):
    for img,label  in pneumonia_dataset:
        save_image(img, 'img'+str(img_num)+'.png')
        img_num +=1


