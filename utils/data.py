import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

pneumonia_frame = pd.read_csv('output/pneumonia_images_and_labels.csv')

print(img_name)

def __init__(self, data, transform=None):
    self.image_frame = data
    self.transform = transform

def __len__(self, idx):
    return len(self.image_frame)

def __getitem__(self, idx):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    label = self.image_frame.loc[idx, 'Pneumonia']
    pic = Path(self.image_frame.loc[idx, 'image_path'])
    img = Image.open(pic)

    if self.transform:
        image = self.transform(img)