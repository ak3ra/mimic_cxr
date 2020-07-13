import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms

pneumonia_frame = pd.read_csv('../output/pneumonia_images_and_labels.csv')
n = 5
img_name = pneumonia_frame.iloc[n,0]

print(img_name)
