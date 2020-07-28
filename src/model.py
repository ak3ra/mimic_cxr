import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
import torch.nn.Functional as F
from data import PneumoniaDataset
from torchvision import transforms

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    ])


dataset = PneumoniaDataset(csv_file="output/pneumonia_images_and_labels.csv", 
                        root_dir = "/scratch/akera/mmic_data/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
                        transform = my_transforms)


train_set, test_set = torch.utils.data.random_split(dataset, [2000,5000])

dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=1, shuffle=False)

