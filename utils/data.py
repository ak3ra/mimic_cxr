import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms
from PIL import Image

class PneumoniaDataset(Dataset):
    def __init__(self, csv_path):
        ## transforms
        self.to_tensor = transforms.ToTensor()
        # read csv file
        self.data_info = pd.read_csv('output/pneumonia_images_and_labels.csv', header=None)
        #images
        self.image_arr = np.asarray(self.data_info.iloc[:,4])
        #labels
        self.label_arr = np.asarray(self.data_info.iloc[:,5])
        # length of dataset
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        # transform to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        # get label
        single_image_label = self.label_arr[index]
        return (img_as_tensor, single_image_label)
    
    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    #import pdb; pdb.set_trace()
    data = PneumoniaDataset('../output/pneumonia_images_and_labels.csv')
    print(data)
