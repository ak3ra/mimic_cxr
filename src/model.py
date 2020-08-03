import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from data import PneumoniaDataset
from torchvision import transforms
import time

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 8,
            kernel_size = (3,3),
            stride=(1,1),
            padding = (1,1)
        )

        self.pool = nn.MaxPool2d(kernel_size =  (2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(
            in_channels = 8,
            out_channels = 16,
            kernel_size = (3,3),
            stride = (1,1),
            padding = (1,1)

        )

        self.fc1 = nn.Linear(50176, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x

class Convnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=3)

        
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)
        
#         print(X.shape)
        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        #X = F.log_softmax(X)
        X = self.fc3(X)
        
#         X = torch.sigmoid(X)
        return X



my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    ])

## Hyper Parameters

batch_size = 4
num_classes = 3
learning_rate = 1e-3
num_epochs = 2
in_channel = 1

dataset = PneumoniaDataset(csv_file="output/pneumonia_images_and_labels_modified.csv", 
                        root_dir = "/scratch/akera/mmic_data/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
                        transform = my_transforms)


train_set, test_set = torch.utils.data.random_split(dataset, [50,50])

train_loader = DataLoader(dataset=train_set,batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size, shuffle=True)

import pdb;pdb.settrace() 

model = Convnet().to(device)

losses = []
accuracies = []
epoches = 50
start = time.time()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(epoches):
    epoch_loss = 0
    epoch_accuracy = 0
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        preds = model(X)
        loss = loss_fn(preds, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        accuracy = ((preds.argmax(dim=1) == y).float().mean())
        epoch_accuracy += accuracy
        epoch_loss += loss
        print('.', end='', flush=True)
        
    epoch_accuracy = epoch_accuracy/len(train_loader)
    accuracies.append(epoch_accuracy)
    epoch_loss = epoch_loss / len(train_loader)
    losses.append(epoch_loss)
    print("Epoch: {}, train loss: {:.4f}, train accracy: {:.4f}, time: {}".format(epoch, epoch_loss, epoch_accuracy, time.time() - start))


    with torch.no_grad():
        val_epoch_loss = 0
        val_epoch_accuracy = 0
        for val_X, val_y in test_loader:
            val_X = val_X.to(device)
            val_y = val_y.to(device)
            val_preds = model(val_X)
            val_loss = loss_fn(val_preds, val_y)

            val_epoch_loss += val_loss            
            val_accuracy = ((val_preds.argmax(dim=1) == val_y).float().mean())
            val_epoch_accuracy += val_accuracy
        val_epoch_accuracy = val_epoch_accuracy/len(test_loader)
        val_epoch_loss = val_epoch_loss / len(test_loader)
        print("Epoch: {}, valid loss: {:.4f}, valid accracy: {:.4f}, time: {}\n".format(epoch, val_epoch_loss, val_epoch_accuracy, time.time() - start))
