import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from data import PneumoniaDataset
from torchvision import transforms

#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



## Custom Model
class CNN(nn.Module):
    def __init__(self, in_channel=1, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 8,
            kernel_size = (3,3),
            strid=(1,1),
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

        self.fc1 = nn.Linear(16*7*7, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = reshape(x.shape[0],-1)
        x = self.fc1(x)

        return x



my_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.RandomCrop((224,224)),
    transforms.ToTensor(),
    ])

## Hyper Parameters

batch_size = 16
num_classes = 2
learning_rate = 1e-3
num_epochs = 2
in_channel = 1

dataset = PneumoniaDataset(csv_file="output/pneumonia_images_and_labels.csv", 
                        root_dir = "/scratch/akera/mmic_data/physionet.org/files/mimic-cxr-jpg/2.0.0/files",
                        transform = my_transforms)


train_set, test_set = torch.utils.data.random_split(dataset, [80000,9280])

train_loader = DataLoader(dataset=train_set,batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set,batch_size=batch_size, shuffle=True)


model = CNN().to(device)

#loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)

        ## forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())

        ## Backward
        optimizer.zero_grad()
        loss.backward()

        ## Gradient descent

        optimizer.step()

    print(f'cost at epoch {epoch} is {sum(losses)/len(losses)}')



