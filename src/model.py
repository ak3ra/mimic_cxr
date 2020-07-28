import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
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


# Model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

#loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_set):
        data = data.to(device)
        targets = targets.to(device)

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



