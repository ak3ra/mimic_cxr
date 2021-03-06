import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from data import PneumoniaDataset
from data import my_transforms,dataset,train_set,test_set, train_loader, test_loader
from torchvision import transforms, models
from model import Convnet, CustomResnet
import time
from torch.utils.tensorboard import SummaryWriter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CustomResnet().to(device) #To use resnet model and transfer learning
writer = SummaryWriter()

batch_size = 8
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
    writer.add_scalar('Loss/Train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/Train', epoch_accuracy, epoch)
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
        writer.add_scalar('Loss/Test', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/Test', val_epoch_accuracy, epoch)
        print("Epoch: {}, valid loss: {:.4f}, valid accracy: {:.4f}, time: {}\n".format(epoch, val_epoch_loss, val_epoch_accuracy, time.time() - start))

print("Finished Training")
print("Saving trained model")
torch.save(model.state_dict(), '../models/model.pt')
