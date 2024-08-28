import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from model import CNN

train_data = mnist.MNIST('data/mnist', train=True, transform=torchvision.transforms.ToTensor(), download=False)
test_data = mnist.MNIST('data/mnist', train=False, transform=torchvision.transforms.ToTensor(), download=False)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

plt.title('HI')
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.show()

device = torch.device('cuda')
model = CNN(1, 10).to(device)
loss_f = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

def train():
    model.train()
    train_loss = 0
    train_acc = 0

    for batch ,(data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()

        pred = model(data)
        loss = loss_f(pred, target)
        
        loss.backward()
        opt.step()

        train_loss += loss.item()
        _, y = pred.max(1)
        correct = (y == target).sum().item()
        acc = correct / data.shape[0]
        train_acc += acc
    
    return train_loss/len(train_loader), train_acc/len(test_loader)

def test():
    test_loss = 0
    test_acc = 0
    preds = []
    model.eval()
    for batch ,(data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        y = model(data)
        loss = loss_f(y, target)
        test_loss += loss.item()
        _, pred = y.max(1)
        num_correct = (pred == target).sum().item()  
        acc = num_correct / data.shape[0]
        test_acc += acc
        preds.append(pred)
    
    return test_loss/len(test_loader), test_acc/len(test_loader), preds

train_losses = []
train_acces = []
test_losses = []
test_acces = []

for i in range(0, 10):
    loss, acc = train()
    test_loss, test_acc, y = test()

    train_losses.append(loss)
    train_acces.append(acc)
    test_losses.append(test_loss)
    test_acces.append(test_acc)
    


