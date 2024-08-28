import torch
from torch import nn, optim
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from model import regression
from readData import covid19Data
from torch.utils.data import DataLoader, random_split

batch_size = 32
lr = 1e-3
epochs = 300

df = pd.read_csv("covid_train.csv", header=0)
df = df.to_numpy(dtype = np.float32)
df = torch.from_numpy(df)
data = covid19Data(df)

train_len = int(len(data) // 3)
test_len = len(data) - train_len*2
train_data, val_data, test_data = random_split(data, [train_len, train_len, test_len])

train_loader = DataLoader(train_data, batch_size=batch_size)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

device = torch.device("cuda")
model = regression(87, 1).to(device)
cirterion = nn.MSELoss()
opt = optim.Adagrad(model.parameters(), lr=lr)


def train(epoch):
    model.train()
    losses = 0
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()

        pred = model(data)
        loss = cirterion(pred, target)
        losses += loss.item()

        loss.backward()
        opt.step()

        if (epoch+1) % 100 == 0: torch.save(model.state_dict(), 'pt/regression_{}.pt'.format(epoch))
    print("train epoch:{}, loss:{:.6f}".format(epoch+1, losses/len(train_loader)))
    return losses/len(train_loader)

def val(epoch):
    model.eval()
    losses = 0

    for idx, (data, target) in enumerate(val_loader):
        data, target = data.to(device), target.to(device)

        pred = model(data)
        loss = cirterion(pred, target)
        losses += loss.item()

    print("val epoch:{}, loss:{:.6f}".format(epoch+1, losses/len(val_loader)))
    return losses/len(val_loader)

def test():
    model.eval()
    acc = 0
    total = 0
    with torch.no_grad():
        for data, target in test_data:
            data, target = data.to(device), target.to(device)
            pred = model(data).item()
            error = pred - target.item()
            acc += math.pow(error, 2)
            total += target.size(0)
        mse = math.sqrt(acc / total)
    return mse

if __name__ == "__main__":
    train_loss = []
    test_loss = []

    for i in range(epochs):
        train_loss.append(train(i))
        test_loss.append(val(i))
    
    RMSE = test()
    print("test rmse:{:.6f}",format(RMSE))

    plt.figure(1)
    plt.plot(train_loss, "r", label="train_loss")
    plt.plot(test_loss, "b", label="test_loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.show()