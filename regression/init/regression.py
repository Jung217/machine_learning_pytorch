import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from model import regression

x = torch.randn(100, 1)
w = torch.tensor([10.])
b = torch.tensor([3.])
y = w * x + b + torch.randn(x.shape) * 2.5

print(y)

device = torch.device("cuda")
model = regression(1, 1).to(device)

loss_f = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=0.1)

epoch = 51
losses = []

for i in range(epoch):
    input, target = x.to(device), y.to(device)
    model.train()

    opt.zero_grad()

    pred = model(input)
    loss = loss_f(target, pred)

    loss.backward()
    opt.step()

    losses.append(loss.item())

    print('epoch:{}, loss:{}'.format(i, loss.item()))  

plt.figure(1)
plt.plot(x, y, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('data')
plt.show()

plt.figure(2)
plt.plot(losses, label="loss")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss')
plt.show()