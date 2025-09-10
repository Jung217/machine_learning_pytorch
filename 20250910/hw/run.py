import os
import torch
import pandas as pd
import numpy as np
from torch import nn, optim
from model import Model
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

data = np.genfromtxt('taxi_fare_training.csv', delimiter=',', skip_header=1)
x = torch.tensor(data[:, 0], dtype=torch.float32).view(-1, 1)
y = torch.tensor(data[:, 1], dtype=torch.float32).view(-1, 1)

net = Model()
opt = optim.SGD(net.parameters(), lr=0.01)
loss_f = nn.MSELoss()
loss_hist = []

for epoch in range(51):
    y_hat = net(x)
    loss = loss_f(y, y_hat)
    opt.zero_grad()
    loss.backward()
    opt.step()
    loss_hist.append(float(loss.item()))
    print(f'epoch:{epoch}, loss:{loss.item()}')
    print(loss)

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

plt.figure()
plt.plot(range(len(loss_hist)), loss_hist, marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
loss_path = os.path.join(OUT_DIR, 'loss.png')
plt.savefig(loss_path, dpi=150)
plt.close()

net.eval()
with torch.no_grad():
    y_hat = net(x)

arr = np.hstack([
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    y_hat.detach().cpu().numpy()
])
pred_csv_path = os.path.join(OUT_DIR, 'pred.csv')
np.savetxt(pred_csv_path, arr)

idx = x.squeeze(1).argsort()
x_sorted = x[idx]
yhat_sorted = y_hat[idx]

plt.figure()
plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), 'o', alpha=0.6, label='Data (x, y)')
plt.plot(x.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), 'x', alpha=0.8, label='Model output on training x')
plt.plot(x_sorted.detach().cpu().numpy(), yhat_sorted.detach().cpu().numpy(), '-', alpha=0.8, label='Connected model outputs')

plt.xlabel('x')
plt.ylabel('value')
plt.title('Model output on training points')
plt.legend()
plt.tight_layout()
scatter_outpus_path = os.path.join(OUT_DIR, 'data_with_true_output_mod.png')
plt.savefig(scatter_outpus_path, dpi=150)
plt.close()