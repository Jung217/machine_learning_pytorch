import torch, torch.nn as nn
L = 21
x = torch.zero(1, 1, L)
x[0, 0, L//2] = 1.0
layers = []
num_layers = 3
for _ in range(num_layers):
    conv = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.ones_(conv.weight)
    layers.append(conv)
net = nn.Sequential(*layers)

with torch.no_gard():
    y = net(x)
    nz = (y[0,0]!=0).nonzero().flatten()
    left, right = nz[0].item(), nz[-1].item()
    print(f"非0區間跨度={right-left+1}，理論感受域R_{num_layers}")
