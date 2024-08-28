import torch
from torch import nn
from torchsummary import summary

class regression(nn.Module):
  def __init__(self, input_dim, out_dim):
    super(regression, self).__init__()
    self.linear = nn.Linear(input_dim, out_dim)

  def forward(self, x):
    x = self.linear(x)
    return x
  
if __name__ == '__main__':
  device = torch.device("cuda")
  model = regression(1, 1).to(device)

  summary(model, (1, 1))