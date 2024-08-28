# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 16, 24, 24]             416
#               ReLU-2           [-1, 16, 24, 24]               0
#          MaxPool2d-3           [-1, 16, 12, 12]               0
#             Conv2d-4             [-1, 32, 8, 8]          12,832
#               ReLU-5             [-1, 32, 8, 8]               0
#          MaxPool2d-6             [-1, 32, 4, 4]               0
#            Flatten-7                  [-1, 512]               0
#             Linear-8                  [-1, 100]          51,300
#             Linear-9                   [-1, 10]           1,010
# ================================================================
# Total params: 65,558
# Trainable params: 65,558
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.20
# Params size (MB): 0.25
# Estimated Total Size (MB): 0.45
# ----------------------------------------------------------------
import torch
from torch import nn
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 16, 5, 1) #16 ； kernel 5x5 ； 1 strait?
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*4*4, 100)
        self.fc2 = nn.Linear(100, out_dim)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':
  device = torch.device("cuda")
  model = CNN(1, 10).to(device)

  summary(model, (1, 28, 28))
    
