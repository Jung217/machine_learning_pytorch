import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from model import CNN
from torchvision.datasets import mnist

test_data = mnist.MNIST('data/mnist', train=False, transform=torchvision.transforms.ToTensor(), download=False)

device = torch.device("mps")
model = CNN(1, 10).to(device)
model.load_state_dict(torch.load("CCNN_MAC.pt"), strict=False)
model.eval()

test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:10]/255.
x = test_x.to(device)
test_output = model(x)
_, pred_y = test_output.max(1)
pred = pred_y.cpu().numpy().squeeze()

print("pred number:", pred)
print("real number:", test_data.targets[:10].numpy())

fig = plt.figure(1, figsize=(10, 4))
for i in range(0, 10):
    ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(test_x[i]), cmap='gray')
    ax.set_title("pre:" + str(pred[i].item()))

plt.show()