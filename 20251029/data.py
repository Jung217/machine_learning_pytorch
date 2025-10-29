import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./datasets/cifar10',
    train=True,
    download=False,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./datasets/cifar10',
    train=False,
    download=False,
    transform=transform
)

print(f"train data size:{len(trainset)}")
print(f"test data size:{len(testset)}")
print("class =", len(trainset.classes))
print("class names =", trainset.classes)
print("class to idx =", trainset.class_to_idx)

img, lbl = trainset[0]
print("single img shape =", img.shape, "label =", lbl, trainset.classes[lbl])

train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size

train_dataset, val_dataset = random_split(
    trainset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(9527)
)

print(f"train dataset size:{len(train_dataset)}")
print(f"validation dataset size:{len(val_dataset)}")
print(f"test dataset size:{len(testset)}")

batch_size = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"train batches:{len(train_loader)}")
print(f"validation batches:{len(val_loader)}")
print(f"test batches:{len(test_loader)}")

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow()
    plt.axis('off')

dataiter = iter(train_loader)
images, labels = next(dataiter)

fig = plt.figure(figsize=(12, 6))
fig.suptitle('CIFAR-10', fontsize=16, fontweight='bold')

classes = trainset.classes

for idx in range(8):
    ax = plt.subplot(2, 4, idx+1)
    img = images[idx]
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'class:{classes[labels[idx]]}')
    plt.axis('off')

plt.tight_layout()
os.makedirs('./outputs', exist_ok=True)
plt.savefig('./outputs/cifar10_sample.png', dpi=150, bbox_inches='tight')
print("img saved")