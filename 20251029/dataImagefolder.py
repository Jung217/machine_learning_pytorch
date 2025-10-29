import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from torchvision.datasets import ImageFolder

data_root = "./datasets2/cifar10_folders"
train_dir = os.path.join(data_root, 'train')
test_dir = os.path.join(data_root, 'test')

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if not os.path.exists(data_root):
    os.makedirs(data_root, exist_ok=True)
    transform_to_pil = transforms.ToPILImage()

    cifar_train = torchvision.datasets.CIFAR10(
        root='./dataset2/cifar10',
        train=True,
        download=True
    )
    cifar_test = torchvision.datasets.CIFAR10(
        root='./dataset2/cifar10',
        train=False,
        download=True
    )

    for class_name in classes: os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    for class_name in classes: os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    print("saving...")
    class_counts = {i: 0 for i in range(10)}
    for idx, (img, label) in enumerate(cifar_train):
        if class_counts[label] < 100:
            img.save(os.path.join(train_dir, classes[label], f'{class_counts[label]}.png'))
            class_counts[label] += 1
        if all(count >= 100 for count in class_counts.values()): break

    class_counts = {i: 0 for i in range(10)}
    for idx, (img, label) in enumerate(cifar_test):
        if class_counts[label] < 100:
            img.save(os.path.join(test_dir, classes[label], f'{class_counts[label]}.png'))
            class_counts[label] += 1
        if all(count >= 100 for count in class_counts.values()): break
    print("done")
else:
    print("Dataset already exists")

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

full_train_dataset = ImageFolder(
    root=train_dir,
    transform=train_transform
)

test_dataset = ImageFolder(
    root=test_dir,
    transform=test_transform
)

train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(9527)
)

batch_size = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

dataiter = iter(train_loader)
images, labels = next(dataiter)

fig = plt.figure(figsize=(12, 6))
fig.suptitle('ImageFolder sapmle', fontsize=16, fontweight='bold')

classes = full_train_dataset.classes

for idx in range(8):
    ax = plt.subplot(2, 4, idx+1)
    img = images[idx]
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(f'class:{classes[labels[idx]]}')
    plt.axis('off')

plt.tight_layout()
os.makedirs('./outputs2', exist_ok=True)
plt.savefig('./outputs2/imagefolder_sample.png', dpi=150, bbox_inches='tight')
print("img saved")