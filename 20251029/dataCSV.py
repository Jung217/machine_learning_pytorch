import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

data_root = "./datasets3/cifar10_folders"
imges_dir = os.path.join(data_root, 'images')
os.makedirs(imges_dir, exist_ok=True)

CSV_path = os.path.join(data_root, 'label.csv')

if not os.path.exists(CSV_path):
    cifar_dataset = torchvision.datasets.CIFAR10(
        root='./dataset3/cifar10',
        train=True,
        download=True
    )
    
    data_list = []
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    class_counts = {i: 0 for i in range(10)}
    for idx, (img, label) in enumerate(cifar_dataset):
        if class_counts[label] < 10:
            img_name = f'{classes[label]}_{class_counts[label]}.png'
            img_path = os.path.join(imges_dir, img_name)
            img.save(img_path)

            data_list.append({
                'image_name': img_name,
                'label': label,
                'class_name':classes[label]
            })

            class_counts[label] += 1
        if all(count >= 10 for count in class_counts.values()): break

        df = pd.DataFrame(data_list)
        df.to_csv(CSV_path, index=False)
        print(f"created {len(df)} data entries")
        print(f"file at {CSV_path}")
        print(f"Img directory {imges_dir}")
else:
    print("Dataset already exists")
    df = pd.read_csv(CSV_path)
    print(f"Load {len(df)} entries")
