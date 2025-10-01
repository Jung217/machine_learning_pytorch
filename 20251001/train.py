import os
import argparse

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CNN

import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data', help="資料目錄")
    parser.add_argument("--out_dir", type=str, default='result', help="訓練輸出目錄")
    parser.add_argument("--epochs", type=int, default=5, help="訓練回合數")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data_dir={args.data_dir}, out_dir={args.out_dir}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    train_dataset = datasets.FashionMNIST( root=args.data_dir, train=True, download=False, transform=tfm) #True
    print(f"訓練筆數={len(train_dataset)}(location:{args.data_dir})")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print(f"DataLoader is Ready (batch_size={args.batch_size})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes=10).to(device)
    print(f"使用裝置: {device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("[S5] Loss/Optimizer 準備完成")

    loss_csv = os.path.join(args.out_dir, 'loss.csv')
    with open(loss_csv, 'w', encoding='utf-8') as f:
        f.write('epoch, train_loss\n')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, n = 0.0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            n += bs

        train_loss = running_loss / n

        with open(loss_csv, 'a', encoding='utf-8') as f:
            f.write(f'{epoch}, {train_loss:.6f}\n')

        dt = time.time() -t0
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} ({dt:.1f}s)")

    print(f"最小訓練完成，loss已寫入:{loss_csv}")

if __name__ == '__main__':
    main()