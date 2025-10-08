import os
import argparse

from torchvision import datasets, transforms

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CNN

import time

from torch.utils.data import random_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data', help="資料目錄")
    parser.add_argument("--out_dir", type=str, default='result', help="訓練輸出目錄")
    parser.add_argument("--epochs", type=int, default=10, help="訓練回合數")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data_dir={args.data_dir}, out_dir={args.out_dir}")

    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
    # train_dataset = datasets.FashionMNIST( root=args.data_dir, train=True, download=False, transform=tfm) #True
    # print(f"訓練筆數={len(train_dataset)}(location:{args.data_dir})")

    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # print(f"DataLoader is Ready (batch_size={args.batch_size})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes=10).to(device)
    print(f"使用裝置: {device}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("[S5] Loss/Optimizer 準備完成")

    full_train = datasets.FashionMNIST(root=args.data_dir, train=True, download=False, transform=tfm)
    val_len = 10000
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    loss_csv = os.path.join(args.out_dir, 'loss.csv')
    with open(loss_csv, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')

    best_val_loss = float('inf')
    best_path = os.path.join(args.out_dir, 'best_cnn.pth')
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels) # CE needs long/int64

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_acc += (logits.argmax(1)==labels).float().sum().item()
            n += bs

        train_loss = running_loss / n
        train_acc = running_acc / n

        #---- validate ----
        model.eval()
        val_loss, val_acc, m = 0.0, 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                bs = labels.size(0)
                val_loss += loss.item() * bs
                val_acc += (logits.argmax(1)==labels).float().sum().item()
                m += bs
            val_loss /= m
            val_acc /= m
        with open(loss_csv, 'a', encoding='utf-8') as f:
            f.write(f'{epoch}, {train_loss:.6f}, {val_loss:.6f}, {train_acc:.4f}, {val_acc:.4f}\n')

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} val_loss={val_loss:.6f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # dt = time.time() -t0
        # print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} ({dt:.1f}s)")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoxh': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_path)

    print(f"最小訓練完成，loss已寫入:{loss_csv}")

if __name__ == '__main__':
    main()

