# Pytorch
### Regression
* model.py
```py
import torch
from torch import nn, optim

# 定義一個繼承自 nn.Module 的自訂模型類別
class Model(nn.Module):
    def __init__(self, in_dim=1, out_dim=1):
        """
        初始化模型結構
        參數：
            in_dim: 輸入維度（預設 1）
            out_dim: 輸出維度（預設 1）
        """
        super().__init__()                   # 初始化父類別 (nn.Module)
        
        # 建立一層線性層（Linear Layer）
        # y = W * x + b
        # W 的形狀為 (out_dim, in_dim)
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        """
        定義前向傳播 (Forward Pass)
        輸入：
            x: 張量輸入 (tensor)，形狀為 [batch_size, in_dim]
        輸出：
            模型預測值 (tensor)，形狀為 [batch_size, out_dim]
        """
        return self.lin(x)                   # 呼叫線性層完成 y = Wx + b 的運算


# 建立模型實例，輸入 1 維 → 輸出 1 維
model = Model(1, 1)

# 可印出模型結構確認
# print(model)
```
* train.py
```py
# 匯入所需模組
import os
import torch
import numpy as np
from torch import nn, optim           # nn 提供神經網路層、loss function 等，optim 提供優化器
from model import Model              # 從自定義的 model.py 載入模型類別
import matplotlib.pyplot as plt      # 用於畫出訓練過程與結果

# 讀取資料集 (假設為兩欄：x, y)
data = np.genfromtxt('taxi_fare_training.csv', delimiter=',', skip_header=1)

# 將 numpy array 轉成 PyTorch tensor，並調整成 (n, 1) 的形狀
x = torch.tensor(data[:, 0], dtype=torch.float32).view(-1, 1)
y = torch.tensor(data[:, 1], dtype=torch.float32).view(-1, 1)

# ---- 以下為可選：產生假資料測試模型 ----
# torch.manual_seed(0)
# n = 100
# x = torch.randn(n, 1)                         # 隨機產生 n 筆輸入 x
# w_true = torch.tensor([10.0])                 # 真實權重
# b_true = torch.tensor([3.0])                  # 真實偏差
# y = w_true * x + b_true + torch.randn(n, 1) * 2.5  # 加入雜訊模擬真實資料

# 建立模型實例（Model 類別定義在 model.py 中）
net = Model()

# 建立 SGD (隨機梯度下降) 優化器，學習率 lr = 0.01
opt = optim.SGD(net.parameters(), lr=0.01)

# 均方誤差 (Mean Squared Error) 作為損失函數
loss_f = nn.MSELoss()

# 用來紀錄每個 epoch 的 loss
loss_hist = []

# ------------------------------
#         模型訓練迴圈
# ------------------------------
for epoch in range(31):  # 共訓練 31 個 epoch
    y_hat = net(x)                   # 前向傳播：模型輸出預測值
    loss = loss_f(y, y_hat)          # 計算 MSE 損失

    opt.zero_grad()                  # 清空上一次的梯度
    loss.backward()                  # 反向傳播：計算每個參數的梯度
    opt.step()                       # 根據梯度更新模型參數

    loss_hist.append(float(loss.item()))  # 將損失記錄下來
    print(f"epoch:{epoch}, loss:{loss.item()}")  # 顯示每輪的 loss
    print(loss)                                # 顯示張量型態的 loss（for debug）

# ------------------------------
#       繪製訓練損失圖表
# ------------------------------
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)   # 若資料夾不存在則建立

plt.figure()
plt.plot(range(len(loss_hist)), loss_hist, marker='o')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# 將 loss 曲線圖儲存下來
loss_path = os.path.join(OUT_DIR, 'loss.png')
plt.savefig(loss_path, dpi=150)
plt.close()

# ------------------------------
#      評估模型預測結果
# ------------------------------
net.eval()                           # 切換到 evaluation 模式 (關閉 dropout/batchnorm)
with torch.no_grad():                # 關閉梯度計算（節省記憶體與時間）
    y_hat = net(x)                   # 模型預測結果

# 將 x、真實 y、預測 y_hat 拼接成一個 numpy 陣列
arr = np.hstack([
    x.detach().cpu().numpy(),
    y.detach().cpu().numpy(),
    y_hat.detach().cpu().numpy()
])

# 儲存預測結果到 CSV
pred_csv_path = os.path.join(OUT_DIR, 'pred.csv')
np.savetxt(pred_csv_path, arr)

# ------------------------------
#       視覺化模型輸出
# ------------------------------

# 根據 x 值排序，方便畫連續線
idx = x.squeeze(1).argsort()
x_sorted = x[idx]
yhat_sorted = y_hat[idx]

plt.figure()
# 畫出原始資料點 (x, y)
plt.plot(x.detach().cpu().numpy(), y.detach().cpu().numpy(), 'o', alpha=0.6, label='Data (x, y)')

# 畫出模型在訓練點的預測結果 (散點)
plt.plot(x.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), 'x', alpha=0.8, label='Model output on training x')

# 畫出模型輸出曲線 (已排序)
plt.plot(x_sorted.detach().cpu().numpy(), yhat_sorted.detach().cpu().numpy(), '-', alpha=0.8, label='Connected model outputs')

plt.xlabel('x')
plt.ylabel('value')
plt.title('Model output on training points')
plt.legend()
plt.tight_layout()

# 儲存結果圖表
scatter_outpus_path = os.path.join(OUT_DIR, 'data_with_true_output_mod.png')
plt.savefig(scatter_outpus_path, dpi=150)
plt.close()
```

### DNN
* iris.py
```py
# 匯入必要套件
import os, numpy as np, pandas as pd
from typing import List
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn

import matplotlib.pyplot as plt

# ===============================
# 基本設定與資料夾建立
# ===============================
ROOT = "iris_course"
ARTIFACTS = os.path.join(ROOT, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)  # 儲存中間產物資料夾

# 顯示統計摘要的輔助函式
def describe_stats(X: np.ndarray, names: List[str], title: str):
    m, s = X.mean(axis=0), X.std(axis=0)
    print(f"\n[{title}]")
    for n, mi, sd in zip(names, m, s):
        print(f"{n:14s} mean={mi:8.4f} std={sd:8.4f}")

# ===============================
# STEP 1. 載入資料與初步探索
# ===============================
print("=== STEP 1 | load data and explore")
iris = load_iris()  # 內建 iris 資料集
x, y = iris.data, iris.target
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target_names = iris.target_names.tolist()

# 建立 DataFrame 方便檢視
df = pd.DataFrame(x, columns=feature_names)
df["targets"] = y

# 顯示前 5 筆資料與類別分布
print("\n前 5 筆資料:")
print(df.head())
print("\n類別分布:")
for i, name in enumerate(target_names):
    print(f" {i}={name:<10s} : {(y==i).sum()} 筆")

describe_stats(x, feature_names, "origin data (not normalized)")

# 儲存前 20 筆資料做預覽
out_csv = os.path.join(ARTIFACTS, "iris_preview.csv")
df.head(20).to_csv(out_csv, index=False)
print(f"\n save 20 data:{out_csv}")
print("STEP 1 finished\n")

# ===============================
# STEP 2. 資料切分 (train/val/test)
# ===============================
print("\n=== STEP 2 | split train/val/test ===")
X_trainval, X_test, y_trainval, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)
print(f"切分形狀: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
print("STEP 2 finished\n")

# ===============================
# STEP 3. 標準化 (StandardScaler)
# ===============================
print("=== STEP 3 | normalize (only train set fit) and save")

# 只用訓練資料 fit，再對 val/test transform
scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

# 顯示標準化前後統計
describe_stats(X_train, feature_names, "train set before normalize")
describe_stats(X_train_sc, feature_names, "train set after normalize")

# 儲存標準化後的資料
npz_path = os.path.join(ARTIFACTS, "train_val_test_scaled.npz")
np.savez(npz_path, X_train_sc=X_train_sc, y_train=y_train,
         X_val_sc=X_val_sc, y_val=y_val,
         X_test_sc=X_test_sc, y_test=y_test,
         feature_names=np.array(feature_names, dtype=object),
         target_names=np.array(target_names, dtype=object))
# 儲存 scaler 供未來反轉或推論使用
scaler_path = os.path.join(ARTIFACTS, "scalar.pkl")
joblib.dump(scaler, scaler_path)
print(f"save normalize data to {npz_path}")
print(f"save 標準化器 to {scaler_path}")
print("STEP 3 finished\n")

# ===============================
# STEP 4. 轉成 Tensor 與 DataLoader
# ===============================
print("=== STEP 4 | tensor and dataloader")
X_train_t = torch.tensor(X_train_sc, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val_sc, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=16, shuffle=False)

# 檢查第一批資料
xb, yb = next(iter(train_loader))
print(f"NO.1 batch:xb.shape={xb.shape}, yb.shape={yb.shape}")
print(f" xb[0](normalized) = {xb[0].tolist()}")
print(f" yb[0](type) = {yb[0].item()}")

# 將 batch 輸出儲存預覽
batch_preview = os.path.join(ARTIFACTS, "batch_preview.csv")
pd.DataFrame(xb.numpy(), columns=feature_names).assign(label=yb.numpy()).to_csv(batch_preview, index=False)
print(f"save batch preview to {batch_preview}")
print("STEP 4 finished\n")

# ===============================
# STEP 5. 建立模型與超參數設定
# ===============================
print("=== STEP 5 | define model and parameters")
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

# 定義一個兩層隱藏層的 MLP 模型
class IrisMLP(nn.Module):
    def __init__(self, in_dim=4, hidden1=64, hidden2=32, out_dim=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, out_dim)  # 最後輸出 3 類別
        )
    def forward(self, x):
        return self.net(x)

# 計算模型中可訓練參數數量
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IrisMLP().to(device)
print(model)
print(f"trainable parameters : {count_trainable_params(model):,}")

# 儲存模型架構描述
arch_txt = os.path.join(MODELS, "model_arch.txt")
with open(arch_txt, "w", encoding="utf-8") as f:
    f.write(str(model) + "\n")
    f.write(f"trainable parameters : {count_trainable_params(model)}\n")
print(f"->saved structure describe:{arch_txt}")
print("STEP 5 finished\n")

# ===============================
# STEP 6. 模型訓練 + 早停
# ===============================
print("\n== STEP 6 [訓練][含早停]==")
criterion = nn.CrossEntropyLoss()                  # 適用多類別分類
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 驗證函式：回傳 loss 與 accuracy
def evaluate(m, loader):
    m.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = m(xb)                         # 前向傳播
            loss_sum += criterion(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return loss_sum / total, correct / total

best_state, best_val, patience, bad = None, 0.0, 15, 0
hist = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}

# 訓練迴圈
for ep in range(1, 201):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)

    tr_loss, tr_acc = loss_sum/total, correct/total
    va_loss, va_acc = evaluate(model, val_loader)
    hist["tr_loss"].append(tr_loss); hist["tr_acc"].append(tr_acc)
    hist["va_loss"].append(va_loss); hist["va_acc"].append(va_acc)
    print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

    # Early stopping：若 val_acc 提升則更新最佳狀態
    if va_acc > best_val:
        best_val = va_acc
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
        if bad >= patience:
            print(f"早停：{patience} epochs 未提升。")
            break

# 載回最佳模型
if best_state:
    model.load_state_dict(best_state)

# ===============================
# STEP 6.2 繪製訓練曲線
# ===============================
xs = np.arange(1, len(hist["tr_loss"]) + 1)
plt.figure(figsize=(8,4))
plt.plot(xs, hist["tr_loss"], label="train_loss")
plt.plot(xs, hist["va_loss"], label="val_loss")
plt.plot(xs, hist["tr_acc"], label="train_acc")
plt.plot(xs, hist["va_acc"], label="val_acc")
plt.xlabel("epoch"); plt.ylabel("value"); plt.title("Training Curves")
plt.legend(); plt.tight_layout()

PLOTS = "plots"
os.makedirs(PLOTS, exist_ok=True)
curve_path = os.path.join(PLOTS, "curves.png")
plt.savefig(curve_path, dpi=150); plt.close()
print(f"=> training curve saved：{curve_path}")

# 儲存最佳模型權重
best_path = os.path.join(MODELS, "best.pt")
torch.save(model.state_dict(), best_path)
print(f"=> best weight saved：{best_path}")
print("STEP 6 finished")

# ===============================
# STEP 7. 測試與模型評估
# ===============================
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("\n== STEP 7 test & evaluate==")

model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_test_sc, dtype=torch.float32).to(device))
    y_pred = logits.argmax(1).cpu().numpy()

# 輸出測試集準確率與詳細報告
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}")
print("分類報告：")
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

# 混淆矩陣可視化
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4.5,4.5))
plt.imshow(cm, interpolation="nearest"); plt.title("Confusion Matrix"); plt.colorbar()
ticks = np.arange(len(target_names))
plt.xticks(ticks, target_names, rotation=30); plt.yticks(ticks, target_names)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("predicted"); plt.ylabel("true"); plt.tight_layout()
cm_path = os.path.join(PLOTS, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150); plt.close()
print(f"=> confusion_matrix saved {cm_path}")

# ===============================
# STEP 8. 範例輸出：顯示部分測試樣本
# ===============================
X_test_orig = scaler.inverse_transform(X_test_sc)
show_n = min(10, len(X_test_orig))
tbl = pd.DataFrame(X_test_orig[:show_n], columns=feature_names)
tbl["true"] = [target_names[i] for i in y_test[:show_n]]
tbl["pred"] = [target_names[i] for i in y_pred[:show_n]]

samples_csv = os.path.join(ARTIFACTS, "test_samples.csv")
tbl.to_csv(samples_csv, index=False)
print(f"=> preview test samples saved to {samples_csv}")
```

### CNN
* cnn.py
```py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        """
        簡單的卷積神經網路 (適合 MNIST / FashionMNIST 等 28x28 單通道影像)
        輸入 shape: (N, 1, 28, 28)
        輸出 shape: (N, num_classes)
        """
        super().__init__()  # 初始化父類別 (nn.Module)，用於參數註冊等

        # features: 提取影像特徵的卷積層堆疊
        # 注意每個 Conv2d 的 padding=1 與 kernel_size=3，能保持 spatial size（在 pooling 前）
        self.features = nn.Sequential(
            # 第一段 conv block: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # -> (N,32,28,28)
            nn.BatchNorm2d(32),                           # 批次正規化，加速收斂並穩定訓練
            nn.ReLU(inplace=True),                        # 非線性激活（inplace 可省記憶體）

            nn.Conv2d(32, 32, kernel_size=3, padding=1), # -> (N,32,28,28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2), # 空間降采樣: 28 -> 14   -> (N,32,14,14)

            # 第二段 conv block: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # -> (N,64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1), # -> (N,64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2), # 空間降采樣: 14 -> 7    -> (N,64,7,7)

            nn.Dropout(0.25), # 輕量 dropout，減少過擬合
        )

        # classifier: 將卷積特徵映射到類別分數 (logits)
        # Flatten 後的維度為 64 * 7 * 7 = 3136
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (N,64,7,7) -> (N, 64*7*7)
            nn.Linear(64 * 7 * 7, 256),   # 全連接層：3136 -> 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),              # 較高比例 dropout，進一步 regularize
            nn.Linear(256, num_classes),  # 輸出層：256 -> num_classes (logits)
        )

        # 可選：初始化權重（建議但非必須）
        self._init_weights()

    def forward(self, x):
        """
        前向傳播
        x: Tensor, shape (N, 1, 28, 28)
        回傳 logits: shape (N, num_classes)
        """
        # 經過 feature extractor
        x = self.features(x)
        # 經過 classifier（包含 flatten）
        x = self.classifier(x)
        return x

    def _init_weights(self):
        """
        建議的權重初始化：
         - Conv2d 使用 Kaiming for ReLU
         - Linear 使用 Xavier（Glorot）
         - bias 初始化為 0
        好處：通常能讓訓練更穩定、收斂更好
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# 當作腳本執行時的快速自我檢查
if __name__ == '__main__':
    m = CNN(num_classes=10)                    # 建立模型（預設 10 類）
    x = torch.randn(8, 1, 28, 28)              # 模擬一個 batch: N=8, C=1, H=28, W=28
    y = m(x)
    print(y.shape)  # 預期 output: torch.Size([8, 10])
```
* train.py
```py
import os
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CNN   # 匯入你自己定義的 CNN 類別（在 cnn.py 中）
import time

# ============================================================
# 主程式入口
# ============================================================
def main():
    # ------------------------------------------------------------
    # 參數設定：可由命令列指定，也可用預設值
    # 例如：python train.py --epochs 20 --lr 0.0005
    # ------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data', help="資料目錄")
    parser.add_argument("--out_dir", type=str, default='result', help="訓練輸出目錄")
    parser.add_argument("--epochs", type=int, default=10, help="訓練回合數")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="學習率")
    args = parser.parse_args()

    # 確保資料與輸出目錄存在
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"data_dir={args.data_dir}, out_dir={args.out_dir}")

    # ============================================================
    # [S1] 資料前處理：ToTensor + Normalize
    # FashionMNIST 圖像為灰階(1通道)，像素範圍[0,1]，Normalize後到[-1,1]
    # ============================================================
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # ============================================================
    # [S2] 準備訓練 / 驗證資料集
    # 先載入整個 training set，再切成 train/val 兩部分
    # ============================================================
    full_train = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=False,     # 若第一次執行可改 True，自動下載
        transform=tfm
    )

    val_len = 10000                    # 保留 10,000 筆作驗證
    train_len = len(full_train) - val_len
    train_set, val_set = random_split(full_train, [train_len, val_len])

    # ============================================================
    # [S3] DataLoader 建立
    # shuffle=True：訓練集要隨機打亂
    # num_workers：用多執行緒加速載入
    # pin_memory=True：加速 GPU 傳輸
    # ============================================================
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # ============================================================
    # [S4] 模型、Loss、Optimizer 準備
    # ============================================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(num_classes=10).to(device)  # 10 類別 (T-shirt, Trouser, Coat ...)
    print(f"使用裝置: {device}")

    criterion = nn.CrossEntropyLoss()        # 適用於分類問題的損失函數
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    print("[S5] Loss/Optimizer 準備完成")

    # ============================================================
    # [S6] 訓練過程：建立紀錄檔
    # ============================================================
    loss_csv = os.path.join(args.out_dir, 'loss.csv')
    with open(loss_csv, 'w', encoding='utf-8') as f:
        f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')

    # 儲存最佳模型
    best_val_loss = float('inf')
    best_path = os.path.join(args.out_dir, 'best_cnn.pth')

    # ============================================================
    # [S7] 主訓練迴圈
    # ============================================================
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Training Phase ---
        model.train()  # 啟用 dropout, batchnorm 的訓練模式
        running_loss, running_acc, n = 0.0, 0.0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).long()

            # Forward + Backward + Update
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # 累積統計資訊
            bs = labels.size(0)
            running_loss += loss.item() * bs
            running_acc += (logits.argmax(1) == labels).float().sum().item()
            n += bs

        # 計算平均訓練 loss 與 acc
        train_loss = running_loss / n
        train_acc = running_acc / n

        # --- Validation Phase ---
        model.eval()
        val_loss, val_acc, m = 0.0, 0.0, 0
        with torch.no_grad():  # 關閉梯度以節省記憶體
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                bs = labels.size(0)
                val_loss += loss.item() * bs
                val_acc += (logits.argmax(1) == labels).float().sum().item()
                m += bs

        val_loss /= m
        val_acc /= m

        # --- 儲存紀錄到 CSV ---
        with open(loss_csv, 'a', encoding='utf-8') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.4f},{val_acc:.4f}\n')

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.6f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        # --- 檢查是否為最佳模型 (以驗證 loss 為準) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
            }, best_path)

    # ============================================================
    # [S8] 訓練完成後的結果總結
    # ============================================================
    print(f"訓練完成，loss/acc 紀錄已寫入: {loss_csv}")
    print(f"最佳模型權重已儲存至: {best_path}")

# ============================================================
# 程式進入點
# ============================================================
if __name__ == '__main__':
    main()
```
* test.py
```py
import os
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from cnn import CNN  # 匯入自己定義的 CNN 模型
from matplotlib import font_manager

# ================== 可修改參數 ==================
DATA_DIR = "data"                 # 資料存放目錄
OUT_DIR = "result"                # 推論輸出資料夾
CKPT_PATH = Path(OUT_DIR) / "best_cnn.pth"  # 已訓練模型權重位置
ROWS, COLS = 2, 5                 # 顯示圖片的行列數
SEED = 20                          # 隨機種子
MEAN, STD = 0.2861, 0.3530        # Normalize 均值與標準差 (需與訓練一致)
FIGSIZE = (16, 8)                  # 圖片大小
TITLE_FZ = 11                       # 子圖標題字體大小
ZH_FONT_CANDIDATES = [             # 中文字型清單
    r"C:\Windows\Fonts\msjh.ttc",
    r"C:\Windows\Fonts\msjhl.ttc",
    r"C:\Windows\Fonts\mingliu.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
]

CLASS_NAMES = [                    # FashionMNIST 類別名稱
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ================== 工具函式 ==================
def set_seed(seed=42):
    """設定隨機種子，確保可重現"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_zh_font():
    """全域設定中文字型，避免中文顯示方塊"""
    for p in ZH_FONT_CANDIDATES:
        if Path(p).exists():
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False
            print(f"[INFO] 使用中文字型: {prop.get_name()} ({p})")
            return
    print("[WARN] 找不到中文字型,請安裝「思源正黑體」或更新 ZH_FONT_CANDIDATES")

def denorm_img(x):
    """將正規化圖片還原到 0~1，方便顯示"""
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]  # 去掉 channel 維度
    x = x * STD + MEAN  # 還原 Normalize
    return x.clamp(0, 1)  # 限制在 [0,1]

# ================== 主程式 ==================
def main():
    # 設定隨機種子 & 中文字型
    set_seed(SEED)
    set_zh_font()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用裝置: {device}")

    # 與訓練相同的 Transform
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])

    # 讀取測試集 (官方 10,000 張)
    test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=tfm)

    # 建立 CNN 模型並載入最佳權重
    model = CNN(num_classes=10).to(device)
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        state = ckpt.get("model_state", ckpt)  # 兼容只存 state_dict 或完整 ckpt
        model.load_state_dict(state)
        print(f"[INFO] 已載入 checkpoint: {CKPT_PATH}")
    else:
        print(f"[WARN] checkpoint 不存在: {CKPT_PATH}，使用隨機初始化權重 (僅示意)")

    model.eval()  # 評估模式 (停用 dropout/batchnorm 更新)

    # 隨機抽樣 N 張圖片進行推論
    N = ROWS * COLS
    indices = list(range(len(test_set)))
    random.shuffle(indices)
    indices = indices[:N]

    images, gts, preds, confs = [], [], [], []

    # ================== 推論 ==================
    with torch.no_grad():  # 不計算梯度，加快速度
        for idx in indices:
            img_t, label = test_set[idx]                # 取得圖片與真實標籤
            logits = model(img_t.unsqueeze(0).to(device))  # [1,10] 模型輸出
            prob = F.softmax(logits, dim=1)[0]         # 轉成機率分佈
            pred_id = int(torch.argmax(prob).item())    # 取得預測類別 ID
            conf = float(prob[pred_id].item())         # 取得該類別信心度

            images.append(img_t.cpu())                  # 保存原始張量
            gts.append(label)                           # 真實標籤
            preds.append(pred_id)                        # 預測標籤
            confs.append(conf)                          # 信心度

    # ================== 繪圖 ==================
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)

    for i in range(N):
        ax = fig.add_subplot(ROWS, COLS, i + 1)
        img = denorm_img(images[i])
        ax.imshow(img, cmap="gray", interpolation="nearest")  # 保留像素風格
        ax.axis("off")

        # 標題顯示：真實 / 預測 / 信心度
        gt_name = CLASS_NAMES[gts[i]]
        pred_name = CLASS_NAMES[preds[i]]
        conf_txt = f"{confs[i]:.3f}"
        title = f"真實:{gt_name}\n預測:{pred_name}\n信心度:{conf_txt}"
        ax.set_title(title, color="green", fontsize=TITLE_FZ, pad=6)

    # 大標題
    fig.suptitle("模型預測結果", fontsize=16, y=0.98)

    # 儲存圖片
    out_path = Path(OUT_DIR) / "test_grid.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已輸出: {out_path.resolve()}")

# ================== 程式入口 ==================
if __name__ == "__main__":
    main()
```
* plot_metrics.py
```py
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import math

# CSV 來源檔案 & 圖片輸出路徑
CSV_PATH = Path("result/loss.csv")
OUT_PATH = Path("result/metrics.png")

# ================== 讀取訓練/驗證指標 ==================
def read_metrics(csv_path: Path):
    """
    讀取 CSV 檔案，抓取 epoch、train_loss、val_loss、train_acc、val_acc 欄位
    回傳 dict 形式
    """
    # 初始化欄位字典
    cols = {k: [] for k in ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]}

    # 嘗試用不同編碼讀取
    try:
        f = open(csv_path, "r", encoding="utf-8-sig")
        first_try = f
    except Exception:
        f = open(csv_path, "r", encoding="utf-8")
        first_try = f
    
    # CSV DictReader
    with first_try as ff:
        reader = csv.DictReader(ff)
        if not reader.fieldnames: 
            raise RuntimeError("CSV without headers!")  # 沒有欄位標頭

        # 欄位大小寫映射
        fmap = {k.strip().lower(): k for k in reader.fieldnames}
        need = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
        miss = [k for k in need if k not in fmap]

        if miss: 
            raise RuntimeError(f"必要欄位缺失:{miss}; CSV 欄位:{reader.fieldnames}")

        # 逐行解析
        for row in reader:
            try:
                e = int(float(row[fmap["epoch"]]))
                tl = float(row[fmap["train_loss"]])
                vl = float(row[fmap["val_loss"]])
                ta = float(row[fmap["train_acc"]])
                va = float(row[fmap["val_acc"]])
            except Exception:
                continue  # 若該行有錯就跳過

            cols["epoch"].append(e)
            cols["train_loss"].append(tl)
            cols["val_loss"].append(vl)
            cols["train_acc"].append(ta)
            cols["val_acc"].append(va)

    if not cols["epoch"]: 
        raise RuntimeError("沒有有效資料")
    return cols

# ================== 工具函式 ==================
def argmin(lst):
    """取得列表中最小值的索引"""
    bi, bv = None, math.inf
    for i, v in enumerate(lst): 
        if v < bv: bi, bv = i, v
    return bi

def argmax(lst):
    """取得列表中最大值的索引"""
    bi, bv = None, -math.inf
    for i, v in enumerate(lst): 
        if v > bv: bi, bv = i, v
    return bi

# ================== 主程式 ==================
def main():
    # 確保輸出資料夾存在
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 讀取指標
    data = read_metrics(CSV_PATH)
    epochs = data["epoch"]

    # 找出最小驗證損失與最大驗證精度的 epoch
    i_min_vl = argmin(data["val_loss"])
    i_max_va = argmax(data["val_acc"])
    ep_min_vl = epochs[i_min_vl]
    ep_max_va = epochs[i_max_va]

    # 建立圖表
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)

    # ------------------ Loss ------------------
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, data["train_loss"], label="train_loss", linewidth=2)
    ax1.plot(epochs, data["val_loss"], label="test_loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss (train vs test)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="best")

    # 標示最小驗證損失
    ax1.axvline(ep_min_vl, linestyle="--", alpha=0.6)
    ax1.annotate(
        f"min test_loss @ ep {ep_min_vl}\n(data['val_loss'][i_min_vl]:.4f)",
        xy=(ep_min_vl, data["val_loss"][i_min_vl]),
        xytext=(0, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=9, ha="center", annotation_clip=False
    )
    ax1.margins(x=0.05, y=0.10)

    # ------------------ Accuracy ------------------
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, data["train_acc"], label="train_acc", linewidth=2)
    ax2.plot(epochs, data["val_acc"], label="test_acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy (train vs test)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="best")

    # 標示最大驗證精度
    ax2.axvline(ep_max_va, linestyle="--", alpha=0.6)
    ax2.annotate(
        f"max test_acc @ ep {ep_max_va}\n(data['val_acc'][i_max_va]:.4f)",
        xy=(ep_max_va, data["val_acc"][i_max_va]),
        xytext=(0, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=9, ha="center", annotation_clip=False
    )
    ax2.margins(x=0.05, y=0.10)

    # 總標題
    fig.suptitle("Training Metrics", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"[OK] 圖片已輸出: {OUT_PATH}")

# ================== 程式入口 ==================
if __name__ == "__main__":
    main()
```

### RNN
* rnn.py
```py
import torch
import torch.nn as nn

# =================== 定義簡單 RNN 模型 ===================
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1, output_size=1):
        """
        初始化簡單 RNN 模型

        參數:
        input_size : int, 輸入特徵維度 (每個時間步的特徵數量)
        hidden_size: int, 隱藏層神經元數量 (RNN 每層的隱藏單元數)
        num_layers : int, RNN 層數
        output_size: int, 輸出特徵維度
        """
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定義 RNN 層
        # batch_first=True 表示輸入 x 的形狀是 [batch, seq_len, features]
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 定義全連接層 (linear layer)
        # 將最後的 hidden state 轉換為輸出
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        """
        RNN 前向運算

        參數:
        x: Tensor, 輸入序列, shape = [batch, seq_len, input_size]
        hidden: Tensor, 初始 hidden state, shape = [num_layers, batch, hidden_size]
        
        回傳:
        output: Tensor, RNN 輸出, shape = [batch, seq_len, output_size]
        hidden: Tensor, 最後 timestep 的 hidden state, shape = [num_layers, batch, hidden_size]
        """
        # 如果未提供 hidden state，初始化為全零
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            if x.is_cuda:  # 若在 GPU 上運算，hidden 也要放到 GPU
                hidden = hidden.cuda()

        # RNN 前向傳播
        # rnn_out: [batch, seq_len, hidden_size]
        # hidden: [num_layers, batch, hidden_size]
        rnn_out, hidden = self.rnn(x, hidden)

        # 將 RNN 輸出通過全連接層，得到最終輸出
        # output: [batch, seq_len, output_size]
        output = self.fc(rnn_out)
        
        return output, hidden

    def predict_next(self, x, hidden=None):
        """
        根據輸入序列 x 預測下一個時間步的值

        只取序列最後 timestep 的輸出作為預測值
        """
        output, hidden = self.forward(x, hidden)
        # 取最後一個時間步的輸出
        prediction = output[:, -1, :]  # shape: [batch, output_size]
        return prediction, hidden

# =================== 測試程式 ===================
if __name__ == "__main__":
    print("=" * 50)
    print("test rnn model")
    print("=" * 50)

    # 建立模型
    model = SimpleRNN(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    print("\nmodel structure:")
    print(model)

    # 建立測試輸入
    batch_size = 2
    sequence_length = 10
    input_size = 1
    # 隨機生成測試資料 [batch, seq_len, input_size]
    test_input = torch.randn(batch_size, sequence_length, input_size)

    # 前向測試 [batch, seq_len, output_size] => [2,10,1]
    output, hidden = model(test_input)
    print("\n輸出資料形狀 (output.shape):")
    print(f" - batch_size (批次大小): {output.shape[0]}") 
    print(f" - sequence_length (序列長度): {output.shape[1]}")
    print(f" - output_size (輸出特徵數): {output.shape[2]}")
    
    # 隱藏狀態形狀列印 [num_layers, batch, hidden_size] => [1,2,32]
    print("\n隱藏狀態形狀 (hidden.shape):")
    print(f" - num_layers (層數): {hidden.shape[0]}")
    print(f" - batch_size (批次大小): {hidden.shape[1]}")
    print(f" - hidden_size (隱藏層大小): {hidden.shape[2]}")

    # 測試 predict_next 方法 [2,1]
    prediction, hidden = model.predict_next(test_input) 
    print(f"\n預測下一個單一值形狀 (prediction.shape): {prediction.shape}")
    
    print("-" * 50)
    print("模型測試成功！")
    print("-" * 50)
```
* train.py
```py
import torch
import torch.nn as nn
import numpy as np
from rnn import SimpleRNN  # 引入自訂的 RNN 模型
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# =================== 訓練程式標題 ===================
print("-" * 60)
print("RNN訓練程式 - 使用正弦波資料")
print("-" * 60)

# =================== 超參數設定 ===================
SEQUENCE_LENGTH = 20    # 每個輸入序列長度
INPUT_SIZE = 1          # 輸入特徵維度 (sin波每個時間點只有一個值)
OUTPUT_SIZE = 1         # 輸出特徵維度
HIDDEN_SIZE = 64        # RNN 隱藏層大小
NUM_LAYERS = 1          # RNN 層數
NUM_EPOCHS = 200        # 訓練回合數
LEARNING_RATE = 0.001   # 學習率
BATCH_SIZE = 32         # 批次大小
NUM_SAMPLES = 1000      # 總樣本數

# 列印超參數
print(f"\nHyperparameters:")
print(f"序列長度: {SEQUENCE_LENGTH}")
print(f"隱藏層大小: {HIDDEN_SIZE}")
print(f"訓練回合: {NUM_EPOCHS}")
print(f"學習率: {LEARNING_RATE}")
print(f"批次大小: {BATCH_SIZE}")

# =================== 生成訓練資料 ===================
print("\n" + "-" * 60)
print("生成訓練資料")
print("-" * 60)

def generate_sine_wave_data(num_samples, sequence_length):
    """
    生成正弦波訓練資料
    
    回傳:
    X: 輸入序列, shape = [樣本數, 序列長度, 1]
    y: 對應目標值, shape = [樣本數, 1]
    """
    t = np.linspace(0, 4 * np.pi, num_samples)  # 生成時間點
    data = np.sin(t)                             # 正弦波數值

    X = []
    y = []
    # 建立滑動窗口
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length]
        X.append(sequence)
        y.append(target)

    # 轉換為 numpy array
    X = np.array(X)
    y = np.array(y)

    # 調整形狀: [樣本數, seq_len, 1]
    X = X.reshape(-1, sequence_length, 1)
    y = y.reshape(-1, 1)

    # 轉為 PyTorch tensor
    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    return X, y

# 生成訓練資料
X_train, y_train = generate_sine_wave_data(NUM_SAMPLES, SEQUENCE_LENGTH)

# 列印訓練資料資訊
print("訓練資料形狀：")
print(f"輸入 X: {X_train.shape} (樣本數, 序列長度, 特徵數)")
print(f"目標 y: {y_train.shape} (樣本數, 輸出數)")
print(f"總共有 {len(X_train)} 筆訓練資料")

# 封裝成 DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"\n每個 epoch 有 {len(train_loader)} 個批次")

# =================== 創建模型 ===================
print("\n" + "-" * 60)
print("創建模型")
print("-" * 60)

model = SimpleRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE
)
print(model)
print(f"\n模型參數總數: {sum(p.numel() for p in model.parameters())}")

# 定義損失函數和優化器
criterion = nn.MSELoss()  # 均方誤差
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =================== 開始訓練 ===================
print("\n" + "-" * 60)
print("開始訓練")
print("-" * 60)

loss_history = []
for epoch in range(NUM_EPOCHS):
    model.train()  # 訓練模式
    epoch_loss = 0.0

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()            # 梯度清零
        output, _ = model(batch_X)       # 前向傳播
        prediction = output[:, -1, :]    # 取最後 timestep 作為預測
        loss = criterion(prediction, batch_y)  # 計算 MSE
        loss.backward()                  # 反向傳播
        optimizer.step()                 # 更新權重
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {avg_loss:.6f}")

print("\n訓練完成！")

# =================== 儲存模型 ===================
print("\n" + "-" * 60)
print("儲存模型")
print("-" * 60)

model_path = "rnn_model.pth"
torch.save(model.state_dict(), model_path)
print(f"模型已儲存至: {model_path}")

# =================== 繪製訓練損失曲線 ===================
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Training Loss Over Time", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (MSE)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
print("訓練損失曲線已儲存為: training_loss.png")

# =================== 生成預測結果範例 ===================
print("\n生成預測結果範例...")
model.eval()  # 評估模式
with torch.no_grad():
    test_X = X_train[:100]
    test_y = y_train[:100]
    output, _ = model(test_X)
    predictions = output[:, -1, :].numpy()  # 取最後 timestep

    # 繪製預測結果與真實值
    plt.figure(figsize=(12, 5))
    plt.plot(test_y.numpy(), 'b.-', label='True', linewidth=2)
    plt.plot(predictions, 'r.--', label='Predicted', linewidth=2)
    plt.title("RNN Prediction vs True Values", fontsize=14)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150)
    print("預測結果圖已儲存為: prediction_result.png")

# =================== 結束 ===================
print("\n" + "-" * 60)
print("所有訓練流程完成！")
print("-" * 60)
print("生成的檔案：")
print(" 1. rnn_model.pth 訓練好的模型權重")
print(" 2. training_loss.png 損失曲線圖")
print(" 3. prediction_result.png 預測結果範例圖")
print("\n可執行 test.py 來載入模型做新資料測試。")
```