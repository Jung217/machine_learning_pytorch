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

ROOT = "iris_course"
ARTIFACTS = os.path.join(ROOT, "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

def describe_stats(X: np.ndarray, names: List[str], title: str):
    m, s = X.mean(axis=0), X.std(axis=0)
    print(f"\n[{title}]")
    for n, mi, sd in zip(names, m, s): print(f"{n:14s} mean={mi:8.4f} std={sd:8.4f}")

print("=== STEP 1 | load data and explore")
iris = load_iris()
x, y = iris.data, iris.target
feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target_names = iris.target_names.tolist()

df = pd.DataFrame(x, columns=feature_names)
df["targets"] = y
print("\n前 5 筆資料:")
print(df.head())
print("\n類別分布:")
for i, name in enumerate(target_names): print(f" {i}={name:<10s} : {(y==i).sum()} 筆")
describe_stats(x, feature_names, "origin data (not normalized)")

out_csv = os.path.join(ARTIFACTS, "iris_preview.csv")
df.head(20).to_csv(out_csv, index=False)
print(f"\n save 20 data:{out_csv}")
print("STEP 1 finished\n")

print("\n=== STEP 2 | split train/val/test ===")
X_trainval, X_test, y_trainval, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval)
print(f"切分形狀: train={X_train.shape} val={X_val.shape} test={X_test.shape}")
print("STEP 2 finished\n")

print("=== STEP 3 | normalize (only train set fit) and save")
scaler = StandardScaler().fit(X_train)
X_train_sc = scaler.transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

describe_stats(X_train, feature_names, "train set before normalize")
describe_stats(X_train_sc, feature_names, "train set after normalize")

npz_path = os.path.join(ARTIFACTS, "train_val_test_scaled.npz")
np.savez(npz_path, X_train_sc=X_train_sc, y_train=y_train,
         X_val_sc=X_val_sc, y_val=y_val,
         X_test_sc=X_test_sc, y_test=y_test,
         feature_names=np.array(feature_names, dtype=object),
         target_names=np.array(target_names, dtype=object))
scaler_path = os.path.join(ARTIFACTS, "scalar.pkl")
joblib.dump(scaler, scaler_path)
print(f"save normalize data to {npz_path}")
print(f"save 標準化器 to {scaler_path}")
print("STEP 3 finished\n")

print("=== STEP 4 | tensor and dataloader")
X_train_t = torch.tensor(X_train_sc, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val_sc, dtype=torch.float32)
y_yal_t = torch.tensor(y_val, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_t, y_yal_t), batch_size=16, shuffle=False)

xb, yb = next(iter(train_loader))
print(f"NO.1 batch:xb.shape={xb.shape}, yb.shape={yb.shape}")
print(f" xb[0](normalized) = {xb[0].tolist()}")
print(f" yb[0](type) = {yb[0].item}")

batch_preview = os.path.join(ARTIFACTS, "batch_preview.csv")
pd.DataFrame(xb.numpy(), columns=feature_names).assign(label=yb.numpy()).to_csv(batch_preview, index=False)
print(f"save batch preview to {batch_preview}")
print("STEP 4 finished\n")

print("=== STEP 5 | define model and parameters")
MODELS = os.path.join(ROOT, "models")
os.makedirs(MODELS, exist_ok=True)

class IrisMLP(nn.Module):
    def __init__(self, in_dim=4, hidden1=64, hidden2=32, out_dim=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden2, out_dim)
        )
    def forward(self, x): return self.net(x)

def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IrisMLP().to(device)
print(model)
print(f"trainable parameters : {count_trainable_params(model):,}")

arch_txt = os.path.join(MODELS, "model_arch.txt")
with open(arch_txt, "w", encoding="utf-8") as f:
    f.write(str(model) + "\n")
    f.write(f"trainable parameters : {count_trainable_params(model)}\n")
print(f"->saved structure describe:{arch_txt}")
print("STEP 5 finished\n")

print("\n== STEP 6 [訓練][含早停]==")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(m, loader):
    m.eval()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = m(xb)
            print(logits)
            loss_sum += criterion(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return loss_sum/total, correct/total

best_state, best_val, patience, bad = None, 1e9, 15, 0
hist = {"tr_loss":[], "tr_acc":[], "va_loss":[], "va_acc":[]}

for ep in range(1, 201):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item() * xb.size(0)
        correct += (logits.argmax(1) == yb).sum().item()
        total += yb.size(0)
    tr_loss, tr_acc = loss_sum/total, correct/total
    va_loss, va_acc = evaluate(model, val_loader)
    hist["tr_loss"].append(tr_loss); hist["tr_acc"].append(tr_acc)
    hist["va_loss"].append(va_loss); hist["va_acc"].append(va_acc)
    print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | val_loss={va_loss:.4f} acc={va_acc:.3f}")

    if va_acc > best_val: best_val, best_state, bad = va_acc, {k: v.cpu() for k, v in model.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= patience: print(f"早停：{patience} epochs 未提升。"); break

# 載回最佳模型
if best_state: model.load_state_dict(best_state)

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

# best model saved
best_path = os.path.join(MODELS, "best.pt")
torch.save(model.state_dict(), best_path)
print(f"=> best weight saved：{best_path}")
print("STEP 6 finished")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("\n== STEP 7 test & evaluate==")
model.eval()
with torch.no_grad():
    logits = model(torch.tensor(X_test_sc, dtype=torch.float32).to(device))
    y_pred = logits.argmax(1).cpu().numpy()

acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.3f}")
print("分類報告：")
print(classification_report(y_test, y_pred, target_names=target_names, digits=3))

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

# 回到原始位置看測試樣本
X_test_orig = scaler.inverse_transform(X_test_sc)
show_n = min(10, len(X_test_orig))
tbl = pd.DataFrame(X_test_orig[:show_n], columns=feature_names)
tbl["true"] = [target_names[i] for i in y_test[:show_n]]
tbl["pred"] = [target_names[i] for i in y_pred[:show_n]]
samples_csv = os.path.join(ARTIFACTS, "test_samples.csv")
