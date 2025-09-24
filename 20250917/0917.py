import os, numpy as np, pandas as pd
from typing import List
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import joblib

import torch
from torch.utils.data import TensorDataset, DataLoader

from torch import nn

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
X_val_t = torch.tensor(X_train_sc, dtype=torch.float32)
y_yal_t = torch.tensor(X_train_sc, dtype=torch.long)

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

class IrisMLP(n.module):
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

