from __future__ import annotations
import argparse
import random
import math
from dataclasses import dataclass
from typing import Tuple, Literal, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_positional_encoding(max_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

def make_dataset_ends_equal(n: int, L: int, vocab: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randint(0, vocab, (n, L))
    y = torch.zeros(n, dtype=torch.long)
    half = n // 2
    X[:half, -1] = X[:half, 0]
    y[:half] = 1
    for i in range(half, n):
        a = X[i, 0].item()
        b = random.randint(0, vocab - 2)
        if b >= a:
            b += 1
        X[i, -1] = b
        y[i] = 0
    idx = torch.randperm(n)
    return X[idx], y[idx]

def make_dataset_compare_ij(n: int, L: int, vocab: int, i: int, j: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert 0 <= i < L and 0 <= j < L and i != j
    X = torch.randint(0, vocab, (n, L))
    y = (X[:, i] > X[:, j]).long()
    return X, y

@dataclass
class TinyConfig:
    vocab: int = 100
    d_model: int = 128
    num_heads: int = 4
    dropout: float = 0.0

class TinyEncoder(nn.Module):
    def __init__(self, cfg: TinyConfig):
        super().__init__()
        self.emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.mha = nn.MultiheadAttention(cfg.d_model, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.cls = nn.Linear(cfg.d_model, 2)

    def forward(self, x_ids: torch.Tensor, pe: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.emb(x_ids)
        if pe is not None:
            h = h + pe[:, : h.size(1), :].to(h.dtype)
        y, attn = self.mha(h, h, h, need_weights=True)
        h = self.norm(h + y)
        pooled = h.mean(dim=1)
        return self.cls(pooled), attn

def train_one(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, pe: Optional[torch.Tensor]) -> Tuple[float, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb, pe=pe)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, total_correct / total

@torch.no_grad()
def eval_one(model: nn.Module, loader: DataLoader, device: torch.device, pe: Optional[torch.Tensor]) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits, _ = model(xb, pe=pe)
        loss = loss_fn(logits, yb)
        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, total_correct / total

def run_tiny_task_student_version(task, use_pe:bool): #--task compare_ij --i 1 --j 10 --len 12 --n 6000 --epoch 10 --pe none
    if use_pe: X, y = make_dataset_ends_equal(6000, 12, 100)
    else: X, y = make_dataset_compare_ij(6000, 12, 100, 1, 10)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train = int(0.8 * 6000)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:], y[n_train:]

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=128, shuffle=False)

    cfg = TinyConfig(vocab=100, d_model=128, num_heads=4, dropout=0.0)
    model = TinyEncoder(cfg).to(device)

    pe = None
    if use_pe: pe = get_positional_encoding(12, cfg.d_model, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for ep in range(1, 10 + 1):
        tl, ta = train_one(model, train_loader, optim, device, pe)
        vl, va = eval_one(model, val_loader, device, pe)
        print(f"[EP {ep:02d}] train loss={tl:.4f} acc={ta:.3f} | val loss={vl:.4f} acc={va:.3f}")

    return 0



def main():
    parser = argparse.ArgumentParser(description="Tiny tasks for MHA + Positional Encoding")
    parser.add_argument("--task", type=str, choices=["ends_equal", "compare_ij"], default="ends_equal")
    parser.add_argument("--i", type=int, default=0)
    parser.add_argument("--j", type=int, default=-1)
    parser.add_argument("--len", type=int, default=12)
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--vocab", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pe", type=str, choices=["none", "sincos"], default="sincos")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L = args.len
    vocab = args.vocab
    n = args.n
    j = args.j if args.j >= 0 else L - 1

    if args.task == "ends_equal":
        X, y = make_dataset_ends_equal(n, L, vocab)
    else:
        X, y = make_dataset_compare_ij(n, L, vocab, i=args.i, j=j)

    n_train = int(0.8 * n)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva = X[n_train:], y[n_train:]

    train_loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva, yva), batch_size=args.batch_size, shuffle=False)

    cfg = TinyConfig(vocab=vocab, d_model=args.d_model, num_heads=args.heads, dropout=args.dropout)
    model = TinyEncoder(cfg).to(device)

    pe = None
    if args.pe == "sincos":
        pe = get_positional_encoding(L, cfg.d_model, device=device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"[INFO] task={args.task} pe={args.pe} L={L} n={n} d_model={cfg.d_model} heads={cfg.num_heads}")
    for ep in range(1, args.epochs + 1):
        tl, ta = train_one(model, train_loader, optim, device, pe)
        vl, va = eval_one(model, val_loader, device, pe)
        print(f"[EP {ep:02d}] train loss={tl:.4f} acc={ta:.3f} | val loss={vl:.4f} acc={va:.3f}")

if __name__ == "__main__":
    #main()
    print("---No PE (use_pe=False)---")
    run_tiny_task_student_version(task="ends_equal", use_pe=False)
    print("---Yes PE (use_pe=True)---")
    run_tiny_task_student_version(task="ends_equal", use_pe=True)