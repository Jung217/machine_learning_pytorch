import csv
from pathlib import Path
import matplotlib.pyplot as plt
import math

CSV_PATH = Path("result/loss.csv")
OUT_PATH = Path("result/metrics.png")

def read_metrics(csv_path: Path):
    cols = {k: [] for k in ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]}

    try:
        f = open(csv_path, "r", encoding="utf-8-sig")
        first_try = f
    except Exception:
        f = open(csv_path, "r", encoding="utf-8")
        first_try = f
    
    with first_try as ff:
        reader = csv.DictReader(ff)
        if not reader.fieldnames: raise RuntimeError("CSV without headers!")
        
        fmap = {k.strip().lower(): k for k in reader.fieldnames}
        need = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
        miss = [k for k in need if k not in fmap]

        if miss: raise RuntimeError(f"necessary col miss:{miss};find col:{reader.fieldnames}")
        
        for row in reader:
            try:
                e = int(float(row[fmap["epoch"]]))
                tl = float(row[fmap["train_loss"]])
                vl = float(row[fmap["val_loss"]])
                ta = float(row[fmap["train_acc"]])
                va = float(row[fmap["val_acc"]])
            except Exception:
                continue

            cols["epoch"].append(e)
            cols["train_loss"].append(tl)
            cols["val_loss"].append(vl)
            cols["train_acc"].append(ta)
            cols["val_acc"].append(va)

    if not cols["epoch"]: raise RuntimeError("No any valid data read")
    return cols

def argmin(lst):
    bi, bv = None, math.inf
    for i, v in enumerate(lst): 
        if v < bv: bi, bv = i, v
    return bi

def argmax(lst):
    bi, bv = None, -math.inf
    for i, v in enumerate(lst): 
        if v > bv: bi, bv = i, v
    return bi

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = read_metrics(CSV_PATH)
    epochs = data["epoch"]

    i_min_vl = argmin(data["val_loss"])
    i_max_va = argmax(data["val_acc"])
    ep_min_vl = epochs[i_min_vl]
    ep_max_va = epochs[i_max_va]

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)


    # -- loss --
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, data["train_loss"], label="train_loss", linewidth=2)
    ax1.plot(epochs, data["val_loss"], label="test_loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss (train vs test)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="best")
    ax1.axvline(ep_min_vl, linestyle="--", alpha=0.6)
    ax1.annotate(
        f"min test_loss @ ep {ep_min_vl}\n(data['val_loss'][i_min_vl]:.4f)",
        xy=(ep_min_vl, data["val_loss"][i_min_vl]),
        xytext=(0, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=9, ha="center", annotation_clip=False
    )
    ax1.margins(x=0.05, y=0.10)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, data["train_acc"], label="train_acc", linewidth=2)
    ax2.plot(epochs, data["val_acc"], label="test_acc", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy (train vs test)")
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="best")
    ax2.axvline(ep_max_va, linestyle="--", alpha=0.6)
    ax2.annotate(
        f"max test_acc @ ep {ep_max_va}\n(data['val_acc'][i_max_va]:.4f)",
        xy=(ep_max_va, data["val_acc"][i_max_va]),
        xytext=(0, 15), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=9, ha="center", annotation_clip=False
    )
    ax2.margins(x=0.05, y=0.10)

    fig.suptitle("Training Metrics", fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    print(f"Output at {OUT_PATH}")

if __name__ == "__main__":
    main()
