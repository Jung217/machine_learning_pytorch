import os
from pathlib import Path
import random
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


from cnn import CNN # 你的模型
import csv


from matplotlib import font_manager


# ============ 可隨意修改 ============
DATA_DIR = "data"
OUT_DIR = "result"
CKPT_PATH = Path(OUT_DIR) / "best_cnn.pth"
ROWS, COLS = 2, 5
SEED = 20
MEAN, STD = 0.2861, 0.3530
FIGSIZE = (16, 8) # 圖片大小
TITLE_FZ = 11 # 標題字體大小
ZH_FONT_CANDIDATES = [
r"C:\Windows\Fonts\msjh.ttc",
r"C:\Windows\Fonts\msjhl.ttc",
r"C:\Windows\Fonts\mingliu.ttc",
r"C:\Windows\Fonts\simsun.ttc",
]


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_zh_font():
    """全域設定一個可用的中文字型,避免中文顯示方塊"""
    from pathlib import Path as Path
    # The loop 'for p in ZH_FONT_CANDIDATES:' is inferred from the context
    for p in ZH_FONT_CANDIDATES:
        if Path(p).exists():
            prop = font_manager.FontProperties(fname=p)
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["axes.unicode_minus"] = False # 讓負號正常顯示
            print(f"[INFO] 使用中文字型: {prop.get_name()} ({p})")
            return
    print("[WARN] 找不到中文字型,請安裝「思源正黑體」Noto Sans CJK」或更新 ZH_FONT_CANDIDATES")




def denorm_img(x):
    """普通還原張量 [H,W] or [1,H,W], 把 Normalize 還原到 0~1"""
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    x = x * STD + MEAN
    return x.clamp(0, 1)


def main():
    set_seed(SEED)
    set_zh_font()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Transform 與訓練一致
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MEAN,), (STD,))
    ])


    # 讀取 test set (官方 10,000 張)
    test_set = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=tfm)


    # 準備模型並載入最佳權重
    model = CNN(num_classes=10).to(device)
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        state = ckpt.get("model_state", ckpt) # 兼容只存 state_dict 或整包的情況
        model.load_state_dict(state)
        print(f"[INFO] loaded checkpoint: {CKPT_PATH}")
    else:
        print(f"[WARN] checkpoint not found: {CKPT_PATH}，使用隨機初始化權重 (僅示意)")


    model.eval()


    # 抽樣 N 張
    N = ROWS * COLS
    indices = list(range(len(test_set)))
    random.shuffle(indices)
    indices = indices[:N]


    # 推論
    images = []
    gts = []
    preds = []
    confs = []


    with torch.no_grad():
        for idx in indices:
            img_t, label = test_set[idx] # img_t: [1,H,W] (normalized)
            logits = model(img_t.unsqueeze(0).to(device)) # [1,10]
            prob = F.softmax(logits, dim=1)[0]       # [10]
            pred_id = int(torch.argmax(prob).item())
            conf = float(prob[pred_id].item())


            images.append(img_t.cpu())
            gts.append(label)
            preds.append(pred_id)
            confs.append(conf)




    # 畫圖
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)


    for i in range(N):
        ax = fig.add_subplot(ROWS, COLS, i + 1)
        img = denorm_img(images[i])
        ax.imshow(img, cmap="gray", interpolation="nearest") # 保留像素風格
        ax.axis("off")


        gt_name = CLASS_NAMES[gts[i]]
        pred_name = CLASS_NAMES[preds[i]]
        conf_txt = f"{confs[i]:.3f}"


        # 三行標字: 真實 / 預測 / 信心度
        title = f"真實:{gt_name}\n預測:{pred_name}\n信心度:{conf_txt}"
        ax.set_title(title, color="green", fontsize=TITLE_FZ, pad=6)


    # 大標題 (略高於子圖)
    fig.suptitle("模型預測結果", fontsize=16, y=0.98)


    out_path = Path(OUT_DIR) / "test_grid.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 已輸出: {out_path.resolve()}")


if __name__ == "__main__":
    main()