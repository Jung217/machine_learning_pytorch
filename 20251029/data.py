import torch # 導入 PyTorch 庫，用於張量操作和深度學習
import torchvision # 導入 TorchVision 庫，包含數據集、模型和圖像轉換工具
import torchvision.transforms as transforms # 導入圖像轉換模塊
from torch.utils.data import DataLoader, random_split # 導入 DataLoader 和隨機劃分數據集函數
import matplotlib.pyplot as plt # 導入 Matplotlib 庫，用於繪圖和圖像顯示
import numpy as np # 導入 NumPy 庫，用於數組操作
import os # 導入 OS 庫，用於文件系統操作 (如創建目錄)
from PIL import Image # 導入 PIL (Pillow) 庫，用於圖像處理

# --- 1. 定義數據轉換 (Preprocessing) ---

transform = transforms.Compose([
    # 將 PIL 圖像或 NumPy ndarray 轉換為 PyTorch 張量 (H x W x C) -> (C x H x W)
    transforms.ToTensor(), 
    # 對張量進行標準化: (x - mean) / std
    # CIFAR-10 的均值和標準差 (RGB 三個通道)
    # (0.5, 0.5, 0.5) 均值將數據範圍從 [0, 1] 轉換到 [-0.5, 0.5]
    # (0.5, 0.5, 0.5) 標準差將數據範圍從 [-0.5, 0.5] 轉換到 [-1, 1]
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- 2. 載入 CIFAR-10 數據集 ---
# 

# 載入訓練集 (train=True)
trainset = torchvision.datasets.CIFAR10(
    root='./datasets/cifar10', # 數據集存放目錄
    train=True, # 載入訓練數據
    download=False, # 假設數據集已下載 (如果為 True 且不存在則自動下載)
    transform=transform # 應用上面定義的轉換
)

# 載入測試集 (train=False)
testset = torchvision.datasets.CIFAR10(
    root='./datasets/cifar10',
    train=False, # 載入測試數據
    download=False,
    transform=transform
)

# 打印數據集基本資訊
print(f"train data size:{len(trainset)}")
print(f"test data size:{len(testset)}")
print("class =", len(trainset.classes))
print("class names =", trainset.classes)
print("class to idx =", trainset.class_to_idx)

# 查看第一個樣本的形狀和標籤
img, lbl = trainset[0]
print("single img shape =", img.shape, "label =", lbl, trainset.classes[lbl]) # 輸出應為 [3, 32, 32]

# --- 3. 劃分訓練集和驗證集 (Train/Validation Split) ---

# 計算訓練集大小 (80%)
train_size = int(0.8 * len(trainset))
# 計算驗證集大小 (剩餘 20%)
val_size = len(trainset) - train_size

# 使用 random_split 函數將原始 trainset 劃分成兩個子集
train_dataset, val_dataset = random_split(
    trainset, # 原始訓練集
    [train_size, val_size], # 劃分大小
    # 設置種子確保每次劃分結果一致
    generator=torch.Generator().manual_seed(9527) 
)

# 打印劃分後的數據集大小
print(f"train dataset size:{len(train_dataset)}")
print(f"validation dataset size:{len(val_dataset)}")
print(f"test dataset size:{len(testset)}")

# --- 4. 創建數據載入器 (DataLoaders) ---

batch_size = 32 # 每個批次的樣本數量
# 訓練數據載入器
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False, # 此處設置為 False，通常訓練集會設置為 True
    num_workers=0 # 0 表示在主進程中載入數據，>0 表示使用多進程加速
)
# 驗證數據載入器
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False, # 驗證集和測試集通常不打亂
    num_workers=0
)
# 測試數據載入器
test_loader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# 打印批次數量
print(f"train batches:{len(train_loader)}")
print(f"validation batches:{len(val_loader)}")
print(f"test batches:{len(test_loader)}")

# --- 5. 圖像顯示函數 (針對 PyTorch 張量) ---

def imshow(img):
    # 反標準化操作: 將範圍從 [-1, 1] 變回 [0, 1]
    img = img / 2 + 0.5 
    # 將 PyTorch 張量轉換為 NumPy 數組
    npimg = img.numpy() 
    # 原代碼這裡缺少繪圖指令
    # plt.imshow() 
    # plt.axis('off')

# --- 6. 顯示樣本圖像並保存 ---

# 獲取一個批次的訓練數據
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 設置 Matplotlib 繪圖
fig = plt.figure(figsize=(12, 6))
fig.suptitle('CIFAR-10', fontsize=16, fontweight='bold')

classes = trainset.classes

# 循環顯示前 8 張圖像
for idx in range(8):
    # 創建子圖 (2行4列)
    ax = plt.subplot(2, 4, idx+1) 
    img = images[idx]
    
    # 執行反標準化 (將數據範圍從 [-1, 1] 變回 [0, 1])
    img = img / 2 + 0.5 
    npimg = img.numpy()
    
    # 調整維度順序: PyTorch 是 (C, H, W)，Matplotlib 需要 (H, W, C)
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    # 設置標題為圖像類別名稱
    plt.title(f'class:{classes[labels[idx]]}') 
    # 隱藏座標軸
    plt.axis('off') 

plt.tight_layout() # 自動調整子圖間距
os.makedirs('./outputs', exist_ok=True) # 創建輸出目錄 (如果不存在)
# 將圖表保存為 PNG 文件
plt.savefig('./outputs/cifar10_sample.png', dpi=150, bbox_inches='tight') 
print("img saved")