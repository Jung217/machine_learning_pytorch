import torch # 導入 PyTorch 庫
import torchvision # 導入 TorchVision 庫
import torchvision.transforms as transforms # 導入圖像轉換模塊
from torch.utils.data import DataLoader, random_split # 導入數據載入器和隨機劃分函數
import matplotlib.pyplot as plt # 導入 Matplotlib 繪圖庫
import numpy as np # 導入 NumPy 數組操作庫
import os # 導入 OS 庫，用於文件系統操作
from PIL import Image # 導入 PIL 庫 (Pillow)，用於圖像處理 (在保存圖片時用到)
from torchvision.datasets import ImageFolder # 導入 ImageFolder 類，用於從文件夾結構中載入圖像數據集

# --- 1. 定義路徑和類別 ---

data_root = "./datasets2/cifar10_folders" # 數據集根目錄
train_dir = os.path.join(data_root, 'train') # 訓練數據文件夾路徑
test_dir = os.path.join(data_root, 'test') # 測試數據文件夾路徑

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] # CIFAR-10 類別名稱

# --- 2. 數據集文件結構創建邏輯 ---

# 檢查目標數據集文件夾是否已存在
if not os.path.exists(data_root):
    os.makedirs(data_root, exist_ok=True) # 創建根目錄
    transform_to_pil = transforms.ToPILImage() # 實際上此處未用到，因為 CIFAR10 數據集返回的就是 PIL Image

    # 載入原始 CIFAR-10 數據集
    cifar_train = torchvision.datasets.CIFAR10(
        root='./dataset2/cifar10',
        train=True,
        download=True # 如果不存在則下載
    )
    cifar_test = torchvision.datasets.CIFAR10(
        root='./dataset2/cifar10',
        train=False,
        download=True
    )

    # 為每個類別在 train 和 test 目錄下創建子文件夾 (ImageFolder 要求的文件結構)
    # 預期結構: data_root/train/airplane/, data_root/train/automobile/, ...
    for class_name in classes: os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    for class_name in classes: os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    print("saving...")
    
    # --- 提取訓練集子集 (每個類別 100 張) ---
    class_counts = {i: 0 for i in range(10)}
    for idx, (img, label) in enumerate(cifar_train):
        if class_counts[label] < 100:
            # 將 PIL 圖像保存到對應類別的文件夾中，文件名為計數 (e.g., train/airplane/0.png)
            img.save(os.path.join(train_dir, classes[label], f'{class_counts[label]}.png'))
            class_counts[label] += 1
        # 檢查是否所有類別都提取了 100 張
        if all(count >= 100 for count in class_counts.values()): break

    # --- 提取測試集子集 (每個類別 100 張) ---
    class_counts = {i: 0 for i in range(10)}
    for idx, (img, label) in enumerate(cifar_test):
        if class_counts[label] < 100:
            img.save(os.path.join(test_dir, classes[label], f'{class_counts[label]}.png'))
            class_counts[label] += 1
        # 檢查是否所有類別都提取了 100 張
        if all(count >= 100 for count in class_counts.values()): break
    print("done")
else:
    print("Dataset already exists")

# --- 3. 定義圖像轉換 (Transforms) ---

# 訓練集轉換 (包含數據增強)
train_transform = transforms.Compose([
    transforms.Resize((32, 32)), # 調整圖像大小 (保持 CIFAR 原始尺寸 32x32)
    transforms.RandomHorizontalFlip(), # 數據增強: 隨機水平翻轉
    transforms.RandomRotation(10), # 數據增強: 隨機旋轉最多 10 度
    transforms.ToTensor(), # 將圖像轉換為 PyTorch 張量 (HWC -> CHW, 0-255 -> 0-1)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 標準化 (範圍約為 [-1, 1])
])

# 測試集轉換 (無數據增強，只有標準化和張量轉換)
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- 4. 使用 ImageFolder 載入數據集 ---
# 

# 載入完整的訓練數據集，應用 train_transform
full_train_dataset = ImageFolder(
    root=train_dir,
    transform=train_transform
)

# 載入測試數據集，應用 test_transform
test_dataset = ImageFolder(
    root=test_dir,
    transform=test_transform
)

# --- 5. 訓練集/驗證集拆分 (Split) ---

# 拆分比例: 85% 訓練，15% 驗證
train_size = int(0.85 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

# 隨機拆分 full_train_dataset 成 train_dataset 和 val_dataset
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(9527) # 固定隨機拆分結果
)

# --- 6. 創建數據載入器 (DataLoaders) ---

batch_size = 32
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True, # 訓練集需要打亂
    num_workers=0
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False, # 驗證集不需要打亂
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False, # 測試集不需要打亂
    num_workers=0
)

# --- 7. 顯示一個批次的樣本圖像 ---

dataiter = iter(train_loader)
images, labels = next(dataiter)

fig = plt.figure(figsize=(12, 6))
fig.suptitle('ImageFolder sapmle', fontsize=16, fontweight='bold')

classes = full_train_dataset.classes # 從 ImageFolder 對象獲取類別名稱

# 循環顯示前 8 張圖像
for idx in range(8):
    ax = plt.subplot(2, 4, idx+1)
    img = images[idx]
    
    # 反標準化: 將範圍從 [-1, 1] 變回 [0, 1]
    img = img / 2 + 0.5 
    npimg = img.numpy()
    
    # 調整維度順序: (C, H, W) -> (H, W, C) 以便 Matplotlib 顯示
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.title(f'class:{classes[labels[idx]]}')
    plt.axis('off')

plt.tight_layout()
os.makedirs('./outputs2', exist_ok=True)
# 將圖表保存為 PNG 文件
plt.savefig('./outputs2/imagefolder_sample.png', dpi=150, bbox_inches='tight') 
print("img saved")