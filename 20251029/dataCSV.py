import torch                                            # 導入 PyTorch 庫
import torchvision                                      # 導入 TorchVision 庫，用於數據集
import torchvision.transforms as transforms             # 圖像轉換 (雖然在這段程式碼中未直接使用 transform 參數，但仍習慣性導入)
from torch.utils.data import DataLoader, random_split   # 數據處理工具
import matplotlib.pyplot as plt     # 繪圖庫
import numpy as np                  # 數組操作庫
import os                           # 文件系統操作庫
import pandas as pd                 # 導入 Pandas 庫，用於處理和創建 CSV 文件

# --- 1. 定義路徑和準備目錄 ---

data_root = "./datasets3/cifar10_folders" # 根目錄名稱
imges_dir = os.path.join(data_root, 'images') # 圖片存放子目錄路徑
os.makedirs(imges_dir, exist_ok=True) # 創建圖片存放目錄 (如果目錄已存在則不報錯)

CSV_path = os.path.join(data_root, 'label.csv') # CSV 標籤文件的完整路徑

# --- 2. 數據集提取和保存邏輯 ---

# 檢查 CSV 文件是否已存在。如果不存在，則執行數據集生成邏輯。
if not os.path.exists(CSV_path):
    # 載入完整的 CIFAR-10 訓練集
    cifar_dataset = torchvision.datasets.CIFAR10(
        root='./dataset3/cifar10', # 原始 CIFAR-10 數據的下載目錄
        train=True, # 載入訓練數據
        download=True # 如果本地不存在，則下載
    )
    
    data_list = [] # 用於收集要寫入 CSV 的字典列表
    # CIFAR-10 的類別名稱
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # 追蹤每個類別已經提取的圖片數量
    class_counts = {i: 0 for i in range(10)} 
    
    # 遍歷原始 CIFAR-10 訓練集
    for idx, (img, label) in enumerate(cifar_dataset):
        # 判斷當前類別 (label) 的圖片數量是否小於 10
        if class_counts[label] < 10:
            # 構造圖片文件名: e.g., 'airplane_0.png', 'cat_3.png'
            img_name = f'{classes[label]}_{class_counts[label]}.png'
            img_path = os.path.join(imges_dir, img_name)
            
            # 將 PIL 圖像對象保存到指定路徑
            img.save(img_path)

            # 將該圖片的資訊添加到列表中
            data_list.append({
                'image_name': img_name,
                'label': label, # 數字標籤 (0-9)
                'class_name':classes[label] # 類別名稱
            })

            # 增加該類別的計數
            class_counts[label] += 1
            
        # 檢查是否所有 10 個類別都已經收集了至少 10 張圖片
        if all(count >= 10 for count in class_counts.values()): 
            break # 如果滿足條件，則跳出循環，停止提取

    # 將收集到的列表轉換為 Pandas DataFrame
    df = pd.DataFrame(data_list)
    # 將 DataFrame 寫入 CSV 文件
    df.to_csv(CSV_path, index=False) # index=False 表示不寫入行索引
    
    # 打印完成訊息
    print(f"created {len(df)} data entries") # 預期是 10 * 10 = 100 條
    print(f"file at {CSV_path}")
    print(f"Img directory {imges_dir}")

# 如果 CSV 文件已存在 (即數據集已準備好)，則直接載入 CSV
else:
    print("Dataset already exists")
    df = pd.read_csv(CSV_path) # 讀取 CSV 文件
    print(f"Load {len(df)} entries")