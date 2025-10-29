import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from rnn import SimpleRNN

# -----------------------------
# 超參數
# -----------------------------
SEQUENCE_LENGTH = 20
INPUT_SIZE = 1
OUTPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_EPOCHS = 500
LEARNING_RATE = 0.001
BATCH_SIZE = 32

print(f"\nHyperparameters:")
print(f"序列長度: {SEQUENCE_LENGTH}")
print(f"隱藏層大小: {HIDDEN_SIZE}")
print(f"訓練回合: {NUM_EPOCHS}")
print(f"學習率: {LEARNING_RATE}")
print(f"批次大小: {BATCH_SIZE}")

# -----------------------------
# 讀取並標準化資料
# -----------------------------
df = pd.read_csv("./AirPassengers.csv")
df['Passengers'] = df['Passengers'].astype(float)
data = df['Passengers'].values

mean = np.mean(data)
std = np.std(data)
data = (data - mean) / std

# -----------------------------
# 生成監督式訓練資料
# -----------------------------
def make_supervised(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    X = np.array(X)
    y = np.array(y)

    X = X.reshape(-1, sequence_length, 1)
    y = y.reshape(-1, 1)

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)
    return X, y

X_train, y_train = make_supervised(data, SEQUENCE_LENGTH)

print("訓練資料形狀：")
print(f"輸入 X: {X_train.shape}")
print(f"目標 y: {y_train.shape}")
print(f"總共有 {len(X_train)} 筆訓練資料")

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"\n每個 epoch 有 {len(train_loader)} 個批次")

# -----------------------------
# 創建模型
# -----------------------------
model = SimpleRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE
)
print(model)
print(f"\n模型參數總數: {sum(p.numel() for p in model.parameters())}")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# 訓練
# -----------------------------
loss_history = []
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        output, _ = model(batch_X)
        prediction = output[:, -1, :]
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {avg_loss:.6f}")

print("\n訓練完成！")

# -----------------------------
# 儲存模型與繪圖
# -----------------------------
torch.save(model.state_dict(), "rnn_model.pth")
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png")

# -----------------------------
# 預測範例
# -----------------------------
model.eval()
with torch.no_grad():
    test_X = X_train[:100]
    test_y = y_train[:100]
    output, _ = model(test_X)
    predictions = output[:, -1, :].numpy()

plt.figure(figsize=(12, 5))
plt.plot(test_y.numpy(), 'b.-', label='True')
plt.plot(predictions, 'r.--', label='Predicted')
plt.legend()
plt.title("RNN Prediction vs True Values")
plt.grid(True)
plt.tight_layout()
plt.savefig("prediction_result.png")

print("\ndone")
