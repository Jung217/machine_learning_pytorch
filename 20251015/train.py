import torch
import torch.nn as nn
import numpy as np
from rnn import SimpleRNN
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

print("-" * 60)
print("RNN訓練程式 - 使用正弦波資料")
print("-" * 60)

SEQUENCE_LENGTH = 20
INPUT_SIZE = 1
OUTPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_LAYERS = 1
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_SAMPLES = 1000

print(f"\nHyperparameters:")
print(f"序列長度: {SEQUENCE_LENGTH}")
print(f"隱藏層大小: {HIDDEN_SIZE}")
print(f"訓練回合: {NUM_EPOCHS}")
print(f"學習率: {LEARNING_RATE}")
print(f"批次大小: {BATCH_SIZE}")

print("\n" + "-" * 60)
print("生成訓練資料")
print("-" * 60)

def generate_sine_wave_data(num_samples, sequence_length):
    t = np.linspace(0, 4 * np.pi, num_samples)
    data = np.sin(t)

    X = []
    y = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length]
        X.append(sequence)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(-1, sequence_length, 1)
    y = y.reshape(-1, 1)

    X = torch.FloatTensor(X)
    y = torch.FloatTensor(y)

    return X, y

X_train, y_train = generate_sine_wave_data(NUM_SAMPLES, SEQUENCE_LENGTH)

print("訓練資料形狀：")
print(f"輸入 X: {X_train.shape} (樣本數, 序列長度, 特徵數)")
print(f"目標 y: {y_train.shape} (樣本數, 輸出數)")
print(f"總共有 {len(X_train)} 筆訓練資料")

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"\n每個 epoch 有 {len(train_loader)} 個批次")

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

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("\n" + "-" * 60)
print("開始訓練")
print("-" * 60)

loss_history = []
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        optimizer.zero_grad()
        output, _ = model(batch_X)
        prediction = output[:, -1, :] 
        loss = criterion(prediction, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {avg_loss:.6f}")

print("\n訓練完成！")

print("\n" + "-" * 60)
print("儲存模型")
print("-" * 60)

model_path = "rnn_model.pth"
torch.save(model.state_dict(), model_path)
print(f"模型已儲存至: {model_path}")

plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Training Loss Over Time", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (MSE)", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
print("訓練損失曲線已儲存為: training_loss.png")

print("\n生成預測結果範例...")
model.eval()
with torch.no_grad():
    test_X = X_train[:100]
    test_y = y_train[:100]
    output, _ = model(test_X)
    predictions = output[:, -1, :].numpy()

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

print("\n" + "-" * 60)
print("所有訓練流程完成！")
print("-" * 60)
print("生成的檔案：")
print(" 1. rnn_model.pth 訓練好的模型權重")
print(" 2. training_loss.png 損失曲線圖")
print(" 3. prediction_result.png 預測結果範例圖")
print("\n可執行 test.py 來載入模型做新資料測試。")