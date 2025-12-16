import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# 嘗試使用官方 mamba_ssm 套件
# 若沒安裝，退化成 Transformer Encoder 當 placeholder
try:
    from mamba_ssm import Mamba
except ImportError:
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state, d_conv, expand):
            super().__init__()
            # fallback：用 Transformer Encoder 模擬 Mamba 行為
            self.layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                batch_first=True,
            )

        def forward(self, x):
            return self.layer(x)

import numpy as np


# ===============================
# 簡單字元級 Dataset
# ===============================
class SimpleTextDataset(Dataset):
    def __init__(self, texts, vocab, seq_length=64):
        self.texts = texts                      # 原始文本 list
        self.vocab = vocab                      # char → id 字典
        self.seq_length = seq_length            # 每筆序列長度
        self.full_text = "".join(texts)         # 將所有文字串接成一條長字串

    def __len__(self):
        # 能取多少個滑動視窗
        return max(0, len(self.full_text) - self.seq_length)

    def __getitem__(self, idx):
        # 取長度 = seq_length + 1（input + target）
        seq = self.full_text[idx:idx + self.seq_length + 1]
        
        # input: 前 seq_length 個字元
        input_ids = torch.tensor(
            [self.vocab.get(c, 0) for c in seq[:-1]],
            dtype=torch.long
        )

        # target: 向右 shift 一個字元（next token prediction）
        target_ids = torch.tensor(
            [self.vocab.get(c, 0) for c in seq[1:]],
            dtype=torch.long
        )
   
        return input_ids, target_ids


# ===============================
# Mamba 語言模型（Decoder-only）
# ===============================
class MambaLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 疊多層 Mamba block
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            )
            for _ in range(n_layers)
        ])
        
        # 最後正規化
        self.norm = nn.LayerNorm(d_model)

        # 語言模型輸出頭（logits）
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        # input_ids: (batch, seq_len)
        
        # embedding → (batch, seq_len, d_model)
        x = self.embedding(input_ids)
        
        # 經過多層 Mamba
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)
        
        # layer norm
        x = self.norm(x)

        # 預測每個位置的 vocab logits
        logits = self.lm_head(x)
        
        return logits
        
    # ===============================
    # 自回歸文字生成
    # ===============================
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None):
        self.eval() 
        
        with torch.no_grad():
            # 一個 token 一個 token 生
            for _ in range(max_length - input_ids.shape[1]):
                logits = self.forward(input_ids)

                # 只取最後一個位置
                next_token_logits = logits[:, -1, :] / temperature

                # top-k sampling（可選）
                if top_k is not None:
                    threshold = torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    indices_to_remove = next_token_logits < threshold
                    next_token_logits[indices_to_remove] = float('-inf')

                # softmax → 機率
                probs = torch.softmax(next_token_logits, dim=-1)

                # 依機率抽樣
                next_token = torch.multinomial(probs, num_samples=1)

                # 接到序列後面
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
        return input_ids


# ===============================
# 單一 epoch 訓練流程
# ===============================
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        # forward
        logits = model(input_ids)

        # CrossEntropyLoss（flatten batch 與時間維）
        loss = F.cross_entropy(
            logits.view(-1, model.vocab_size),
            target_ids.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # 每 10 個 batch 印一次
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ===============================
# 主程式
# ===============================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 範例訓練文字
    sample_text = "the quick brown fox jumps over the lazy dog."

    # 建立 char-level vocab
    chars = sorted(set(sample_text))
    vocab = {c: i for i, c in enumerate(chars)}
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {chars}")

    # 重複文本以增加資料量
    texts = [sample_text * 10]

    # Dataset / Dataloader
    dataset = SimpleTextDataset(texts, vocab, seq_length=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset length (sequences): {len(dataset)}")

    # 建立模型
    model = MambaLanguageModel(
        vocab_size=vocab_size,
        d_model=64,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    # 參數數量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ===== 訓練 =====
    print("\n--- Training initiated ---")
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1} finished. Average loss: {avg_loss:.4f}")

    # ===== 生成文字 =====
    print("\n--- Text generation initiated ---")
    model.eval()

    start_text = "the"

    # 將起始字串轉成 id
    input_ids = torch.tensor(
        [[vocab.get(c, 0) for c in start_text]],
        dtype=torch.long
    ).to(device)
    
    generated_ids = model.generate(
        input_ids,
        max_length=50,
        temperature=0.8
    )

    # id → char
    reverse_vocab = {i: c for c, i in vocab.items()}
    generated_text = "".join(
        reverse_vocab.get(i.item(), '?')
        for i in generated_ids[0]
    )
    
    print(f"Generated text: {generated_text}")
