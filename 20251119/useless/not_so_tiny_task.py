from __future__ import annotations
import argparse
import random
import math
from dataclasses import dataclass
from typing import Tuple, Literal, Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# 導入 Matplotlib 庫用於繪圖
import matplotlib.pyplot as plt

# --- 1. 實用函數 (保持不變) ---

def set_seed(seed: int = 1337):
    """設置隨機種子以確保結果可重現。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_positional_encoding(max_len: int, d_model: int, device: torch.device) -> torch.Tensor:
    """計算 Sinusoidal 位置編碼 (Sinusoidal Positional Encoding)。"""
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=device) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)

def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
    """生成上三角形的遮罩 (Look-Ahead Mask)，用於 Decoder 的自注意力層。"""
    mask = torch.triu(torch.full((size, size), float('-inf'), device=device), diagonal=1)
    return mask

def make_padding_mask(X: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """生成 Padding Mask，用於遮蔽填充 token。"""
    # 輸出形狀: (Batch_size, Sequence_Length)
    return (X == pad_idx)


# --- 2. 數據集生成 (Seq2Seq 任務演示) ---

def make_dataset_seq2seq(n: int, L_src: int, L_tgt: int, vocab: int, sos_idx: int, eos_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成一個演示用的 Seq2Seq 數據集 (將序列反轉)，已修復長度邏輯。
    
    Returns:
        X_src (Encoder Input), X_tgt_input (Decoder Input), X_tgt_output (Target Label)
    """
    # 目標序列的內容長度 = L_tgt - 1 (因為最後是 EOS)
    content_len = L_tgt - 1 
    
    # 確保源序列足夠長度 (內容長度 + SOS + EOS)
    min_src_len = content_len + 2
    if L_src < min_src_len:
        # 僅用於警告/調整，確保不會因為長度不足而報錯
        # 在 main 函數中，我們使用 argparse 參數，這裡保持原始 L_src
        pass 
    
    # 1. 隨機內容張量 (長度: content_len)
    # 避開特殊 token (PAD=0, SOS=1, EOS=2)
    X_content = torch.randint(3, vocab, (n, content_len)) 
    
    # 2. 構建源序列 (Encoder Input)
    # Src: [SOS, x1, x2, ..., xL, EOS, PAD...]
    src_full = torch.zeros((n, L_src), dtype=torch.long) # 預設為 padding
    src_full[:, 0] = sos_idx
    
    # 確保複製長度不超過 L_src-2 (為 SOS/EOS 預留空間)
    len_to_copy = min(content_len, L_src - 2)
    if len_to_copy > 0:
        src_full[:, 1:1+len_to_copy] = X_content[:, :len_to_copy]
    
    # 在序列結束處加上 EOS (如果長度允許)
    if 1 + len_to_copy < L_src:
        src_full[:, 1+len_to_copy] = eos_idx
    
    # 3. 構建目標序列 (Target Output - Label)
    # Tgt (Output): [y1, y2, ..., yL_content, EOS, PAD...]
    tgt_output = torch.zeros((n, L_tgt), dtype=torch.long) # 預設為 padding
    
    # 反轉 X_content (長度 content_len)
    tgt_content = torch.flip(X_content, [1])
    
    # 複製反轉內容
    tgt_output[:, :content_len] = tgt_content
    
    # 在序列末尾加上 EOS (索引為 content_len)
    if content_len < L_tgt:
        tgt_output[:, content_len] = eos_idx 
    
    # 4. 構建目標序列 (Decoder Input)
    # Tgt (Input): [SOS, y1, y2, ..., yL_content, PAD...]
    tgt_input = torch.zeros((n, L_tgt), dtype=torch.long) # 預設為 padding
    tgt_input[:, 0] = sos_idx
    # 複製 Tgt Output 的內容 (不包含最後的 EOS)
    tgt_input[:, 1:] = tgt_output[:, :-1] 
    
    return src_full, tgt_input, tgt_output

# --- 3. Transformer 配置和核心模塊 (保持不變) ---

@dataclass
class TransformerConfig:
    vocab_size: int = 100
    pad_idx: int = 0
    sos_idx: int = 1
    eos_idx: int = 2
    d_model: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    max_len: int = 50

class EncoderLayer(nn.Module):
    """一個標準的 Transformer Encoder Layer。"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力子層: x + Dropout(MHA(Norm(x)))
        norm_x = self.norm1(x)
        # padding_mask: (Batch, Src_Len) -> MHA 內部會處理成 (Batch, 1, Src_Len) 或 (Batch, Src_Len)
        attn_output, _ = self.mha(norm_x, norm_x, norm_x, key_padding_mask=padding_mask, need_weights=False) 
        x = x + self.dropout1(attn_output)

        # 前饋網路子層: x + Dropout(FFN(Norm(x)))
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output
        
        return x

class DecoderLayer(nn.Module):
    """一個標準的 Transformer Decoder Layer。"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        
        # 1. 遮蔽自注意力子層
        self.norm1 = nn.LayerNorm(d_model)
        self.masked_mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        # 2. 交叉注意力子層
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        
        # 3. 前饋網路 (FFN) 子層
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, 
                 self_attn_mask: Optional[torch.Tensor] = None, # 因果遮罩 (Seq_len, Seq_len)
                 self_padding_mask: Optional[torch.Tensor] = None, # 自注意力 padding 遮罩 (Batch, Tgt_Len)
                 cross_padding_mask: Optional[torch.Tensor] = None # 交叉注意力 padding 遮罩 (Batch, Src_Len)
                ) -> torch.Tensor:
        
        # 1. 遮蔽自注意力
        norm_x = self.norm1(x)
        # attn_mask: (Tgt_Len, Tgt_Len), key_padding_mask: (Batch, Tgt_Len)
        attn_output, _ = self.masked_mha(
            norm_x, norm_x, norm_x, 
            attn_mask=self_attn_mask, 
            key_padding_mask=self_padding_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_output)

        # 2. 交叉注意力 (K/V 來自 Encoder Output)
        norm_x = self.norm2(x)
        # key_padding_mask: (Batch, Src_Len)
        cross_attn_output, _ = self.cross_mha(
            norm_x, encoder_output, encoder_output, 
            key_padding_mask=cross_padding_mask, 
            need_weights=False
        ) 
        x = x + self.dropout2(cross_attn_output)
        
        # 3. 前饋網路
        norm_x = self.norm3(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output
        
        return x

class FullTransformer(nn.Module):
    """完整的 Transformer Seq2Seq 模型。"""
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_idx)
        self.dropout = nn.Dropout(cfg.dropout)
        self.d_model = cfg.d_model
        self.pad_idx = cfg.pad_idx
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(cfg.d_model, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(cfg.d_model, cfg.num_heads, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        
        self.output_linear = nn.Linear(cfg.d_model, cfg.vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """權重初始化 (與原始論文相似的策略)"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
             nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, 
                 src_ids: torch.Tensor, 
                 tgt_ids: torch.Tensor, 
                 pe: torch.Tensor
                ) -> torch.Tensor:
        
        device = src_ids.device
        
        # 1. 遮罩準備
        # Src padding mask: (Batch, Src_Len)
        src_padding_mask = make_padding_mask(src_ids, self.pad_idx) 
        # Tgt padding mask: (Batch, Tgt_Len)
        tgt_padding_mask = make_padding_mask(tgt_ids, self.pad_idx)
        
        tgt_len = tgt_ids.size(1)
        # Tgt causal mask: (Tgt_Len, Tgt_Len)
        tgt_causal_mask = generate_square_subsequent_mask(tgt_len, device)
        
        # --- Encoder 階段 ---
        src = self.token_embed(src_ids) * math.sqrt(self.d_model)
        # 加位置編碼 (PE)
        src = src + pe[:, :src.size(1), :]
        src = self.dropout(src)

        encoder_output = src
        for layer in self.encoder_layers:
            # 傳遞 src_padding_mask (key_padding_mask for Self-Attention)
            encoder_output = layer(encoder_output, padding_mask=src_padding_mask)
        
        # --- Decoder 階段 ---
        tgt = self.token_embed(tgt_ids) * math.sqrt(self.d_model)
        # 加位置編碼 (PE)
        tgt = tgt + pe[:, :tgt.size(1), :]
        tgt = self.dropout(tgt)

        decoder_output = tgt
        for layer in self.decoder_layers:
            # self_attn_mask: 因果遮罩 (Look-ahead)
            # self_padding_mask: Tgt 序列的 Padding 遮罩
            # cross_padding_mask: Src 序列的 Padding 遮罩 (用於 Cross-Attention 的 K/V)
            decoder_output = layer(
                decoder_output, 
                encoder_output, 
                self_attn_mask=tgt_causal_mask, 
                self_padding_mask=tgt_padding_mask,
                cross_padding_mask=src_padding_mask
            )

        # --- 輸出階段 ---
        logits = self.output_linear(decoder_output)
        return logits

# --- 4. 訓練和評估函數 (保持不變) ---

def train_one(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, pe: torch.Tensor, cfg: TransformerConfig) -> Tuple[float, float]:
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_idx, reduction='sum') # 改變為 sum
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    
    for xb_src, xb_tgt_in, yb_tgt_out in loader:
        xb_src, xb_tgt_in, yb_tgt_out = xb_src.to(device), xb_tgt_in.to(device), yb_tgt_out.to(device)
        
        logits = model(xb_src, xb_tgt_in, pe=pe)
        
        loss = loss_fn(logits.view(-1, cfg.vocab_size), yb_tgt_out.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        mask = (yb_tgt_out != cfg.pad_idx)
        current_tokens = mask.sum().item()
        if current_tokens > 0:
            pred = logits.argmax(dim=-1)
            total_correct += (pred == yb_tgt_out)[mask].sum().item()
            total_tokens += current_tokens
            total_loss += loss.item() # loss 已經是 sum
        
    return total_loss / total_tokens if total_tokens > 0 else 0.0, total_correct / total_tokens if total_tokens > 0 else 0.0

@torch.no_grad()
def eval_one(model: nn.Module, loader: DataLoader, device: torch.device, pe: torch.Tensor, cfg: TransformerConfig) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_idx, reduction='sum') # 改變為 sum
    total_loss, total_correct, total_tokens = 0.0, 0, 0

    for xb_src, xb_tgt_in, yb_tgt_out in loader:
        xb_src, xb_tgt_in, yb_tgt_out = xb_src.to(device), xb_tgt_in.to(device), yb_tgt_out.to(device)
        
        logits = model(xb_src, xb_tgt_in, pe=pe)
        
        loss = loss_fn(logits.view(-1, cfg.vocab_size), yb_tgt_out.view(-1))
        
        mask = (yb_tgt_out != cfg.pad_idx)
        current_tokens = mask.sum().item()
        if current_tokens > 0:
            pred = logits.argmax(dim=-1)
            total_correct += (pred == yb_tgt_out)[mask].sum().item()
            total_tokens += current_tokens
            total_loss += loss.item()
        
    return total_loss / total_tokens if total_tokens > 0 else 0.0, total_correct / total_tokens if total_tokens > 0 else 0.0

# --- 5. 主執行邏輯 (新增圖表繪製) ---

def plot_metrics(history: dict, epochs: int):
    """繪製損失和準確度圖表。"""
    epochs_range = range(1, epochs + 1)
    
    # 設置圖表大小
    plt.figure(figsize=(12, 5)) 

    # 1. 繪製損失曲線
    plt.subplot(1, 2, 1) # 1行2列的第1個圖
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--')

    # 2. 繪製準確度曲線
    plt.subplot(1, 2, 2) # 1行2列的第2個圖
    plt.plot(epochs_range, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--')

    plt.tight_layout() # 自動調整子圖參數，使之填充整個圖形區域
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Full Transformer Seq2Seq Model Demo")
    parser.add_argument("--len_src", type=int, default=13, help="Source sequence length (including SOS/EOS)")
    parser.add_argument("--len_tgt", type=int, default=12, help="Target sequence length (including EOS)")
    parser.add_argument("--n", type=int, default=6000, help="Total number of samples")
    parser.add_argument("--vocab", type=int, default=100, help="Vocabulary size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cfg = TransformerConfig(
        vocab_size=args.vocab, d_model=args.d_model, num_heads=args.heads, 
        num_layers=args.layers, dropout=args.dropout, max_len=max(args.len_src, args.len_tgt),
        pad_idx=0, sos_idx=1, eos_idx=2
    )

    # 數據準備
    X_src, X_tgt_in, X_tgt_out = make_dataset_seq2seq(
        args.n, args.len_src, args.len_tgt, cfg.vocab_size, cfg.sos_idx, cfg.eos_idx
    )
    
    n_train = int(0.8 * args.n)
    Xtr_src, Xtr_tgt_in, Ytr_tgt_out = X_src[:n_train], X_tgt_in[:n_train], X_tgt_out[:n_train]
    Xva_src, Xva_tgt_in, Yva_tgt_out = X_src[n_train:], X_tgt_in[n_train:], X_tgt_out[n_train:]
    
    train_loader = DataLoader(TensorDataset(Xtr_src, Xtr_tgt_in, Ytr_tgt_out), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xva_src, Xva_tgt_in, Yva_tgt_out), batch_size=args.batch_size, shuffle=False)

    model = FullTransformer(cfg).to(device)
    # 位置編碼只需要計算一次
    pe = get_positional_encoding(cfg.max_len, cfg.d_model, device=device) 
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"[INFO] Full Transformer Model initialized on {device}")
    print(f"       d_model={cfg.d_model}, heads={cfg.num_heads}, layers={cfg.num_layers}, L_src={X_src.size(1)}, L_tgt={X_tgt_in.size(1)}")
    
    # 訓練歷史記錄字典
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # 訓練循環
    for ep in range(1, args.epochs + 1):
        tl, ta = train_one(model, train_loader, optim, device, pe, cfg)
        vl, va = eval_one(model, val_loader, device, pe, cfg)
        
        # 記錄數據
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_acc'].append(ta)
        history['val_acc'].append(va)
        
        print(f"[EP {ep:02d}] train loss={tl:.4f} acc={ta:.3f} | val loss={vl:.4f} acc={va:.3f}")

    # 繪製圖表
    print("\n--- 訓練結果圖表 ---")
    plot_metrics(history, args.epochs)


if __name__ == "__main__":
    # 使用 try/except 確保即使在沒有 argparse 參數的情況下也能運行
    try:
        main()
    except SystemExit:
        # 如果在 IDE 中運行 main()，可能會拋出 SystemExit，我們忽略它
        pass
    except Exception as e:
        # 如果是其他錯誤，則打印出來
        print(f"An error occurred: {e}")