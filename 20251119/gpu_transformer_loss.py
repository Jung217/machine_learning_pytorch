# -*- coding: utf-8 -*-
# ===============================
# Transformer 英→中 翻譯（教學版）
# ===============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import warnings
import matplotlib.pyplot as plt


# 關閉 PyTorch Transformer 常見的 warning（不影響正確性）
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------
# 1. 裝置設定（GPU / MPS / CPU）
# -------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"目前使用的運算裝置: {device}")

# --------------------------------
# 2. 平行語料（Parallel Corpus）
# --------------------------------
# 每一筆資料 = (英文句子, 中文句子)
raw_data = [
    ("I love AI", "我愛人工智慧"),
    ("Deep learning is fun", "深度學習很有趣"),
    ("Transformer is powerful", "變形金剛很強大"),
    ("This is a long sentence to test padding mechanism", "這是一個用來測試填充機制的長句子"),
    ("GPU makes training faster", "GPU讓訓練變快"),
    ("Seq2Seq model is cool", "序列對序列模型很酷"),
]

# --------------------------------
# 3. 極簡 Tokenizer
# --------------------------------
class SimpleTokenizer:
    """
    功能：
    - 英文：用空格切詞
    - 中文：用「單字元」切詞
    - 建立 word2idx / idx2word
    """

    def __init__(self, data, lang_idx):
        # 特殊符號（一定要有）
        self.word2idx = {
            "<PAD>": 0,  # padding
            "<BOS>": 1,  # sentence begin
            "<EOS>": 2,  # sentence end
            "<UNK>": 3   # unknown
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

        # 蒐集 vocab
        vocab = set()
        for pair in data:
            sentence = pair[lang_idx]
            if lang_idx == 0:          # 英文
                words = sentence.split()
            else:                       # 中文
                words = list(sentence)
            vocab.update(words)

        # 將 vocab 加入字典
        for i, word in enumerate(vocab):
            self.word2idx[word] = i + 4
            self.idx2word[i + 4] = word

    def encode(self, text, lang_type="en"):
        """
        將句子轉成 index list
        """
        words = text.split() if lang_type == "en" else list(text)
        return [self.word2idx.get(w, 3) for w in words]  # 3 = <UNK>

    def decode(self, indices):
        """
        將 index list 還原成文字
        移除 <PAD>, <BOS>, <EOS>
        """
        return "".join(
            [self.idx2word.get(idx, "") for idx in indices if idx not in [0, 1, 2]]
        )

# 建立 source / target tokenizer
src_tokenizer = SimpleTokenizer(raw_data, 0)
tgt_tokenizer = SimpleTokenizer(raw_data, 1)

# --------------------------------
# 4. Dataset + Padding
# --------------------------------
class TranslationDataset(Dataset):
    """
    Dataset 負責：
    - 取資料
    - 加上 <BOS>, <EOS>
    """

    def __init__(self, data, src_tok, tgt_tok):
        self.data = data
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]

        # Encoder input: <BOS> ... <EOS>
        src_ids = [1] + self.src_tok.encode(src_text, "en") + [2]
        # Decoder input / target
        tgt_ids = [1] + self.tgt_tok.encode(tgt_text, "ch") + [2]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    """
    將不同長度句子 padding 成相同長度
    """
    src_batch, tgt_batch = [], []

    for src, tgt in batch:
        src_batch.append(src)
        tgt_batch.append(tgt)

    # padding_value=0 對應 <PAD>
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return src_padded, tgt_padded

# --------------------------------
# 5. Seq2Seq Transformer 模型
# --------------------------------
class Seq2SeqTransformer(nn.Module):
    """
    Encoder-Decoder Transformer
    """

    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=256, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 詞嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # PyTorch 內建 Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )

        # 輸出層
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # Embedding + scale（Transformer 標準做法）
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # Decoder causal mask（防止偷看未來）
        tgt_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(device)

        # Padding mask
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)

        # Transformer forward
        out = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        return self.fc_out(out)

# --------------------------------
# 6. 訓練流程（Teacher Forcing）
# --------------------------------
def train():
    BATCH_SIZE = 2
    EPOCHS = 200
    LR = 1e-4

    dataset = TranslationDataset(raw_data, src_tokenizer, tgt_tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = Seq2SeqTransformer(
        len(src_tokenizer.word2idx),
        len(tgt_tokenizer.word2idx)
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    print("開始訓練...")

    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_input)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f}")

    return model, loss_history

# --------------------------------
# 7. 推論（Autoregressive decoding）
# --------------------------------
def translate(model, sentence):
    model.eval()

    src_ids = [1] + src_tokenizer.encode(sentence, "en") + [2]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    tgt_ids = [1]  # <BOS>

    print(f"\n原文: {sentence}")
    print("翻譯中...")

    for _ in range(20):
        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)

        next_token = logits[0, -1].argmax().item()
        tgt_ids.append(next_token)

        if next_token == 2:  # <EOS>
            break

    print("結果:", tgt_tokenizer.decode(tgt_ids))

def translate_text(model, sentence, max_len=20):
    model.eval()

    src_ids = [1] + src_tokenizer.encode(sentence, "en") + [2]
    src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)

    tgt_ids = [1]  # <BOS>

    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)

        next_token = logits[0, -1].argmax().item()
        tgt_ids.append(next_token)

        if next_token == 2:
            break

    return tgt_tokenizer.decode(tgt_ids)

def save_loss_and_translation(loss_history, model):
    # 測試句子
    test_sentences = [
        "I love AI",
        "Deep learning is fun",
        "Transformer is powerful"
    ]

    translations = []
    for s in test_sentences:
        pred = translate_text(model, s)
        translations.append((s, pred))

    # ====== 畫圖 ======
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # --- (1) Loss Curve ---
    axes[0].plot(loss_history, marker="o")
    axes[0].set_title("Training Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)

    # --- (2) Translation Results (Text) ---
    axes[1].axis("off")

    text = "Translation Results\n\n"
    for i, (src, tgt) in enumerate(translations, 1):
        text += f"{i}. EN: {src}\n"
        text += f"   ZH: {tgt}\n\n"

    axes[1].text(
        0.01, 0.99, text,
        fontsize=11,
        verticalalignment="top",
        family="monospace"
    )

    plt.tight_layout()
    plt.savefig("training_result.png")
    plt.close()

    print("已儲存圖檔：training_result.png")

def save_loss_only(loss_history):

    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, marker="o")
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.close()

    print("已儲存圖檔：training_loss.png")



# --------------------------------
# 8. 主程式
# --------------------------------
if __name__ == "__main__":
    model, loss_history = train()
    save_loss_only(loss_history)
    #model = train()
    translate(model, "I love AI")
    translate(model, "Deep learning is fun")