import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try: from mamba_ssm import Mamba
except ImportError:
    class Mamba(nn.Module):
        def __init__(self, d_model, d_state, d_conv, expand):
            super().__init__()
            self.layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                batch_first=True,
            )
        def forward(self, x):
            return self.layer(x)
import numpy as np


class SimpleTextDataset(Dataset):
    def __init__(self, texts, vocab, seq_length=64):
        self.texts = texts
        self.vocab = vocab
        self.seq_length = seq_length
        self.full_text = "".join(texts)

    def __len__(self):
        return max(0, len(self.full_text) - self.seq_length)

    def __getitem__(self, idx):
        seq = self.full_text[idx:idx + self.seq_length + 1]
        
        input_ids = torch.tensor([self.vocab.get(c, 0) for c in seq[:-1]], dtype=torch.long)
        target_ids = torch.tensor([self.vocab.get(c, 0) for c in seq[1:]], dtype=torch.long)
   
        return input_ids, target_ids


class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        
        for mamba_layer in self.mamba_layers: x = mamba_layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x) 
        
        return logits
        
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=None):
        self.eval() 
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
        return input_ids

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    
    for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)

        loss = nn.functional.cross_entropy(
            logits.view(-1, model.vocab_size),
            target_ids.view(-1)                
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sample_text = "the quick brown fox jumps over the lazy dog."
    chars = sorted(set(sample_text))
    vocab = {c: i for i, c in enumerate(chars)}
    vocab_size = len(vocab)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {chars}")

    texts = [sample_text * 10]
    dataset = SimpleTextDataset(texts, vocab, seq_length=32)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print(f"Dataset length (sequences): {len(dataset)}")

    model = MambaLanguageModel(
        vocab_size=vocab_size,
        d_model=64, 
        n_layers=2, 
        d_state=16, 
        d_conv=4, 
        expand=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable parameters: {total_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n--- Training initiated ---")
    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch} finished. Average loss: {avg_loss:.4f}")

    print("\n--- Text generation initiated ---")
    model.eval()

    start_text = "the"

    input_ids = torch.tensor(
        [[vocab.get(c, 0) for c in start_text]],
        dtype=torch.long
    ).to(device)
    
    generated_ids = model.generate(input_ids, max_length=50, temperature=0.8)

    reverse_vocab = {i: c for c, i in vocab.items()}

    generated_text = "".join([
        reverse_vocab.get(id.item(), '?') 
        for id in generated_ids[0]
    ])
    
    print(f"Generated text: {generated_text}")