import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Now using {device}")

raw_data = [(), 
            (),
            (),
            (),
            (),
            ()
]

class SimpleTokenizer:
    def __init__(self, data, lang_idx):
        self.word2idx = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}

        vocab = set()
        for pair in data:
            sentence = pair[lang_idx]
            if lang_idx == 0: words = sentence.split()
            else: words = list(sentence)
            vocab.updata(words)

        for i, word in enumerate(vocab):
            self.word2idx[word] = i+4
            self.idx2word[i+4] = word
        
    def encode(self, text, lang_type="en"):
        words = text.split() if lang_type == "en" else list(text)
        return [self.word2idx.get(w, 3) for w in words]
    
    def decode(self, indices):
        return "".join([self.idx2word.get(idx, "") for idx in indices if idx not in [0, 1, 2]])
    
scr_tokenizer = SimpleTokenizer(raw_data, 0)
tgt_tokenizer = SimpleTokenizer(raw_data, 1)

class TranslationDataset(Dataset):
    def __init__(self, data, src_tok, tgt_tok):
        self.data = data
        self.src_tok = src_tok
        self.tgt_tok = tgt_tok
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_ids = [1] + self.src_tok.encode(src_text, "en") + [2]
        tgt_ids = [1] + self.tgt_tok.encode(tgt_text, "ch") + [2]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)
    
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(src_sample)
            tgt_batch.append(tgt_sample)
    
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
        tgt_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)

        return src_padded, tgt_padded
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_layers=3, dropot=0.1)
        super().__init__()
        self.d_model = d_model

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        self.transfomer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=2048,
            dropout=dropot,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transfomer.generate_square_subsequent_mask(tgt_seq_len).to(device)

        src_padding_mask = (src==0)
        tgt_padding_mask = (tgt==0)

        outs = self.transfomer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask = src_padding_mask,
            tgt_key_padding_mask = tgt_padding_mask,
            memory_key_padding_mask = src_padding_mask
        )

        return self.fc_out(outs)