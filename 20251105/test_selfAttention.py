import math
import numpy as np
from typing import Optional, Tuple
import torch

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float = 0.0):
    if TORCH_AVAILABLE:
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)

        if dropout_p and dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        output = attn_weights @ v
        return output, attn_weights
    
    else:
        d_k = q.shape[-1]
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        scores_max = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - scores_max)
        attn_weights = exp / exp.sum(axis=-1, keepdims=True)
        
        output = np.matmul(attn_weights, v)
        return output, attn_weights

def build_padding_mask(lengths, max_len):
    try:
        device = lengths.device if hasattr(lengths, "device") else "cpu"
        rng = torch.arange(max_len, device=device).unsqueeze(0)
        mask = (rng < lengths.unsqueeze(1)).to(torch.bool)
        return mask.unsqueeze(1).unsqueeze(1)
    except Exception:
        lengths = np.asarray(lengths)
        rng = np.arange(max_len)[None, :]
        mask = (rng < lengths[:, None])
        return mask[:, None, None, :]

def build_look_ahead_mask(seq_len):
    try:
        return torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool))
    except Exception:
        m = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        return m[None, None, :, :]

if __name__ == "__main__":
    print(f"TORCH_AVAILABLE = {TORCH_AVAILABLE}")
    
    try:
        torch.manual_seed(7)
        B, H, L, D = 2, 2, 5, 4

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        out, attn = scaled_dot_product_attention(q, k, v, mask=None)
        print("[No Mask] output shape:", tuple(out.shape))
        print("[No Mask] attn shape:", tuple(attn.shape))
        print("Row sums (應為 1):", attn[0, 0, 0].sum().item())

        lengths = torch.tensor([3, 5])
        pad_mask = build_padding_mask(lengths, max_len=L)
        out_padded, attn_padded = scaled_dot_product_attention(q, k, v, mask=pad_mask)
        print("[Padding Mask] attn[0,0,0]:", attn_padded[0, 0, 0])
        print("Check pad columns (index>=3) 應接近 0:", attn_padded[0, 0, 0, 3:].sum().item())

        look_mask = build_look_ahead_mask(L)
        la_mask = look_mask.expand(B, H, L, L)
        out_la, attn_la = scaled_dot_product_attention(q, k, v, mask=la_mask)
        print("[Look-ahead Mask] 上三角是否為 0? (印第1列):", attn_la[0, 0, 0])
        print("[Look-ahead Mask] 上三角是否為 0? (印第2列):", attn_la[0, 0, 1])
        print("[Look-ahead Mask] 上三角是否為 0? (印第3列):", attn_la[0, 0, 2])
        print("[Look-ahead Mask] 上三角是否為 0? (印第4列):", attn_la[0, 0, 3])
        print("[Look-ahead Mask] 上三角是否為 0? (印第5列):", attn_la[0, 0, 4])


        scores = torch.tensor([
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        weights = torch.softmax(scores, dim=-1)
        print("3x3 softmax範例\n", weights)

    except Exception as e:
        print(f"\nPyTorch tests failed or not available ({e}), running NumPy tests...")
        
        np.random.seed(7)
        B, H, L, D = 2, 2, 5, 4
        q = np.random.randn(B, H, L, D)
        k = np.random.randn(B, H, L, D)
        v = np.random.randn(B, H, L, D)

        out, attn = scaled_dot_product_attention(q, k, v, mask=None)
        print("[No Mask][NP] output shape:", out.shape)
        print("[No Mask][NP] attn shape:", attn.shape)
        print("Row sums (應為 1):", attn[0, 0, 0].sum())

        lengths = np.array([3, 5])
        pad_mask = build_padding_mask(lengths, max_len=L)
        out_padded, attn_padded = scaled_dot_product_attention(q, k, v, mask=pad_mask)
        print("[Padding Mask][NP] attn[0,0,0]:", attn_padded[0, 0, 0])
        print("Check pad columns (index>=3) 應接近 0:", attn_padded[0, 0, 0, 3:].sum())

        look_mask = build_look_ahead_mask(L)
        la_mask = np.broadcast_to(look_mask, (B, H, L, L))
        out_la, attn_la = scaled_dot_product_attention(q, k, v, mask=la_mask)
        print("[Look-ahead Mask][NP] 上三角是否為 0? (印第1列):", attn_la[0, 0, 0])
        print("[Look-ahead Mask][NP] 上三角是否為 0? (印第2列):", attn_la[0, 0, 1])
        print("[Look-ahead Mask][NP] 上三角是否為 0? (印第3列):", attn_la[0, 0, 2])
        print("[Look-ahead Mask][NP] 上三角是否為 0? (印第4列):", attn_la[0, 0, 3])
        print("[Look-ahead Mask][NP] 上三角是否為 0? (印第5列):", attn_la[0, 0, 4])

        scores = np.array([
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores)
        weights = exp / exp.sum(axis=-1, keepdims=True)
        print("[NP] 3x3 softmax範例\n", weights)
