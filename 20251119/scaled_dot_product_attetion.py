import torch, math
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

try:
    import torch, math
    import torch.nn.functional as F
    TORCH_AVA = True
except Exception as e:
    TORCH_AVA = False


def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float = 0.0):
    if TORCH_AVA:
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        if dropout_p and dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    else:
        d_k = q.shape[-1]
        scores = np.matmul(q, np.swapaxes(k, -2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        score_max = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - score_max)
        attn_weights = exp / exp.sum(axis=-1, keepdims=True)
        output = np.matmul(attn_weights, v)

        return output, attn_weights


def build_padding_mask(lengths, max_len):
    """建立 padding mask: shape => (B, 1, 1, max_len)"""
    try:
        device = lengths.device if hasattr(lengths, "device") else "cpu"
        rng = torch.arange(max_len, device=device).unsqueeze(0)
        mask = (rng < lengths.unsqueeze(1)).to(torch.bool)
        return mask.unsqueeze(1).unsqueeze(1)
    except Exception:
        batch_size = len(lengths)
        rng = np.arange(max_len)[None, :]
        mask = (rng < lengths[:, None])
        return mask[:, None, None, :]


def build_look_ahead_mask(size):
    """建立 look-ahead mask: 下三角為1，上三角為0"""
    if TORCH_AVA:
        mask = torch.tril(torch.ones((size, size), dtype=torch.bool))
    else:
        mask = np.tril(np.ones((size, size), dtype=bool))
    return mask


if __name__ == "__main__":
    print("TORCH_AVAILABLE =", TORCH_AVA)

    if TORCH_AVA:
        torch.manual_seed(7)
        B, H, L, D = 2, 2, 5, 4

        q = torch.randn(B, H, L, D)
        k = torch.randn(B, H, L, D)
        v = torch.randn(B, H, L, D)

        out, attn = scaled_dot_product_attention(q, k, v, mask=None)
        print("[No Mask] output shape:", tuple(out.shape))
        print("[No Mask] attn shape:", tuple(attn.shape))
        print("Row sums (應為1):", attn[0, 0, 0].sum().item())

        lengths = torch.tensor([3, 5])
        pad_mask = build_padding_mask(lengths, max_len=L)
        _, attn_padded = scaled_dot_product_attention(q, k, v, mask=pad_mask)
        print("[Padding Mask] attn[0,0,0]:", attn_padded[0, 0, 0])
        print("Check pad columns (index>=3) 近似 0?:", attn_padded[0, 0, 0, 3:].sum().item())

        look_mask = build_look_ahead_mask(L)
        la_mask = look_mask.expand(B, H, L, L)
        _, attn_la = scaled_dot_product_attention(q, k, v, mask=la_mask)
        print("[Look-ahead Mask] 上三角是否為0？(印第一列):", attn_la[0, 0, 0])

        scores = torch.tensor([
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        weights = torch.softmax(scores, dim=-1)
        print("3x3 softmax:\n", weights)

    else:
        np.random.seed(7)
        B, H, L, D = 2, 2, 5, 4
        q = np.random.rand(B, H, L, D)
        k = np.random.rand(B, H, L, D)
        v = np.random.rand(B, H, L, D)

        out, attn = scaled_dot_product_attention(q, k, v, mask=None)
        print("[No Mask][NP] output shape:", out.shape)
        print("[No Mask][NP] attn shape:", attn.shape)
        print("Row sums (應為1):", attn[0, 0, 0].sum())

        lengths = np.array([3, 5])
        pad_mask = build_padding_mask(lengths, max_len=L)
        _, attn_padded = scaled_dot_product_attention(q, k, v, mask=pad_mask)
        print("[Padding Mask][NP] attn[0,0,0]:", attn_padded[0, 0, 0])
        print("Check pad columns (index>=3) 近似 0?:", attn_padded[0, 0, 0, 3:].sum())

        look_mask = build_look_ahead_mask(L)
        la_mask = np.broadcast_to(look_mask, (B, H, L, L))
        _, attn_la = scaled_dot_product_attention(q, k, v, mask=la_mask)
        print("[Look-ahead Mask][NP] 上三角是否為0?(印第一列):", attn_la[0, 0, 0])

        scores = np.array([
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores)
        weights = exp / exp.sum(axis=-1, keepdims=True)
        print("[NP] 3x3 softmax:\n", weights)
