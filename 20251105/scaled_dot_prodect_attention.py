import torch, math
import torch.nn.functional as F
import numpy as np

try:
    import torch, math
    import torch.nn.functional as F
    TORCH_AVA = True
except Exception as e:
    TORCH_AVA = False

def scaled_dot_product_attention(q, k, v, mask=None, dropout_p: float=0.0):
    if TORCH_AVA:
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        if mask is not None: scores = scores.masked_fill(mask==0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)

        if dropout_p and dropout_p > 0: attn_weights = F.dropout(attn_weights, p=dropout_p)
        output = attn_weights @ v

        return output, attn_weights
    else:
        d_k = q.shape[-1]
        scores = torch.matumul(q, np.swapaxes(k, -2, -1)) / math.sqrt(d_k)
        if mask is not None: scores = np.where(mask==0, -1e9, scores)
        score_max = scores.max(axis=-1, keepdims=True)

        exp = np.exp(scores - score_max)
        attn_weights = exp / exp.sum(axis=-1, keepdims=True)
        output = np.matmul(attn_weights, v)

        return output, attn_weights
    
def build_padding_mask(lengths, max_len):
    try:
        device = lengths.device if hasattr(lengths, "device") else "cpu"
        rng = torch.range(max_len, device=device).unsqueeze(0)
        mask = (rng < lengths.unsqueeze(1)).to(torch.bool)
        return mask.unsqueeze(1).unsqueeze(1)
    except Exception:
