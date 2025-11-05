import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matumul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None: scores = scores.masked_fill(mask==0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    return output, attn_weights

BATCH_SIZE = 4
NUM_HEADS = 8
SEQ_LEN = 10
D_K = 64
D_v = 64

q = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
k = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_K)
v = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_v)

output, weights = scaled_dot_product_attention(q, k ,v)

print("Q shape:", q.shape)
print("K shape:", k.shape)
print("V shape:", v.shape)

print("Output shape:", output.shape)
print("Weights shape:", weights.shape)