import torch, math
import torch.nn.functional as F

def sdp_attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        w = F.softmax(scores, dim=-1)
    return w @ v, w

L, D = 6, 2
q = torch.zeros(1, 1, L, D)
k = torch.zeros(1, 1, L, D)
v = torch.arange(L).float().view(1, 1, L, 1).repeat(1, 1, 1, D)

q[0, 0, -1] = torch.tensor([1.0, 0.0])
k[0, 0, 0] = torch.tensor([1.0, 0.0])
 
for i in range(L - 1):
    q[0, 0, i] = torch.tensor([0.0, 1.0])
    k[0, 0, i + 1] = torch.tensor([0.0, 1.0])

out, attn = sdp_attention(q, k, v)
print(f"最後一列注意力權重 (應該在位置 0 最高)：{attn[0, 0, -1]}")