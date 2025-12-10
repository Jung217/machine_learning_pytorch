import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model:int , eps: float=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * self.weight
        return output
        
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # self.x_proj = nn.Linear(self.d_inner, self.d_state + self.d_model * 2, bias=False)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 3, bias=False)

        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(z)
        y = self.norm(y)
        y = self.out_proj(y)

        return y

    def ssm(self, x):
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x) 
        
        # 這裡 split 需要 x_dbl 的最後一維是 3 * d_state (即 48)
        delta, B, C = x_dbl.split([self.d_state, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))

        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        delta_A = torch.exp(delta.unsqueeze(-1) * A) 
        delta_B_u = (delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)) 

        h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
        ys = []

        for i in range(seq_len):
            h = delta_A[:, i] * h + delta_B_u[:, i]
            y = (h @ C[:, i, :].unsqueeze(-1)).squeeze(-1)
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + u * D

        return y

if __name__ == '__main__':
    d_model = 64
    batch_size = 2
    seq_len = 128

    model = MambaBlock(d_model=d_model)
    print("MambaBlock created successfully.")

    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")

    assert output.shape == x.shape
    print("Output shape is correct.")
