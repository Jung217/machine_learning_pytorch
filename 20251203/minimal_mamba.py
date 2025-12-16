import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ===============================
# RMSNorm：Root Mean Square Normalization
# 用來取代 LayerNorm（Mamba / LLaMA 常用）
# ===============================
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps                               # 避免除以 0
        self.weight = nn.Parameter(torch.ones(d_model))  # 可學習的 scale 參數

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        
        # 計算 RMS：sqrt(mean(x^2))
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # 正規化後再乘上可學習權重
        return x * rms * self.weight


# ===============================
# Mamba Block（核心模組）
# ===============================
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model                      # 輸入 / 輸出維度
        self.d_state = d_state                      # SSM 隱狀態維度
        self.d_conv = d_conv                        # depthwise conv kernel size
        self.expand = expand                        # 通道擴張倍率
        self.d_inner = expand * d_model             # 內部通道數

        # 將 input 投影成兩份（x, z），用於 gated 機制
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise Conv1D：建模短距離關係
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,                    # depthwise convolution
            bias=True,
            padding=d_conv - 1                      # 保持長度
        )

        # 將 x 投影成 Δ, B, C（SSM 參數）
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 3, bias=False)

        # 將 Δ 映射到每個 channel（控制時間步長）
        self.dt_proj = nn.Linear(self.d_state, self.d_inner, bias=True)

        # SSM 的 A 矩陣（負指數，確保穩定）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))     # 用 log 參數化

        # Skip connection 權重
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 正規化與輸出投影
        self.norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        # 投影後切成 x（主路徑）與 z（gate）
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # Conv1D 需要 channel-first
        x = x.transpose(1, 2)                       # (B, C, T)
        x = self.conv1d(x)[:, :, :seq_len]          # 去掉 padding 多出來的部分
        x = x.transpose(1, 2)                       # 回到 (B, T, C)
        x = F.silu(x)                               # 非線性

        # State Space Model
        y = self.ssm(x)

        # Gate（類似 GLU）
        y = y * F.silu(z)

        # 正規化與輸出投影
        y = self.norm(y)
        y = self.out_proj(y)

        return y

    def ssm(self, x):
        # x shape: (batch, seq_len, d_inner)

        # A 必須為負，確保狀態穩定
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        # 產生 Δ, B, C
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split(
            [self.d_state, self.d_state, self.d_state], dim=-1
        )

        # Δ 必須為正（softplus）
        delta = F.softplus(self.dt_proj(delta))

        # 選擇性掃描（時間遞推）
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        # u shape: (batch, seq_len, d_inner)
        batch_size, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # e^{ΔA}
        delta_A = torch.exp(delta.unsqueeze(-1) * A)

        # Δ * B * u
        delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        # 初始化隱狀態
        h = torch.zeros(batch_size, d_inner, d_state, device=u.device)
        ys = []

        # 時間步遞推（SSM 核心）
        for i in range(seq_len):
            h = delta_A[:, i] * h + delta_B_u[:, i]   # 狀態更新
            y = (h @ C[:, i, :].unsqueeze(-1)).squeeze(-1)
            ys.append(y)

        # 將時間步堆疊回序列
        y = torch.stack(ys, dim=1)

        # Skip connection
        return y + u * D


# ===============================
# 測試 MambaBlock
# ===============================
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

    # 輸入輸出 shape 必須一致（才能堆疊）
    assert output.shape == x.shape
    print("Output shape is correct.")