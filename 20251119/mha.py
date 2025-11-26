from __future__ import annotations
import math
from typing import Optional, Tuple  # [修正 1] 補上 Tuple
import torch
import torch.nn as nn              # [修正 1] 補上 nn
import torch.nn.functional as F

# [修正 3] 假設檔名修正為 attention (若你檔名沒改，這裡請維持原樣)
try:
    from .attetion import (
        scaled_dot_product_attention,
        make_causal_mask,
        combine_bool_masks,
    )
except ImportError:
    # 方便單獨執行此檔案時也能 import
    # 假設 attention.py 在同一目錄下
    from attetion import (
        scaled_dot_product_attention,
        make_causal_mask,
        combine_bool_masks,
    )

class MultiHeadAttention(nn.Module):
    """
    多頭注意力（Encoder/Decoder 皆可用）
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必須能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = attn_dropout
        self.resid_drop = nn.Dropout(resid_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        # (B, L, D) -> (B, L, H, Dk) -> (B, H, L, Dk)
        x = x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, L, Dk = x.shape
        # (B, H, L, Dk) -> (B, L, H, Dk) -> (B, L, D)
        x = x.transpose(1, 2).contiguous().view(B, L, H * Dk)
        return x

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        need_weights: bool = True,
        average_attn_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, L, D = q.shape
        # 檢查維度是否正確 (最後一維必須等於 d_model)
        assert D == self.d_model
        
        # 1. 線性投影
        q_proj = self.W_q(q)
        k_proj = self.W_k(k)
        v_proj = self.W_v(v)

        # 2. 拆成多頭 (B, H, L, Dk)
        qh = self._split_heads(q_proj)
        kh = self._split_heads(k_proj)
        vh = self._split_heads(v_proj)

        # 3. 構建遮罩
        mask_bool = None
        
        # A. Padding Mask
        if key_padding_mask is not None:
            # (B, L) -> (B, 1, 1, L)
            mask_k = key_padding_mask.unsqueeze(1).unsqueeze(1)
            mask_bool = mask_k if mask_bool is None else (mask_bool | mask_k)

        # B. Causal Mask
        if causal:
            mask_c = make_causal_mask(L, device=q.device)
            mask_bool = mask_c if mask_bool is None else (mask_bool | mask_c)

        # C. 合併外部傳入的 attn_mask
        final_mask = mask_bool
        if attn_mask is not None:
            # 如果傳入的也是 bool，就用 combine 合併
            if attn_mask.dtype == torch.bool:
                # [修正 4] 傳入 list
                final_mask = combine_bool_masks([mask_bool, attn_mask])
            else:
                # 如果是加性遮罩，目前簡單處理：忽略內部 mask，直接用外部的 (或是你要寫複雜的加法邏輯)
                # 這裡示範直接覆蓋，或是拋出警告
                final_mask = attn_mask 

        # 4. 執行 Scaled Dot-Product Attention
        out_h, attn = scaled_dot_product_attention(
            qh, kh, vh,
            mask=final_mask,
            dropout_p=self.attn_dropout,
            training=self.training,
            return_weights=True
        )

        # 5. 合併頭 + 最後投影
        out = self.W_o(self._merge_heads(out_h))
        out = self.resid_drop(out)

        if not need_weights:
            return out, None

        # [修正 2] 變數邏輯修正
        if average_attn_weights:
            attn = attn.mean(dim=1) # (B, L, L) 取平均
        
        return out, attn

if __name__ == "__main__":
    # 簡單測試
    B, L, D, H = 4, 10, 512, 8
    x = torch.randn(B, L, D)
    mha = MultiHeadAttention(d_model=D, num_heads=H)
    
    # 這裡必須要有 attention.py 且裡面有正確函式才能跑
    try:
        y, w = mha(x, x, x, causal=False)
        print("y:", y.shape)
        print("w:", None if w is None else w.shape)
    except Exception as e:
        print(f"執行錯誤 (可能是因為缺少 attention.py): {e}")