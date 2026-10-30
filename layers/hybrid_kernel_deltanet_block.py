"""
Kernel-DeltaNet Engine
Module: layers.hybrid_kernel_deltanet_block
===========================================
Pre-LayerNorm Transformer Block with Kernel-DeltaNet Attention and SwiGLU FFN.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel_deltanet_layer import KernelDeltaNetLayer, KernelDeltaNetLayer


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020 / LLaMA, 2023).
    FFN_SwiGLU(x) = (SiLU(x W_gate) * (x W_up)) W_down.
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        if d_ff is None:
            # 8/3 expansion factor matches standard LLaMA / Mistral architecture
            d_ff = int(d_model * 8 / 3)
            # Round to multiple of 64 for optimal tensor core alignment
            d_ff = ((d_ff + 63) // 64) * 64

        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class HybridKernelDeltaNetBlock(nn.Module):
    """
    Full Pre-Norm Decoder Block:
      h = x + KernelDeltaNetLayer(RMSNorm(x))
      out = h + SwiGLUFFN(RMSNorm(h))
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rkhs_factor: int = 4,
        chunk_size: int = 64,
        d_ff: Optional[int] = None,
        drop_rate: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.attn = KernelDeltaNetLayer(
            d_model=d_model,
            num_heads=num_heads,
            rkhs_factor=rkhs_factor,
            chunk_size=chunk_size
        )
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        If state is None: executes chunked parallel scan across sequence.
        If state is provided: executes recurrent step and updates state.
        """
        # 1. Attention sub-layer with residual connection
        normed_x = self.norm1(x)
        attn_out, new_state = self.attn(normed_x, state=state)
        h = x + self.dropout(attn_out)

        # 2. SwiGLU FFN sub-layer with residual connection
        ffn_out = self.ffn(self.norm2(h))
        out = h + self.dropout(ffn_out)

        return out, new_state


# Backwards compatibility alias
HybridKernelDeltaNetBlock = HybridKernelDeltaNetBlock


# Backwards compatibility alias
HybridKernelTTTBlock = HybridKernelDeltaNetBlock
