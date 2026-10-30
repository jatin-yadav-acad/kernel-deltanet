"""
Kernel-DeltaNet Engine
Module: models.moe_kernel_deltanet
==================================
Sparse Mixture-of-Kernel-DeltaNet Experts (MoE-Kernel-DeltaNet).
Scales model capacity while keeping per-token inference FLOPs and recurrent state constant.
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.kernel_deltanet_layer import KernelDeltaNetLayer
from layers.hybrid_kernel_deltanet_block import RMSNorm, SwiGLUFFN


class MoEKernelDeltaNetBlock(nn.Module):
    """
    Mixture-of-Experts Block where the sequence layer is shared or dispatched across experts,
    and FFN layers are routed sparsely.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int = 4,
        top_k: int = 2,
        rkhs_factor: int = 4,
        chunk_size: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        self.norm1 = RMSNorm(d_model)
        self.attn = KernelDeltaNetLayer(
            d_model=d_model,
            num_heads=num_heads,
            rkhs_factor=rkhs_factor,
            chunk_size=chunk_size
        )

        self.norm2 = RMSNorm(d_model)
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLUFFN(d_model) for _ in range(num_experts)
        ])

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape

        # 1. Kernel-DeltaNet Sequence Sub-layer
        norm_x = self.norm1(x)
        attn_out, new_state = self.attn(norm_x, state=state)
        h = x + attn_out

        # 2. Sparse MoE FFN Routing
        norm_h = self.norm2(h)
        logits = self.router(norm_h)
        weights, selected_experts = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Compute sparse expert dispatch
        final_ffn = torch.zeros_like(h)
        for k in range(self.top_k):
            exp_indices = selected_experts[..., k]
            exp_weights = weights[..., k:k+1]
            for exp_id in range(self.num_experts):
                mask = (exp_indices == exp_id)
                if mask.any():
                    token_subset = norm_h[mask]
                    exp_out = self.experts[exp_id](token_subset)
                    final_ffn[mask] += exp_out * exp_weights[mask]

        out = h + final_ffn
        return out, new_state


# Backwards compatibility alias
MoEKernelDeltaNetBlock = MoEKernelDeltaNetBlock


# Backwards compatibility alias
MoEKernelTTTBlock = MoEKernelDeltaNetBlock
