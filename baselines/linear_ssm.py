"""
Kernel-DeltaNet Engine
Module: baselines.linear_ssm
=============================================
Linear State Space Model (SSM / Mamba-style) Baseline.
h_t = A_bar * h_{t-1} + B_bar * x_t
y_t = C * h_t
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.hybrid_kernel_deltanet_block import RMSNorm, SwiGLUFFN


class LinearSSMLayer(nn.Module):
    """
    Selective Linear State Space Layer.
    """
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, kernel_size=4, groups=d_model, padding=3
        )
        
        # State space parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        
        # Projection and gated conv
        x_proj = self.in_proj(x)
        x_val, gate = x_proj.chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_val.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        A = -torch.exp(self.A_log) # (D, d_state)
        B_mat = self.B_proj(x_conv) # (B, T, d_state)
        C_mat = self.C_proj(x_conv) # (B, T, d_state)

        # Sequential or scan SSM state accumulation
        h = torch.zeros(B, D, self.d_state, device=x.device) if state is None else state
        outputs = []

        dt = torch.sigmoid(x_conv) # Soft discretization
        for t in range(T):
            xt = x_conv[:, t:t+1, :].transpose(1, 2) # (B, D, 1)
            bt = B_mat[:, t:t+1, :]                 # (B, 1, d_state)
            ct = C_mat[:, t:t+1, :]                 # (B, 1, d_state)
            dt_t = dt[:, t:t+1, :].transpose(1, 2)  # (B, D, 1)

            # Discretized A and B
            dA = torch.exp(dt_t * A.unsqueeze(0))   # (B, D, d_state)
            dB = dt_t * bt                         # (B, D, d_state)

            h = h * dA + xt * dB
            yt = torch.sum(h * ct.unsqueeze(1), dim=-1) # (B, D)
            yt = yt + xt.squeeze(-1) * self.D
            outputs.append(yt)

        out_stack = torch.stack(outputs, dim=1) # (B, T, D)
        out_gated = out_stack * F.silu(gate)
        return self.out_proj(out_gated), h
