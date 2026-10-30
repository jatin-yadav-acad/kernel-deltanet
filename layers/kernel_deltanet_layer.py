"""
Kernel-DeltaNet Engine
Module: layers.kernel_deltanet_layer
====================================
Complete Multi-Head Kernel-DeltaNet Sequence Layer.
- Training: Parallel O(N) associative chunked scan
- Inference: O(1) state recurrence S_t in R^{d_v x D}
"""

import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.rkhs_features import MultiHeadRKHSKernel
from core.chunked_scan import chunked_affine_scan, recurrent_step


class KernelDeltaNetLayer(nn.Module):
    """
    Complete Multi-Head Kernel-DeltaNet Layer.
    - Training: Parallel O(N) associative chunked scan
    - Inference: O(1) state recurrence S_t in R^{d_v x D}
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rkhs_factor: int = 4,
        chunk_size: int = 64,
        gamma_init: float = 2.0,
        initial_decay: float = 0.99,
        initial_eta: float = 0.5
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rkhs_factor = rkhs_factor
        self.rkhs_dim = self.d_head * rkhs_factor
        self.chunk_size = chunk_size

        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model, bias=False)

        # Non-linear RKHS feature kernel with learnable spectral bandwidths per head
        self.kernel = MultiHeadRKHSKernel(
            num_heads=num_heads,
            d_head=self.d_head,
            rkhs_dim=self.rkhs_dim,
            gamma_init=gamma_init,
            orthogonal=True
        )

        # Learnable decay parameter lambda = sigmoid(logit_decay)
        init_logit = math.log(initial_decay / (1.0 - initial_decay + 1e-7))
        self.logit_decay = nn.Parameter(torch.full((1, num_heads, 1, 1), float(init_logit)))

        # Learnable step size eta = softplus(raw_eta)
        init_raw_eta = math.log(math.exp(initial_eta) - 1.0 + 1e-7)
        self.raw_eta = nn.Parameter(torch.full((1, num_heads, 1, 1), float(init_raw_eta)))

    def get_decay(self) -> torch.Tensor:
        """Returns bounded decay lambda in (0, 1)."""
        return torch.sigmoid(self.logit_decay)

    def get_eta(self) -> torch.Tensor:
        """Returns positive step size eta > 0."""
        return F.softplus(self.raw_eta)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Dual-mode forward pass.
        If state is None and T > 1: Run chunked parallel associative scan (Training)
        If state is provided or T == 1: Run single-step constant memory recurrence (Inference)

        Args:
            x: Input tensor of shape (B, T, d_model)
            state: Optional memory state tensor of shape (B, H, d_head, rkhs_dim)
        Returns:
            out: Output tensor of shape (B, T, d_model)
            new_state: Updated state tensor if in recurrent mode, else None
        """
        B, T, D_in = x.shape

        # 1. Project inputs: (B, T, d_model) -> (B, H, T, d_head)
        q = F.normalize(self.w_q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2), dim=-1)
        k = F.normalize(self.w_k(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2), dim=-1)
        v = self.w_v(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # 2. Lift keys and queries into non-linear RKHS space: (B, H, T, rkhs_dim)
        phi_k = self.kernel(k)
        phi_q = self.kernel(q)

        decay = self.get_decay()
        eta = self.get_eta()

        if state is None and T > 1:
            # Training Mode: Hardware-Native Chunked Parallel Associative Scan
            v_reconstructed = chunked_affine_scan(
                phi_k=phi_k,
                v=v,
                decay=decay,
                eta=eta,
                phi_q=phi_q,
                chunk_size=self.chunk_size
            )
            # Output projection: (B, H, T, d_head) -> (B, T, d_model)
            out = v_reconstructed.transpose(1, 2).contiguous().view(B, T, D_in)
            return self.w_out(out), None

        else:
            # Inference Mode: O(1) Memory Recurrent Step (T >= 1)
            current_state = state
            outputs = []

            for t in range(T):
                phi_k_t = phi_k[:, :, t:t+1, :]
                phi_q_t = phi_q[:, :, t:t+1, :]
                v_t = v[:, :, t:t+1, :]

                y_t, current_state = recurrent_step(
                    phi_t=phi_k_t,
                    v_t=v_t,
                    q_t=phi_q_t,
                    state=current_state,
                    decay=decay,
                    eta=eta
                )
                outputs.append(y_t)

            out_stack = torch.cat(outputs, dim=2)
            out_flat = out_stack.transpose(1, 2).contiguous().view(B, T, D_in)
            return self.w_out(out_flat), current_state


# Backwards compatibility alias
KernelTTTLayer = KernelDeltaNetLayer
