"""
Core implementation of Pure Kernel-DeltaNet v2 operators:
1. Spectral Mean Deflation (Theorem 6.4)
2. Scaled RKHS Projection (D=512 and D=1024)
3. Persistent Register Heads (lambda = 1.0)
4. Second-Order Momentum Delta Rule
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

class PureKernelDeltaNetV2Core(nn.Module):
    def __init__(
        self,
        d_k: int = 64,
        d_v: int = 64,
        num_heads: int = 8,
        D: int = 512,
        use_mean_deflation: bool = True,
        num_persistent_heads: int = 2,
        momentum_mu: float = 0.0,
        decay_init: float = 0.99,
        beta_init: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.num_heads = num_heads
        self.D = D
        self.use_mean_deflation = use_mean_deflation
        self.num_persistent_heads = num_persistent_heads
        self.momentum_mu = momentum_mu
        self.device = device
        self.dtype = dtype

        # Structured orthogonal random features: M = D // 2
        self.M = D // 2
        # Generate orthogonal blocks
        num_blocks = int(math.ceil(self.M / d_k))
        blocks = []
        for _ in range(num_blocks):
            g = torch.randn(d_k, d_k, device=device, dtype=dtype)
            q, _ = torch.linalg.qr(g)
            s = torch.sqrt(torch.sum(torch.randn(d_k, d_k, device=device, dtype=dtype) ** 2, dim=-1, keepdim=True))
            blocks.append(q * s)
        omega = torch.cat(blocks, dim=0)[:self.M, :] # (M, d_k)
        self.register_buffer("omega", omega)

        # Precompute theoretical DC centroid for spherical RFF
        # E_k[cos(omega^T k)] for k ~ Unif(S^{d_k-1})
        # For unit vectors k, omega^T k has mean 0 and variance ||omega||^2 / d_k
        # By symmetry, sin component has expectation 0. Cosine component has positive DC bias:
        # Approximate E[cos(w^T k)] ~ exp(- ||w||^2 / (2 * d_k))
        omega_norms_sq = torch.sum(self.omega ** 2, dim=-1) # (M,)
        cos_mean = torch.exp(- omega_norms_sq / (2.0 * d_k)) / math.sqrt(self.M) # (M,)
        sin_mean = torch.zeros(self.M, device=device, dtype=dtype)
        dc_centroid = torch.cat([cos_mean, sin_mean], dim=-1) # (D,)
        self.register_buffer("dc_centroid", dc_centroid)

        # Head-specific decay parameters lambda_h and beta_h
        lambdas = torch.full((num_heads,), decay_init, device=device, dtype=dtype)
        betas = torch.full((num_heads,), beta_init, device=device, dtype=dtype)
        if num_persistent_heads > 0:
            # Persistent Register Heads: strictly lambda = 1.0, lower learning rate beta = 0.08
            lambdas[:num_persistent_heads] = 1.00000
            betas[:num_persistent_heads] = 0.08000
        
        self.register_buffer("lambdas", lambdas.view(1, num_heads, 1, 1))
        self.register_buffer("betas", betas.view(1, num_heads, 1, 1))

    def project_phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lifts ambient keys/queries x: (B, H, T, d_k) into RKHS features (B, H, T, D)
        Optionally applies Spectral Mean Deflation (Theorem 6.4).
        """
        # x is assumed unit-normalized: ||x|| = 1
        proj = torch.matmul(x, self.omega.T) # (B, H, T, M)
        phi_raw = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(self.M) # (B, H, T, D)

        if not self.use_mean_deflation:
            return phi_raw
        
        # Apply Spectral Mean Deflation: subtract stationary DC centroid and re-normalize
        phi_centered = phi_raw - self.dc_centroid.view(1, 1, 1, self.D)
        phi_deflated = F.normalize(phi_centered, p=2, dim=-1)
        return phi_deflated

    def forward_recurrent(
        self,
        k: torch.Tensor, # (B, H, T, d_k)
        v: torch.Tensor, # (B, H, T, d_v)
        q: torch.Tensor, # (B, H, T, d_k)
        initial_state: Optional[torch.Tensor] = None, # (B, H, d_v, D)
        return_last_only: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Step-by-step causal recurrence with support for:
        - Multi-Scale Persistent Register Heads
        - Second-Order Momentum Delta Rule
        - Chunk-lifted feature projection to guarantee O(1) activation memory
        """
        B, H, T, _ = k.shape
        d_v = v.shape[-1]
        D = self.D

        S = torch.zeros(B, H, d_v, D, device=self.device, dtype=self.dtype) if initial_state is None else initial_state.clone()
        M_state = torch.zeros(B, H, d_v, D, device=self.device, dtype=self.dtype) if self.momentum_mu > 0.0 else None

        outputs = [] if not return_last_only else None
        last_y = None

        # Pre-lift query if return_last_only
        q_last = self.project_phi(q[:, :, -1:, :]) if return_last_only else None

        # Chunk lifting: project features in chunks of 256 to avoid allocating O(T*D) memory
        chunk_lift = 256

        for c_start in range(0, T, chunk_lift):
            c_end = min(c_start + chunk_lift, T)
            k_chunk = self.project_phi(k[:, :, c_start:c_end, :]) # (B, H, chunk_len, D)
            q_chunk = self.project_phi(q[:, :, c_start:c_end, :]) if not return_last_only else None

            for step_idx in range(c_end - c_start):
                t = c_start + step_idx
                kt = k_chunk[:, :, step_idx:step_idx+1, :] # (B, H, 1, D)
                vt = v[:, :, t:t+1, :]                     # (B, H, 1, d_v)

                # Causal query readout: y_t = S_{t-1} @ q_t
                if not return_last_only:
                    qt = q_chunk[:, :, step_idx:step_idx+1, :]
                    yt = torch.matmul(S, qt.transpose(-1, -2)).squeeze(-1) # (B, H, d_v)
                    outputs.append(yt)
                elif t == T - 1:
                    last_y = torch.matmul(S, q_last.transpose(-1, -2)).squeeze(-1) # (B, H, d_v)

                # Associative error: error = v_t - S_{t-1} @ k_t
                pred_v = torch.matmul(S, kt.transpose(-1, -2)).squeeze(-1) # (B, H, d_v)
                error = vt.squeeze(2) - pred_v # (B, H, d_v)
                grad_update = torch.matmul(error.unsqueeze(-1), kt) # (B, H, d_v, D)

                # State update
                if self.momentum_mu > 0.0:
                    M_state = self.momentum_mu * M_state + self.betas * grad_update
                    S = self.lambdas * S + M_state
                else:
                    S = self.lambdas * S + self.betas * grad_update

        final_norm = torch.norm(S, dim=(-2, -1)).mean().item()
        if return_last_only:
            out = last_y.unsqueeze(2) # (B, H, 1, d_v)
        else:
            out = torch.stack(outputs, dim=2) # (B, H, T, d_v)
        return out, S, {"state_norms": [final_norm]}

print("PureKernelDeltaNetV2Core module defined successfully.")
