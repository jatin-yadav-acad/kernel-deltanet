r"""
Kernel-DeltaNet Engine
Module: core.rkhs_features
=============================================
High-Dimensional Reproducing Kernel Hilbert Space (RKHS) Feature Projections.

Mathematical Foundation:
1. Bochner's Theorem:
   Every continuous shift-invariant positive-definite kernel k(x, y) = k(x - y) on R^d
   is the Fourier transform of a non-negative Borel measure p(omega):
     k(x, y) = \int_{R^d} p(omega) e^{i omega^T (x - y)} d omega
   For the Gaussian Radial Basis Function (RBF) kernel:
     k(x, y) = exp(-gamma ||x - y||^2)
   the corresponding spectral density is a Gaussian distribution:
     p(omega) ~ N(0, 2 * gamma * I)

2. Random Fourier Features (RFF) (Rahimi & Recht, 2007):
   Approximates the inner product via Monte Carlo sampling with M = D / 2 frequencies:
     phi(x) = 1/sqrt(M) [ cos(Omega x), sin(Omega x) ]^T in R^D
   where Omega_m ~ N(0, 2 * gamma * I).

3. Orthogonal Random Features (ORF) (Yu et al., 2016):
   Imposes exact orthogonality on frequencies via QR decomposition:
     Omega = S * Q
   where Q in O(d) is a Haar-random orthogonal matrix and S is chi-distributed,
   drastically reducing approximation variance.

4. Multi-Scale Spectral Bandwidth Routing:
   Each head h maintains an independent learnable bandwidth gamma_h = exp(log_gamma_h):
     - High gamma (gamma >= 10.0): Ultra-narrow Dirac-like kernel; exact entity matching.
     - Mid gamma (gamma ~ 1.0 - 5.0): Clause-level syntactic binding.
     - Low gamma (gamma <= 0.5): Broad harmonic thematic context.
"""

import math
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiHeadRKHSKernel(nn.Module):
    """
    Random Fourier Feature (RFF) projection with learnable spectral bandwidth per head.
    Approximates Gaussian RBF kernel: k(x, y) = exp(-gamma * ||x - y||^2)
    Capacity scaling: K_max proportional to D * exp(2 * gamma).
    """
    def __init__(
        self,
        num_heads: int,
        d_head: int,
        rkhs_dim: int,
        gamma_init: float = 2.0,
        orthogonal: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.rkhs_dim = rkhs_dim
        assert rkhs_dim % 2 == 0, "rkhs_dim must be even for sin/cos projection"
        self.m = rkhs_dim // 2
        self.orthogonal = orthogonal

        # Initialize learnable spectral bandwidth spectrum per head
        # Distributed logarithmically across heads to cover local to global scales
        if num_heads == 1:
            gamma_values = torch.tensor([float(gamma_init)])
        else:
            gamma_values = torch.logspace(
                math.log10(gamma_init * 0.2), math.log10(max(gamma_init * 4.0, 10.0)), steps=num_heads
            )
        self.log_gamma = nn.Parameter(torch.log(gamma_values))

        # Sample projection matrix Omega: (num_heads, m, d_head)
        omega = self._initialize_omega()
        self.register_buffer("omega", omega)

    def _initialize_omega(self) -> torch.Tensor:
        """
        Samples Gaussian or Orthogonal Random Feature matrices.
        """
        if not self.orthogonal:
            # Standard Gaussian RFF
            return torch.randn(self.num_heads, self.m, self.d_head)
        
        # Orthogonal Random Features (ORF)
        omegas = []
        for _ in range(self.num_heads):
            # Stack orthogonal blocks of size d_head x d_head
            num_blocks = int(math.ceil(self.m / self.d_head))
            blocks = []
            for _ in range(num_blocks):
                g = torch.randn(self.d_head, self.d_head)
                q, _ = torch.linalg.qr(g)
                # Chi-distributed radial scaling: s ~ sqrt(chi^2(d_head))
                s = torch.sqrt(torch.sum(torch.randn(self.d_head, self.d_head) ** 2, dim=-1, keepdim=True))
                blocks.append(q * s)
            full_omega = torch.cat(blocks, dim=0)[:self.m, :] # (m, d_head)
            omegas.append(full_omega)
        return torch.stack(omegas, dim=0) # (num_heads, m, d_head)

    def get_bandwidths(self) -> torch.Tensor:
        """Returns current spectral bandwidths gamma_h for each head."""
        return torch.exp(self.log_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Lifts inputs x into non-linear RKHS feature space.
        Args:
            x: Input tensor of shape (B, num_heads, T, d_head)
        Returns:
            phi: RKHS feature map of shape (B, num_heads, T, rkhs_dim)
        """
        B, H, T, D = x.shape
        assert H == self.num_heads and D == self.d_head, f"Shape mismatch: {x.shape} vs ({self.num_heads}, {self.d_head})"
        
        dtype = x.dtype
        # gamma: (1, H, 1, 1)
        gamma = torch.exp(self.log_gamma).view(1, H, 1, 1).to(dtype=dtype)

        # Scale projection by sqrt(2 * gamma)
        # Omega: (H, m, D) -> einsum over feature dimension D
        projected = torch.einsum("bhtd,hmd->bhtm", x, self.omega.to(dtype=dtype))
        scaled_proj = projected * torch.sqrt(2.0 * gamma)

        # Construct normalized sine/cosine feature representation
        cos_feat = torch.cos(scaled_proj)
        sin_feat = torch.sin(scaled_proj)
        phi = torch.cat([cos_feat, sin_feat], dim=-1) * (1.0 / math.sqrt(self.m))
        return phi.to(dtype=dtype)

    def compute_kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes approximated Gram matrix K_hat(x, y) = phi(x) @ phi(y)^T.
        Args:
            x: (B, H, T_x, d_head)
            y: (B, H, T_y, d_head)
        Returns:
            gram: (B, H, T_x, T_y)
        """
        phi_x = self.forward(x)
        phi_y = self.forward(y)
        return torch.matmul(phi_x, phi_y.transpose(-1, -2))

    def analytical_rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes the exact ground-truth Gaussian RBF kernel matrix:
          k(x, y) = exp(-gamma * ||x - y||^2)
        Args:
            x: (B, H, T_x, d_head)
            y: (B, H, T_y, d_head)
        Returns:
            exact_k: (B, H, T_x, T_y)
        """
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True) # (B, H, T_x, 1)
        y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True).transpose(-1, -2) # (B, H, 1, T_y)
        dot_xy = torch.matmul(x, y.transpose(-1, -2)) # (B, H, T_x, T_y)
        sq_dist = F.relu(x_norm_sq + y_norm_sq - 2.0 * dot_xy)
        
        gamma = self.get_bandwidths().view(1, self.num_heads, 1, 1)
        return torch.exp(-gamma * sq_dist)


class SwiGLUFeatureMap(nn.Module):
    """
    Learned non-linear SwiGLU projection feature map.
    Maps d_head -> rkhs_dim using gated activations.
    """
    def __init__(self, num_heads: int, d_head: int, rkhs_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.rkhs_dim = rkhs_dim

        # Per-head projection matrices
        self.w_gate = nn.Parameter(torch.randn(num_heads, d_head, rkhs_dim) * (1.0 / math.sqrt(d_head)))
        self.w_up = nn.Parameter(torch.randn(num_heads, d_head, rkhs_dim) * (1.0 / math.sqrt(d_head)))
        self.scale = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, T, d_head)
        Returns:
            phi: (B, H, T, rkhs_dim)
        """
        # x: (B, H, T, D)
        gate = torch.einsum("bhtd,hde->bhte", x, self.w_gate)
        up = torch.einsum("bhtd,hde->bhte", x, self.w_up)
        phi = F.silu(gate) * up
        # Unit normalization preserves spectral stability
        return F.normalize(phi, p=2, dim=-1) * self.scale


class PolynomialFeatureMap(nn.Module):
    """
    Homogeneous/Inhomogeneous polynomial feature map of degree p.
    Approximates (x^T y + c)^p.
    """
    def __init__(self, num_heads: int, d_head: int, rkhs_dim: int, degree: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.rkhs_dim = rkhs_dim
        self.degree = degree
        
        self.proj = nn.Parameter(torch.randn(num_heads, d_head, rkhs_dim) * (1.0 / math.sqrt(d_head)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = torch.einsum("bhtd,hde->bhte", x, self.proj)
        if self.degree == 2:
            quad = (linear ** 2) / math.sqrt(2.0)
            phi = torch.cat([linear, quad], dim=-1)[:, :, :, :self.rkhs_dim]
        else:
            phi = (linear ** self.degree) / math.sqrt(math.factorial(self.degree))
        return F.normalize(phi, p=2, dim=-1)
