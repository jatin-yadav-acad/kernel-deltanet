r"""
Kernel-DeltaNet Engine
Module: core.advanced_kernels
================================================
Advanced Reproducing Kernel Hilbert Space (RKHS) Mappings and Spectral Expansions:
1. Cauchy / Heavy-Tailed Kernel (long-tail memory retention)
2. Matérn 3/2 & 5/2 Spectral Kernels (fractional Sobolev smoothness)
3. Arc-Cosine Kernel (infinite-width deep ReLU representation)
4. Fastfood Fast Walsh-Hadamard Transform (O(D log D) projection acceleration)
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CauchyHeavyTailedKernel(nn.Module):
    r"""
    Heavy-Tailed Cauchy Kernel:
      k(x, y) = 1 / (1 + gamma * ||x - y||^2)
    Unlike the Gaussian kernel which decays exponentially exp(-gamma r^2),
    the Cauchy kernel decays as a power law (1 + gamma r^2)^{-1}.
    This prevents catastrophic memory forgetting on long sequence contexts.
    
    By Bochner's theorem, the spectral measure of the Cauchy kernel is
    an isotropic multivariate Laplace/exponential distribution:
      p(omega) \propto exp(-||omega|| / sqrt(gamma))
    """
    def __init__(self, num_heads: int, d_head: int, rkhs_dim: int, gamma_init: float = 2.0):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.rkhs_dim = rkhs_dim
        assert rkhs_dim % 2 == 0, "rkhs_dim must be even"
        self.m = rkhs_dim // 2

        if num_heads == 1:
            gamma_values = torch.tensor([float(gamma_init)])
        else:
            gamma_values = torch.logspace(
                math.log10(gamma_init * 0.2), math.log10(max(gamma_init * 4.0, 10.0)), steps=num_heads
            )
        self.log_gamma = nn.Parameter(torch.log(gamma_values))

        # Sample Laplace-distributed projection frequencies
        # omega ~ Laplace(0, scale)
        omega = self._sample_laplace_frequencies()
        self.register_buffer("omega", omega)

    def _sample_laplace_frequencies(self) -> torch.Tensor:
        # Laplace(0, 1) = U1 * Exponential(1) where U1 is uniform direction
        omegas = []
        for _ in range(self.num_heads):
            # Direction vector: standard normal normalized
            u = torch.randn(self.m, self.d_head)
            u = F.normalize(u, p=2, dim=-1)
            # Radial distance: exponential distribution
            r = -torch.log(torch.rand(self.m, 1) + 1e-9)
            omegas.append(u * r)
        return torch.stack(omegas, dim=0) # (num_heads, m, d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            x: (B, H, T, d_head)
        Returns:
            phi: (B, H, T, rkhs_dim)
        """
        B, H, T, D = x.shape
        gamma = torch.exp(self.log_gamma).view(1, H, 1, 1)
        scale = torch.sqrt(gamma)

        # Projected frequencies: (B, H, T, m)
        proj = torch.einsum("bhtd,hmd->bhtm", x, self.omega) * scale
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * (1.0 / math.sqrt(self.m))
        return phi


class MaternSpectralKernel(nn.Module):
    r"""
    Matérn Class Smoothness Kernels (nu = 3/2 and nu = 5/2).
    Models non-smooth linguistic and reasoning transitions where functions belong
    to Sobolev spaces W_2^s with fractional differentiability.
    
    The spectral density of a Matérn kernel is a Student-t distribution:
      p(omega) \propto (2 * nu / l^2 + ||omega||^2)^{-(nu + d/2)}
    """
    def __init__(
        self,
        num_heads: int,
        d_head: int,
        rkhs_dim: int,
        nu: float = 1.5,
        lengthscale_init: float = 1.0
    ):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.rkhs_dim = rkhs_dim
        self.nu = nu
        assert rkhs_dim % 2 == 0
        self.m = rkhs_dim // 2

        self.log_lengthscale = nn.Parameter(torch.full((num_heads,), math.log(lengthscale_init)))
        omega = self._sample_student_t_frequencies()
        self.register_buffer("omega", omega)

    def _sample_student_t_frequencies(self) -> torch.Tensor:
        # Student-t with 2*nu degrees of freedom
        df = 2.0 * self.nu
        omegas = []
        for _ in range(self.num_heads):
            # Normal / sqrt(Chi2 / df)
            z = torch.randn(self.m, self.d_head)
            # Sample Chi2 with df degrees of freedom via Gamma
            u = torch.randn(self.m, int(math.ceil(df)))
            chi2 = torch.sum(u ** 2, dim=-1, keepdim=True)
            scale = torch.sqrt(chi2 / df)
            omegas.append(z / (scale + 1e-8))
        return torch.stack(omegas, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        l = torch.exp(self.log_lengthscale).view(1, H, 1, 1)
        scaled_x = x / (l + 1e-8)
        proj = torch.einsum("bhtd,hmd->bhtm", scaled_x, self.omega)
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) * (1.0 / math.sqrt(self.m))
        return phi


class ArcCosineKernel(nn.Module):
    r"""
    Arc-Cosine Kernel of Degree 0 and 1 (Cho & Saul, 2009).
    Represents the exact inner product of an infinite-width single-layer neural network
    with step (degree 0) or ReLU (degree 1) activation functions.
    
    k_1(x, y) = (1 / pi) * ||x|| * ||y|| * (sin(theta) + (pi - theta) * cos(theta))
    where theta = arccos(x^T y / (||x|| ||y||)).
    
    Random Feature Approximation:
      phi(x) = sqrt(2 / M) * max(0, Omega * x)
    """
    def __init__(self, num_heads: int, d_head: int, rkhs_dim: int, degree: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_head
        self.rkhs_dim = rkhs_dim
        self.degree = degree
        assert degree in (0, 1), "Only degree 0 (step) and 1 (ReLU) supported"

        # Gaussian weight matrix
        omega = torch.randn(num_heads, rkhs_dim, d_head)
        self.register_buffer("omega", omega)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        proj = torch.einsum("bhtd,hmd->bhtm", x, self.omega) # (B, H, T, rkhs_dim)
        if self.degree == 1:
            # ReLU activation
            phi = F.relu(proj) * math.sqrt(2.0 / self.rkhs_dim)
        else:
            # Heaviside step activation
            phi = (proj > 0).float() * math.sqrt(2.0 / self.rkhs_dim)
        return phi


class FastfoodWalshHadamardKernel(nn.Module):
    r"""
    Fastfood Transform (Le, Sarlós, Smola, 2013).
    Accelerates Random Fourier Feature projections from O(d * D) down to O(D log d).
    Factorizes the random Gaussian matrix Omega as:
      Omega = (1 / (sigma * sqrt(d))) * S * H * G * Pi * H * B
    where:
      B: random diagonal Rademacher matrix (\pm 1)
      H: Walsh-Hadamard Matrix
      Pi: random permutation matrix
      G: random Gaussian diagonal matrix
      S: random scaling diagonal matrix (chi-distributed)
    """
    def __init__(self, num_heads: int, d_head: int, rkhs_dim: int, gamma: float = 2.0):
        super().__init__()
        self.num_heads = num_heads
        # d_head must be a power of 2 for Hadamard transform
        self.d_orig = d_head
        self.d_pow2 = 1 << (d_head - 1).bit_length()
        self.rkhs_dim = rkhs_dim
        assert rkhs_dim % 2 == 0
        self.m = rkhs_dim // 2

        self.num_blocks = int(math.ceil(self.m / self.d_pow2))
        self.gamma = gamma

        # Rademacher diagonal B: \pm 1
        B = torch.randint(0, 2, (num_heads, self.num_blocks, self.d_pow2)).float() * 2.0 - 1.0
        # Permutation Pi
        perm = torch.stack([
            torch.stack([torch.randperm(self.d_pow2) for _ in range(self.num_blocks)])
            for _ in range(num_heads)
        ])
        # Gaussian diagonal G
        G = torch.randn(num_heads, self.num_blocks, self.d_pow2)
        # Chi-scaling diagonal S
        S = torch.randn(num_heads, self.num_blocks, self.d_pow2) ** 2

        self.register_buffer("B", B)
        self.register_buffer("perm", perm)
        self.register_buffer("G", G)
        self.register_buffer("S", S)

    @staticmethod
    def _hadamard_transform(x: torch.Tensor) -> torch.Tensor:
        r"""
        Fast Walsh-Hadamard Transform (FWHT) in O(N log N).
        Args:
            x: (..., N) where N is a power of 2.
        """
        orig_shape = x.shape
        N = orig_shape[-1]
        x_flat = x.reshape(-1, N)
        
        h = 1
        while h < N:
            x_flat = x_flat.view(-1, N // (2 * h), 2, h)
            u = x_flat[:, :, 0, :].clone()
            v = x_flat[:, :, 1, :].clone()
            x_flat[:, :, 0, :] = u + v
            x_flat[:, :, 1, :] = u - v
            x_flat = x_flat.view(-1, N)
            h *= 2
            
        return (x_flat * (1.0 / math.sqrt(N))).view(orig_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Fast projection in O(D log d).
        """
        B_batch, H, T, D = x.shape
        # Pad to power of 2 if needed
        if D < self.d_pow2:
            x_pad = F.pad(x, (0, self.d_pow2 - D))
        else:
            x_pad = x

        projs = []
        for blk in range(self.num_blocks):
            # 1. B @ x
            b = self.B[:, blk, :].view(1, H, 1, self.d_pow2)
            xb = x_pad * b # (B, H, T, d_pow2)
            # 2. H @ (B @ x)
            hxb = self._hadamard_transform(xb)
            # 3. Pi @ H @ B @ x
            perm_idx = self.perm[:, blk, :] # (H, d_pow2)
            # Gather along last dim
            pi_hxb = torch.stack([
                torch.index_select(hxb[:, h, :, :], dim=-1, index=perm_idx[h])
                for h in range(H)
            ], dim=1) # (B, H, T, d_pow2)
            # 4. G @ Pi @ H @ B @ x
            g = self.G[:, blk, :].view(1, H, 1, self.d_pow2)
            g_pi = pi_hxb * g
            # 5. H @ G @ Pi @ H @ B @ x
            hg_pi = self._hadamard_transform(g_pi)
            # 6. S scaling
            s = self.S[:, blk, :].view(1, H, 1, self.d_pow2)
            out_blk = hg_pi * s * math.sqrt(2.0 * self.gamma)
            projs.append(out_blk)

        full_proj = torch.cat(projs, dim=-1)[..., :self.m] # (B, H, T, m)
        phi = torch.cat([torch.cos(full_proj), torch.sin(full_proj)], dim=-1) * (1.0 / math.sqrt(self.m))
        return phi
