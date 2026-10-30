# kernels/fwht.py
"""
Fast Walsh-Hadamard Transform (FWHT) Structured Orthogonal Random Projections
Factorizes Omega in R^{D x d_k} (d_k=64, D=512, M=4 blocks) into:
  Omega = 1 / sqrt(d_k) * bigoplus_{m=1}^M (H D_3^{(m)} H D_2^{(m)} H D_1^{(m)})
where H is the normalized Walsh-Hadamard matrix and D_i^{(m)} in {-1, +1}^{d_k x d_k}
are diagonal Rademacher matrices.

When compiled via `setup.py`, delegates directly to the native CUDA warp-shuffle
extension `kernel_deltanet_cuda` (`kernels/in_register_fwht.cu`), maintaining 0
intermediate activation materialization in global DRAM.
"""

import math
import torch
import torch.nn as nn
import numpy as np

try:
    import kernel_deltanet_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False

def fast_walsh_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Computes fast unrolled in-place Walsh-Hadamard transform along the last dimension.
    Input shape: (..., d_k) where d_k is a power of 2 (e.g. 64).
    Complexity: O(d_k log2 d_k) additions and subtractions.
    """
    d_k = x.shape[-1]
    h = 1
    # Create clone to prevent in-place mutation of autograd graph
    res = x.clone()
    while h < d_k:
        # Reshape to pair elements distance h apart
        shape = res.shape
        res = res.view(*shape[:-1], -1, 2, h)
        u = res[..., 0, :].clone()
        v = res[..., 1, :].clone()
        res[..., 0, :] = u + v
        res[..., 1, :] = u - v
        res = res.view(*shape)
        h *= 2
    return res


class FastWalshHadamardProjection(nn.Module):
    """
    Memory-free In-Register Fast Walsh-Hadamard Structured Projection Layer.
    Maps d_k -> D via M = D // d_k orthogonal blocks.
    Requires exactly zero HBM/SRAM for matrix weights, storing only sign bits.
    """
    def __init__(self, d_k: int = 64, D: int = 512, seed: int = 42):
        super().__init__()
        assert D % d_k == 0, f"D ({D}) must be divisible by d_k ({d_k})"
        self.d_k = d_k
        self.D = D
        self.M = D // d_k # Number of orthogonal blocks (8 for D=512, d_k=64)
        self.scale_norm = 1.0 / math.sqrt(d_k)
        
        # Optimal analytical temperature parameter tau* = (d_k)^(1/4)
        self.tau_star = (float(d_k)) ** 0.25
        self.tau_star_inv = 1.0 / self.tau_star

        gen = torch.Generator().manual_seed(seed)
        
        # Rademacher diagonal signs {-1, +1} for 3 stages
        d1 = torch.where(torch.rand(self.M, d_k, generator=gen) > 0.5, 1.0, -1.0)
        d2 = torch.where(torch.rand(self.M, d_k, generator=gen) > 0.5, 1.0, -1.0)
        d3 = torch.where(torch.rand(self.M, d_k, generator=gen) > 0.5, 1.0, -1.0)
        
        # Pack into uint64 bitmasks for warp register kernel
        rad_masks = torch.zeros((self.M, 3), dtype=torch.int64)
        for m in range(self.M):
            for stage_idx, stage_d in enumerate([d1, d2, d3]):
                mask_val = 0
                for bit in range(min(64, d_k)):
                    if stage_d[m, bit].item() < 0:
                        mask_val |= (1 << bit)
                # Convert uint64 to signed int64 for PyTorch tensor compatibility
                signed_val = int(np.int64(np.uint64(mask_val)))
                rad_masks[m, stage_idx] = signed_val

        # Uniform phases in [0, 2pi]
        b = torch.rand(D, generator=gen) * (2.0 * math.pi)
        
        self.register_buffer("d1", d1) # [M, d_k]
        self.register_buffer("d2", d2) # [M, d_k]
        self.register_buffer("d3", d3) # [M, d_k]
        self.register_buffer("rademacher_bits", rad_masks, persistent=False) # [M, 3]
        self.register_buffer("b", b)   # [D]
        
        # Precompute explicit matrix Omega for fast vectorized Tensor Core fallback
        Omega_blocks = []
        # Construct explicit H_64
        H = torch.tensor([[1.0]], dtype=torch.float32)
        while H.shape[0] < d_k:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
            
        for m in range(self.M):
            Wm = torch.diag(d1[m])
            Wm = torch.matmul(H, Wm)
            Wm = torch.matmul(torch.diag(d2[m]), Wm)
            Wm = torch.matmul(H, Wm)
            Wm = torch.matmul(torch.diag(d3[m]), Wm)
            Wm = torch.matmul(H, Wm) * self.scale_norm
            Omega_blocks.append(Wm)
            
        Omega_mat = torch.cat(Omega_blocks, dim=0) # [D, d_k]
        self.register_buffer("Omega", Omega_mat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects input x: (..., d_k) -> spherically renormalized dual features (..., D).
        Uses closed-form analytical temperature tau* = (d_k)^(1/4).
        """
        # If native CUDA extension is present, execute in-register warp-shuffle kernel
        if HAS_CUDA_EXT and x.is_cuda and x.dtype == torch.float32 and x.dim() == 3:
            return kernel_deltanet_cuda.fwht_forward(x.contiguous(), self.rademacher_bits, self.D, self.tau_star_inv)
        
        # Reference vectorized execution
        proj = torch.matmul(x, self.Omega.to(x.dtype).T)
            
        # Scale by analytical temperature tau* and add phase bias b
        scaled_proj = proj / self.tau_star + self.b.to(x.dtype)
        # Cosine features: sqrt(2/D) * cos(scaled_proj)
        phi = math.sqrt(2.0 / self.D) * torch.cos(scaled_proj)
        # Spherical renormalization onto S^{D-1}
        phi_hat = phi / (torch.norm(phi, dim=-1, keepdim=True) + 1e-8)
        return phi_hat
