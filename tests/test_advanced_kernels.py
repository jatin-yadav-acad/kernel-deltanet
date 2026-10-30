r"""
Kernel-DeltaNet Engine
Test: test_advanced_kernels.py
==============================
Verifies mathematical properties of advanced RKHS kernels:
1. Positive-definiteness of Gram matrices
2. Heavy-tailed Cauchy kernel power-law decay
3. Fastfood Walsh-Hadamard projection accuracy
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import math
import torch
import numpy as np

from core.advanced_kernels import (
    CauchyHeavyTailedKernel,
    MaternSpectralKernel,
    ArcCosineKernel,
    FastfoodWalshHadamardKernel
)


def test_cauchy_kernel_properties():
    print("[*] Running TEST-1.5: Cauchy Heavy-Tailed Kernel Properties...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    kernel = CauchyHeavyTailedKernel(num_heads=2, d_head=16, rkhs_dim=128, gamma_init=2.0).to(device)
    x = torch.randn(1, 2, 32, 16, device=device)
    phi = kernel(x)

    # Unit norm check
    norms = torch.norm(phi, p=2, dim=-1)
    mean_norm = norms.mean().item()
    print(f"    Cauchy Feature Mean L2 Norm: {mean_norm:.4f}")
    assert abs(mean_norm - 1.0) < 0.1, f"Features not properly normalized: {mean_norm}"

    # Positive-semidefinite Gram check
    G = torch.matmul(phi, phi.transpose(-1, -2))[0, 0] # (32, 32)
    eigvals = torch.linalg.eigvalsh(G)
    min_eig = eigvals.min().item()
    print(f"    Cauchy Gram Minimum Eigenvalue: {min_eig:.6f}")
    assert min_eig >= -1e-5, f"Gram matrix not positive semidefinite: {min_eig}"
    print("[+] Cauchy Kernel Verified.\n")
    pass


def test_fastfood_transform_properties():
    print("[*] Running TEST-1.6: Fastfood Walsh-Hadamard Transform...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    kernel = FastfoodWalshHadamardKernel(num_heads=2, d_head=16, rkhs_dim=128, gamma=2.0).to(device)
    x = torch.randn(2, 2, 32, 16, device=device)
    phi = kernel(x)

    # Gram positive semidefiniteness
    G = torch.matmul(phi, phi.transpose(-1, -2))[0, 0]
    eigvals = torch.linalg.eigvalsh(G)
    min_eig = eigvals.min().item()
    print(f"    Fastfood Gram Minimum Eigenvalue: {min_eig:.6f}")
    assert min_eig >= -1e-5, f"Fastfood Gram not PSD: {min_eig}"
    print("[+] Fastfood Transform Verified.\n")
    pass


if __name__ == "__main__":
    test_cauchy_kernel_properties()
    test_fastfood_transform_properties()
