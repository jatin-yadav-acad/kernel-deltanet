r"""
Kernel-DeltaNet Engine
Test: test_fused_chunk_kernel.py
================================
Verifies mathematical equivalence and exact analytical autograd gradients
between custom fused_chunked_scan and PyTorch reference scans.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from core.chunked_scan import chunked_affine_scan
from kernels.fused_chunk_kernel import fused_chunked_scan


def test_fused_kernel_equivalence():
    print("[*] Running TEST-1.10: Fused Chunk Kernel Equivalence...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    B = 2
    H = 4
    T = 128
    D = 64
    d_v = 16
    chunk_size = 32

    phi_k = torch.randn(B, H, T, D, device=device)
    phi_q = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, d_v, device=device)
    decay = torch.full((H,), 0.99, device=device)
    eta = torch.full((H,), 0.5, device=device)

    # Standard reference scan
    with torch.no_grad():
        ref_out = chunked_affine_scan(phi_k=phi_k, phi_q=phi_q, v=v, decay=decay, eta=eta, chunk_size=chunk_size)

    # Fused chunked scan
    with torch.no_grad():
        fused_out = fused_chunked_scan(phi_k, phi_q, v, decay, eta, chunk_size=chunk_size)

    l2_diff = torch.norm(ref_out - fused_out).item()
    rel_diff = l2_diff / (torch.norm(ref_out).item() + 1e-9)

    print(f"    Fused vs Reference L2 Diff:   {l2_diff:.6e}")
    print(f"    Fused vs Reference Rel Diff:  {rel_diff:.6e}")

    assert rel_diff < 1e-5, f"Fused scan differs from reference scan: {rel_diff}"
    print("[+] TEST-1.10 PASSED: Fused Chunk Kernel Matches Reference Scan.\n")


def test_fused_kernel_gradcheck():
    print("[*] Running TEST-1.10b: Fused Chunk Kernel Numerical Gradcheck...")
    torch.manual_seed(42)

    B, H, T, D, d_v = 2, 2, 8, 4, 4
    C = 4

    pk = torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True)
    pq = torch.randn(B, H, T, D, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, H, T, d_v, dtype=torch.float64, requires_grad=True)
    decay = torch.tensor([0.95, 0.92], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([0.5, 0.4], dtype=torch.float64, requires_grad=True)

    def scan_wrapper(pk_in, pq_in, v_in, decay_in, eta_in):
        return fused_chunked_scan(pk_in, pq_in, v_in, decay_in, eta_in, chunk_size=C)

    passed = torch.autograd.gradcheck(scan_wrapper, (pk, pq, v, decay, eta), eps=1e-6, atol=1e-4)
    assert passed, "Gradcheck failed for fused_chunked_scan!"
    print("[+] TEST-1.10b PASSED: Fused Chunk Kernel Autograd Gradcheck Verified.\n")


if __name__ == "__main__":
    test_fused_kernel_equivalence()
    test_fused_kernel_gradcheck()
