"""
Kernel-DeltaNet Engine
Test: test_parallel_vs_recurrent.py
===================================
Automated verification of Parallel Chunked Scan vs Sequential Recurrence Equivalence.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import math
import torch
from core.chunked_scan import chunked_affine_scan, recurrent_step
from layers.kernel_deltanet_layer import KernelDeltaNetLayer


def test_chunked_scan_vs_recurrence():
    print("[*] Running TEST-1.2: Parallel Scan vs Recurrence Equivalence...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    B, H, T, D, d_v = 2, 4, 128, 64, 32
    phi = torch.randn(B, H, T, D, device=device)
    phi = torch.nn.functional.normalize(phi, p=2, dim=-1)
    v = torch.randn(B, H, T, d_v, device=device)
    decay = torch.full((1, H, 1, 1), 0.98, device=device)
    eta = torch.full((1, H, 1, 1), 0.6, device=device)

    # 1. Parallel Chunked Scan
    out_parallel = chunked_affine_scan(phi, v, decay, eta, chunk_size=32)

    # 2. Sequential Recurrent Scan
    out_rec = []
    state = None
    for t in range(T):
        phi_t = phi[:, :, t:t+1, :]
        v_t = v[:, :, t:t+1, :]
        q_t = phi_t # self-readout for identity test
        y_t, state = recurrent_step(phi_t, v_t, q_t, state, decay, eta)
        out_rec.append(y_t)
    out_recurrent = torch.cat(out_rec, dim=2)

    l2_diff = torch.norm(out_parallel - out_recurrent).item()
    rel_diff = l2_diff / (torch.norm(out_recurrent).item() + 1e-9)

    print(f"    L2 Absolute Difference: {l2_diff:.6e}")
    print(f"    Relative Difference:    {rel_diff:.6e}")

    assert rel_diff < 1e-5, f"Fatal: Parallel scan diverges from recurrence! Rel diff: {rel_diff}"
    print("[+] TEST-1.2 PASSED: Parallel Scan Matches Recurrence down to machine precision.\n")
    pass


def test_layer_dual_mode_equivalence():
    print("[*] Running TEST-1.2b: KernelDeltaNetLayer Dual-Mode Forward Equivalence...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(123)

    layer = KernelDeltaNetLayer(d_model=64, num_heads=4, rkhs_factor=4, chunk_size=32).to(device)
    layer.eval()

    x = torch.randn(2, 64, 64, device=device)

    # Parallel mode
    with torch.no_grad():
        out_parallel, _ = layer(x, state=None)

    # Recurrent mode
    with torch.no_grad():
        out_rec, _ = layer(x, state=torch.zeros(2, 4, 16, 64, device=device))

    l2_diff = torch.norm(out_parallel - out_rec).item()
    rel_diff = l2_diff / (torch.norm(out_rec).item() + 1e-9)

    print(f"    Layer L2 Absolute Difference: {l2_diff:.6e}")
    print(f"    Layer Relative Difference:    {rel_diff:.6e}")

    assert rel_diff < 1e-5, f"Fatal: Layer dual-mode divergence! Rel diff: {rel_diff}"
    print("[+] TEST-1.2b PASSED: Layer Dual-Mode Equivalence Verified.\n")
    pass


if __name__ == "__main__":
    test_chunked_scan_vs_recurrence()
    test_layer_dual_mode_equivalence()
