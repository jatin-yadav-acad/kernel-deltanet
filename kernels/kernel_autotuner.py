r"""
Kernel-DeltaNet Engine
Module: kernels.kernel_autotuner
=================================================
Kernel Chunk Size Autotuner for NVIDIA RTX 5070 Ti.
Searches over chunk sizes C in {16, 32, 64, 128, 256}
to optimize hardware throughput, cache reuse, and register pressure.
"""

import os
import json
import time
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np

from core.chunked_scan import chunked_affine_scan
from core.rkhs_features import MultiHeadRKHSKernel


def autotune_chunk_size(
    seq_len: int = 4096,
    d_model: int = 64,
    num_heads: int = 4,
    rkhs_dim: int = 256,
    candidate_chunks: List[int] = [16, 32, 64, 128, 256],
    num_trials: int = 20,
    save_path: str = "results/autotuning_results.json"
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"KERNEL-DELTANET HARDWARE CHUNK SIZE AUTOTUNER ON {device}")
    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Sequence Length T = {seq_len}, d_model = {d_model}, RKHS Dim = {rkhs_dim}")
    print("=" * 80)

    d_head = d_model // num_heads
    d_v = d_head
    B = 1

    kernel = MultiHeadRKHSKernel(num_heads=num_heads, d_head=d_head, rkhs_dim=rkhs_dim).to(device)
    x = torch.randn(B, num_heads, seq_len, d_head, device=device)
    phi = kernel(x)
    v = torch.randn(B, num_heads, seq_len, d_v, device=device)
    decay = torch.full((num_heads,), 0.99, device=device)
    eta = torch.full((num_heads,), 0.5, device=device)

    results = {}
    best_c = candidate_chunks[0]
    best_speed = 0.0

    for C in candidate_chunks:
        # Warmup
        for _ in range(5):
            _ = chunked_affine_scan(phi_k=phi, phi_q=phi, v=v, decay=decay, eta=eta, chunk_size=C)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(num_trials):
            t0 = time.perf_counter()
            _ = chunked_affine_scan(phi_k=phi, phi_q=phi, v=v, decay=decay, eta=eta, chunk_size=C)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(times_ms))
        std_ms = float(np.std(times_ms))
        throughput = float((B * seq_len) / (mean_ms / 1000.0))

        results[f"chunk_{C}"] = {
            "chunk_size": C,
            "mean_latency_ms": mean_ms,
            "std_latency_ms": std_ms,
            "throughput_tok_per_sec": throughput
        }

        print(f"  Chunk Size C = {C:>3} | Latency: {mean_ms:>6.2f} +/- {std_ms:.2f} ms | Throughput: {throughput:>10.1f} tok/s")

        if throughput > best_speed:
            best_speed = throughput
            best_c = C

    results["optimal_chunk_size"] = best_c
    results["optimal_throughput_tok_per_sec"] = best_speed

    print("=" * 80)
    print(f"[+] OPTIMAL HARDWARE CONFIGURATION: Chunk Size C = {best_c} ({best_speed:,.1f} tok/s)")
    print("=" * 80)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    autotune_chunk_size()
