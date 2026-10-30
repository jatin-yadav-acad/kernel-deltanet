"""
Kernel-DeltaNet Engine
Module: evaluation.eval_hardware_throughput
=============================================================
Hardware-Level Throughput & Latency Cliff Benchmark on NVIDIA GPU.
"""

import os
import json
import time
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np

from layers.kernel_deltanet_layer import KernelDeltaNetLayer
from baselines.standard_transformer import CausalSoftmaxAttention
from baselines.ttt_linear_deltanet import TTTLinearLayer
from baselines.ttt_mlp_sequential import TTTMLPLayer


def run_throughput_benchmark(
    d_model: int = 64,
    batch_size: int = 1,
    seq_lengths: List[int] = [512, 1024, 2048, 4096],
    num_warmup: int = 3,
    num_trials: int = 10,
    save_path: str = "results/throughput_benchmark.json"
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print(f"HARDWARE THROUGHPUT & STEP LATENCY BENCHMARK ON {device}")
    if device.type == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print("=" * 90)

    models = {
        "Standard Transformer (Softmax)": CausalSoftmaxAttention(d_model=d_model, num_heads=4).to(device),
        "TTT-Linear (DeltaNet Scan)":     TTTLinearLayer(d_model=d_model, chunk_size=32).to(device),
        "TTT-MLP (Sequential Inner GELU)": TTTMLPLayer(d_model=d_model, inner_hidden=128).to(device),
        "Kernel-DeltaNet (Our Closed-Form Dual)": KernelDeltaNetLayer(d_model=d_model, num_heads=4, rkhs_factor=4, chunk_size=32).to(device)
    }

    results = {}

    for T in seq_lengths:
        print(f"\n--- Sequence Length T = {T:>4} Tokens ---")
        cfg_key = f"T_{T}"
        results[cfg_key] = {}

        x = torch.randn(batch_size, T, d_model, device=device)

        for name, m in models.items():
            m.eval()
            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    _ = m(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Benchmark trials
            times_ms = []
            with torch.no_grad():
                for _ in range(num_trials):
                    t0 = time.perf_counter()
                    _ = m(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    times_ms.append((t1 - t0) * 1000.0)

            mean_ms = float(np.mean(times_ms))
            tokens_per_sec = float((batch_size * T) / (mean_ms / 1000.0))

            results[cfg_key][name] = {
                "step_ms": mean_ms,
                "tokens_per_sec": tokens_per_sec
            }
            print(f"  -> {name:<36} | Latency: {mean_ms:>8.2f} ms | Throughput: {tokens_per_sec:>9.1f} tok/s")

        # Compute speedup of Kernel-DeltaNet over TTT-MLP
        mlp_ms = results[cfg_key]["TTT-MLP (Sequential Inner GELU)"]["step_ms"]
        ker_ms = results[cfg_key]["Kernel-DeltaNet (Our Closed-Form Dual)"]["step_ms"]
        speedup = mlp_ms / (ker_ms + 1e-9)
        results[cfg_key]["speedup_kernel_over_mlp"] = float(speedup)
        print(f"  ==> SPEEDUP of Kernel-DeltaNet over TTT-MLP: {speedup:.2f}x Faster!")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nBenchmark results saved to {save_path}")
    return results


if __name__ == "__main__":
    run_throughput_benchmark()
