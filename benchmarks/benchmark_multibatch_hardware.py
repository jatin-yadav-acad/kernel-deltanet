"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: benchmarks.benchmark_multibatch_hardware
==================================================================
Realistic Multi-Batch Hardware Benchmarking on RTX 5070 Ti Silicon:
- Batch Sizes: B in [1, 4, 8, 16]
- Sequence Lengths: T in [1024, 2048, 4096, 8192, 16384]
- Evaluates 70M Scale Layer Configuration (d_model=512, heads=8, d_head=64, RKHS D=2048)
- Baseline Comparisons:
  * Kernel-DeltaNet (Ours, C=128, D=2048)
  * FlashAttention-2 (PyTorch SDPA Native CUDA C++ kernel)
  * Chunk-Parallel Linear DeltaNet
  * Mamba-2 / State Space Duality (1-Semiseparable Scan)

Logs tokens/sec, latency (ms), peak VRAM (MB), and GPU memory saturation.
"""

import os
import gc
import json
import time
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.kernel_deltanet_layer import KernelDeltaNetLayer
from baselines.standard_transformer import CausalSoftmaxAttention
from baselines.ttt_linear_deltanet import TTTLinearLayer
from evaluation.eval_fair_hardware_benchmark import Mamba2SSDLayer


def measure_layer_performance(
    layer: nn.Module,
    x: torch.Tensor,
    warmup: int = 3,
    repetitions: int = 10
) -> Tuple[float, float, float]:
    """
    Measures median forward latency (ms), throughput (tokens/sec), and peak VRAM (MB).
    """
    device = x.device
    B, T, _ = x.shape

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    # Warmup
    for _ in range(warmup):
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = layer(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats(device)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]

    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for r in range(repetitions):
            start_events[r].record()
            _ = layer(x)
            end_events[r].record()

    torch.cuda.synchronize()
    latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    median_latency_ms = float(torch.median(torch.tensor(latencies)).item())
    peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    throughput = (B * T) / (median_latency_ms / 1000.0)

    return median_latency_ms, throughput, peak_vram_mb


def run_multibatch_hardware_benchmark(
    batch_sizes: List[int] = [1, 4, 8, 16],
    sequence_lengths: List[int] = [1024, 2048, 4096, 8192, 16384],
    d_model: int = 512,
    num_heads: int = 8,
    rkhs_factor: int = 4, # D = 512 * 4 = 2048
    chunk_size: int = 128,
    output_path: str = "results/multibatch_hardware_benchmark.json"
) -> Dict:
    """
    Executes the full multi-batch hardware matrix on the RTX 5070 Ti.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 95)
    print("KERNEL-DELTANET: REALISTIC MULTI-BATCH HARDWARE BENCHMARKING (RTX 5070 Ti SILICON)")
    print("=" * 95)
    print(f"[*] Hardware Silicon: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[*] Batch Sizes: {batch_sizes} | Sequence Lengths: {sequence_lengths}")
    print(f"[*] Layer Configuration: d_model={d_model}, heads={num_heads}, RKHS D={d_model*rkhs_factor}, chunk_size={chunk_size}")

    layers = {
        "Kernel-DeltaNet (Ours, C=128)": KernelDeltaNetLayer(
            d_model=d_model,
            num_heads=num_heads,
            rkhs_factor=rkhs_factor,
            chunk_size=chunk_size
        ).to(device),
        "FlashAttention-2 (PyTorch SDPA)": CausalSoftmaxAttention(
            d_model=d_model,
            num_heads=num_heads
        ).to(device),
        "Chunk-Parallel DeltaNet": TTTLinearLayer(
            d_model=d_model,
            chunk_size=chunk_size
        ).to(device),
        "Mamba-2 / SSD": Mamba2SSDLayer(
            d_model=d_model,
            num_heads=num_heads,
            chunk_size=chunk_size
        ).to(device)
    }

    results = {}

    for b in batch_sizes:
        b_key = f"B_{b}"
        results[b_key] = {}
        print(f"\n[*] ==================== Batch Size B = {b} ====================")

        for t in sequence_lengths:
            t_key = f"T_{t}"
            results[b_key][t_key] = {}
            total_tokens = b * t

            # Safety check: if total tokens > 131,072 and memory is constrained, skip safely
            if total_tokens > 131072 and b == 16 and t == 16384:
                print(f"  -> Skipping B={b}, T={t} (Total {total_tokens:,} tokens exceeds single-layer VRAM limit)")
                continue

            print(f"  -> Testing Horizon T = {t:,} (Total Batch Tokens = {total_tokens:,})...")

            for name, layer in layers.items():
                layer.eval()
                try:
                    x = torch.randn(b, t, d_model, device=device, dtype=torch.bfloat16)
                    lat_ms, tok_s, vram_mb = measure_layer_performance(layer, x, warmup=2, repetitions=5)
                    results[b_key][t_key][name] = {
                        "latency_ms": round(lat_ms, 2),
                        "throughput_tok_per_sec": round(tok_s, 1),
                        "peak_vram_mb": round(vram_mb, 2),
                        "status": "SUCCESS"
                    }
                    print(f"     {name:35s}: {lat_ms:6.2f} ms | {tok_s:10,.0f} tok/s | VRAM: {vram_mb:6.1f} MB")
                except torch.cuda.OutOfMemoryError:
                    print(f"     {name:35s}: [!] OUT OF MEMORY (OOM)")
                    results[b_key][t_key][name] = {
                        "latency_ms": None,
                        "throughput_tok_per_sec": 0.0,
                        "peak_vram_mb": None,
                        "status": "OOM"
                    }
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    print(f"     {name:35s}: [!] Error: {str(e)[:30]}")
                    results[b_key][t_key][name] = {
                        "latency_ms": None,
                        "throughput_tok_per_sec": 0.0,
                        "peak_vram_mb": None,
                        "status": f"ERROR: {str(e)[:25]}"
                    }

    master_report = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "d_model": d_model,
        "num_heads": num_heads,
        "rkhs_dim": d_model * rkhs_factor,
        "chunk_size": chunk_size,
        "batch_sizes": batch_sizes,
        "sequence_lengths": sequence_lengths,
        "matrix": results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    print(f"\n[+] Multi-batch hardware benchmark logged to {output_path}")
    return master_report


if __name__ == "__main__":
    run_multibatch_hardware_benchmark()
