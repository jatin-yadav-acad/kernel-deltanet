"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: evaluation.eval_hardware_roofline
==========================================================
Rigorous Hardware Roofline Compute and Memory Bandwidth Profiling on RTX 5070 Ti.
- Evaluates Arithmetic Intensity (FLOPs/Byte)
- Measures Achieved TFLOP/s and Achieved Memory Bandwidth (GB/s)
- Characterizes Memory-Bound vs Compute-Bound transitions as a function of:
  * Chunk Size C in [16, 32, 64, 128, 256, 512]
  * RKHS Expansion Factor r in [1, 2, 4, 8, 16] (D in [64, 1024])
- Maps operating points against physical RTX 5070 Ti hardware ceilings.
"""

import os
import json
import time
import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.chunked_scan import chunked_affine_scan
from core.rkhs_features import MultiHeadRKHSKernel


# Hardware constants for NVIDIA GeForce RTX 5070 Ti
RTX_5070_TI_SPECS = {
    "gpu_name": "NVIDIA GeForce RTX 5070 Ti",
    "peak_dense_bf16_tflops": 87.5,      # Theoretical dense BF16 Tensor Core peak
    "peak_sparse_bf16_tflops": 175.0,    # 2:4 structured sparsity ceiling
    "peak_memory_bandwidth_gb_s": 896.0, # 256-bit GDDR7 @ 28 Gbps
    "l2_cache_mb": 48.0,
    "ridge_point_flops_per_byte": 87.5 * 1000.0 / 896.0 # ~97.65 FLOPs/Byte
}


def calculate_chunked_kernel_deltanet_flops_and_bytes(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    d_head: int,
    rkhs_dim: int,
    chunk_size: int,
    dtype_bytes: int = 2 # BF16 = 2 bytes
) -> Tuple[int, int]:
    """
    Computes theoretical FLOPs and DRAM byte traffic for chunked Kernel-DeltaNet.
    
    Intra-chunk Gram formulation:
      1. Gram Matrix: G_c = phi_k @ phi_k^T -> 2 * B * H * C * C * rkhs_dim FLOPs
      2. Inversion / Causal Mask Solve: -> 2 * B * H * C * C * C / 3 FLOPs
      3. Value Projection: G_c_inv @ v -> 2 * B * H * C * C * d_head FLOPs
      4. Query Output: phi_q @ (phi_k^T @ v) -> 2 * B * H * C * C * d_head FLOPs
    Inter-chunk State Recurrence:
      5. State Update: S_c = S_{c-1} * decay + (phi_k^T @ v_res) -> 2 * B * H * C * rkhs_dim * d_head FLOPs
      6. Query Retrieval from State: phi_q @ S_{c-1} -> 2 * B * H * C * rkhs_dim * d_head FLOPs
    """
    num_chunks = math.ceil(seq_len / chunk_size)
    B = batch_size
    H = num_heads
    C = chunk_size
    D = rkhs_dim
    d = d_head

    # FLOPs per chunk
    gram_flops = 2 * B * H * C * C * D
    solve_flops = (2 * B * H * C * C * C) // 3
    value_proj_flops = 2 * B * H * C * C * d
    query_intra_flops = 2 * B * H * C * C * d
    state_update_flops = 2 * B * H * C * D * d
    state_query_flops = 2 * B * H * C * D * d

    flops_per_chunk = (
        gram_flops + solve_flops + value_proj_flops + 
        query_intra_flops + state_update_flops + state_query_flops
    )
    total_flops = flops_per_chunk * num_chunks

    # Byte traffic (Compulsory memory read/write assuming standard cache lines)
    # Reads: phi_k (B, H, T, D), v (B, H, T, d), phi_q (B, H, T, D), initial S
    # Writes: output v_out (B, H, T, d), final state S
    bytes_phi_k = B * H * seq_len * D * dtype_bytes
    bytes_v = B * H * seq_len * d * dtype_bytes
    bytes_phi_q = B * H * seq_len * D * dtype_bytes
    bytes_out = B * H * seq_len * d * dtype_bytes
    bytes_state = B * H * D * d * dtype_bytes * 2 # read + write

    total_bytes = bytes_phi_k + bytes_v + bytes_phi_q + bytes_out + bytes_state
    return int(total_flops), int(total_bytes)


def measure_kernel_scan_throughput(
    phi_k: torch.Tensor,
    v: torch.Tensor,
    phi_q: torch.Tensor,
    decay: torch.Tensor,
    eta: torch.Tensor,
    chunk_size: int,
    warmup: int = 5,
    repetitions: int = 20
) -> float:
    """
    Measures exact median execution latency of chunked_affine_scan in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        _ = chunked_affine_scan(phi_k, v, decay, eta, phi_q, chunk_size=chunk_size)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repetitions)]

    for r in range(repetitions):
        start_events[r].record()
        _ = chunked_affine_scan(phi_k, v, decay, eta, phi_q, chunk_size=chunk_size)
        end_events[r].record()

    torch.cuda.synchronize()
    latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return float(torch.median(torch.tensor(latencies)).item())


def run_hardware_roofline_benchmark(
    batch_size: int = 8,
    seq_len: int = 2048,
    num_heads: int = 4,
    d_head: int = 64,
    output_path: str = "results/hardware_roofline_benchmark.json"
) -> Dict:
    """
    Executes arithmetic intensity sweeps on RTX 5070 Ti silicon.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 90)
    print("KERNEL-DELTANET: HARDWARE ROOFLINE & ARITHMETIC INTENSITY PROFILING (RTX 5070 Ti)")
    print("=" * 90)
    print(f"[*] Target Hardware: {RTX_5070_TI_SPECS['gpu_name']}")
    print(f"[*] Peak Dense BF16 Compute: {RTX_5070_TI_SPECS['peak_dense_bf16_tflops']} TFLOP/s")
    print(f"[*] Peak Memory Bandwidth: {RTX_5070_TI_SPECS['peak_memory_bandwidth_gb_s']} GB/s")
    print(f"[*] Roofline Ridge Point: {RTX_5070_TI_SPECS['ridge_point_flops_per_byte']:.2f} FLOPs/Byte")

    # Sweep 1: Chunk Size C in [16, 32, 64, 128, 256, 512] at fixed RKHS factor r = 4 (D = 256)
    fixed_rkhs_dim = d_head * 4
    chunk_sizes = [16, 32, 64, 128, 256, 512]
    chunk_sweep_results = []

    print("\n[*] Running Sweep 1: Chunk Size Variation (C in [16, 512], D = 256)...")
    torch.manual_seed(42)
    phi_k = torch.randn(batch_size, num_heads, seq_len, fixed_rkhs_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(batch_size, num_heads, seq_len, d_head, device=device, dtype=torch.bfloat16)
    phi_q = torch.randn(batch_size, num_heads, seq_len, fixed_rkhs_dim, device=device, dtype=torch.bfloat16)
    decay = torch.full((1, num_heads, 1, 1), 0.99, device=device, dtype=torch.bfloat16)
    eta = torch.full((1, num_heads, 1, 1), 0.5, device=device, dtype=torch.bfloat16)

    for C in chunk_sizes:
        flops, memory_bytes = calculate_chunked_kernel_deltanet_flops_and_bytes(
            batch_size, seq_len, num_heads, d_head, fixed_rkhs_dim, C
        )
        arithmetic_intensity = flops / max(1, memory_bytes) # FLOPs/Byte
        latency_ms = measure_kernel_scan_throughput(phi_k, v, phi_q, decay, eta, chunk_size=C)

        achieved_tflops = (flops / (latency_ms * 1e-3)) / 1e12
        achieved_gb_s = (memory_bytes / (latency_ms * 1e-3)) / 1e9
        regime = "Compute-Bound" if arithmetic_intensity >= RTX_5070_TI_SPECS["ridge_point_flops_per_byte"] else "Memory-Bound"
        compute_efficiency_pct = (achieved_tflops / RTX_5070_TI_SPECS["peak_dense_bf16_tflops"]) * 100.0

        print(f"  C={C:3d} | Latency: {latency_ms:6.2f} ms | Arith. Intensity: {arithmetic_intensity:6.2f} FLOP/B | "
              f"TFLOP/s: {achieved_tflops:5.2f} ({compute_efficiency_pct:4.1f}%) | BW: {achieved_gb_s:5.1f} GB/s | [{regime}]")

        chunk_sweep_results.append({
            "chunk_size": C,
            "latency_ms": round(latency_ms, 3),
            "flops": flops,
            "memory_bytes": memory_bytes,
            "arithmetic_intensity_flops_per_byte": round(arithmetic_intensity, 2),
            "achieved_tflops": round(achieved_tflops, 3),
            "achieved_bandwidth_gb_s": round(achieved_gb_s, 2),
            "compute_efficiency_pct": round(compute_efficiency_pct, 2),
            "regime": regime
        })

    # Sweep 2: RKHS Dimension Factor r in [1, 2, 4, 8, 16] at optimal chunk size C = 128
    optimal_C = 128
    rkhs_factors = [1, 2, 4, 8, 16]
    rkhs_sweep_results = []

    print(f"\n[*] Running Sweep 2: RKHS Expansion Variation (r in [1, 16], C = {optimal_C})...")
    for r in rkhs_factors:
        D = d_head * r
        phi_k_r = torch.randn(batch_size, num_heads, seq_len, D, device=device, dtype=torch.bfloat16)
        phi_q_r = torch.randn(batch_size, num_heads, seq_len, D, device=device, dtype=torch.bfloat16)

        flops, memory_bytes = calculate_chunked_kernel_deltanet_flops_and_bytes(
            batch_size, seq_len, num_heads, d_head, D, optimal_C
        )
        arithmetic_intensity = flops / max(1, memory_bytes)
        latency_ms = measure_kernel_scan_throughput(phi_k_r, v, phi_q_r, decay, eta, chunk_size=optimal_C)

        achieved_tflops = (flops / (latency_ms * 1e-3)) / 1e12
        achieved_gb_s = (memory_bytes / (latency_ms * 1e-3)) / 1e9
        regime = "Compute-Bound" if arithmetic_intensity >= RTX_5070_TI_SPECS["ridge_point_flops_per_byte"] else "Memory-Bound"
        compute_efficiency_pct = (achieved_tflops / RTX_5070_TI_SPECS["peak_dense_bf16_tflops"]) * 100.0

        print(f"  r={r:2d} (D={D:4d}) | Latency: {latency_ms:6.2f} ms | Arith. Intensity: {arithmetic_intensity:6.2f} FLOP/B | "
              f"TFLOP/s: {achieved_tflops:5.2f} ({compute_efficiency_pct:4.1f}%) | BW: {achieved_gb_s:5.1f} GB/s | [{regime}]")

        rkhs_sweep_results.append({
            "rkhs_factor": r,
            "rkhs_dim": D,
            "latency_ms": round(latency_ms, 3),
            "flops": flops,
            "memory_bytes": memory_bytes,
            "arithmetic_intensity_flops_per_byte": round(arithmetic_intensity, 2),
            "achieved_tflops": round(achieved_tflops, 3),
            "achieved_bandwidth_gb_s": round(achieved_gb_s, 2),
            "compute_efficiency_pct": round(compute_efficiency_pct, 2),
            "regime": regime
        })

    roofline_report = {
        "hardware_specs": RTX_5070_TI_SPECS,
        "parameters": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "d_head": d_head
        },
        "chunk_size_sweep": chunk_sweep_results,
        "rkhs_factor_sweep": rkhs_sweep_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(roofline_report, f, indent=2)

    print(f"\n[+] Hardware roofline benchmark logged to {output_path}")
    return roofline_report


if __name__ == "__main__":
    run_hardware_roofline_benchmark()
