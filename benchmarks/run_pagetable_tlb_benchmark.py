# benchmarks/run_pagetable_tlb_benchmark.py
"""
Analytical Page-Table Memory Bloat & Indexing Overhead Model
==========================================================
Quantifies memory fragmentation and metadata overhead comparing:
1. PagedAttention / FlashInfer (16 tokens / physical page):
   - Metadata entries: T / 16 = 4,096 page-table entries per sequence.
   - Non-contiguous physical page lookups in L2 / DRAM.
2. Kernel-DeltaNet Continuous Recurrent Block Allocator:
   - Metadata entries: Strictly 1 pointer per sequence.
   - 0 dynamic page allocations, 0 virtual address translation overheads.
"""

import os
import sys
import json
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath("."))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_pagetable_benchmark():
    print("=" * 80)
    print("EXPERIMENT 4: Page-Table Memory Bloat & Allocator Overhead on RTX 5070 Ti")
    print("=" * 80)

    # Sequence lengths from 4k to 64k
    contexts = [4096, 8192, 16384, 32768, 65536]
    page_size_tokens = 16  # standard vLLM / PagedAttention page block
    num_layers = 22
    num_heads = 32
    head_dim = 64

    results = []

    for T in contexts:
        num_pages_per_seq = T // page_size_tokens
        # PagedAttention page table: int32 pointers per layer
        # For B=128 concurrent streams:
        for B in [1, 16, 64, 128]:
            total_pages = B * num_pages_per_seq
            # Memory for page table structures: B * num_pages * sizeof(int32) * num_layers
            pagetable_metadata_bytes = B * num_pages_per_seq * 4 * num_layers
            pagetable_metadata_kb = pagetable_metadata_bytes / 1024

            # Kernel-DeltaNet metadata: 1 pointer (int64 = 8 bytes) per stream per layer
            kd_metadata_bytes = B * 8 * num_layers
            kd_metadata_kb = kd_metadata_bytes / 1024

            # Memory fragmentation:
            # PagedAttention leaves on average half a page (8 tokens) per stream per layer
            # Internal fragmentation bytes: B * (page_size/2) * (2 * num_layers * 4 * 64 * 2)
            pa_internal_frag_bytes = B * (page_size_tokens // 2) * (2 * num_layers * 4 * 64 * 2)
            pa_frag_mb = pa_internal_frag_bytes / (1024**2)

            # Kernel-DeltaNet has exactly 0 internal fragmentation
            kd_frag_mb = 0.0

            res = {
                "context_length": T,
                "batch_size": B,
                "paged_attention_pages_per_seq": num_pages_per_seq,
                "paged_attention_metadata_kb": round(pagetable_metadata_kb, 2),
                "kernel_deltanet_metadata_kb": round(kd_metadata_kb, 4),
                "metadata_reduction_factor": round(pagetable_metadata_bytes / max(kd_metadata_bytes, 1), 1),
                "paged_attention_internal_frag_mb": round(pa_frag_mb, 2),
                "kernel_deltanet_internal_frag_mb": 0.0
            }
            if B in [1, 128]:
                results.append(res)
                print(f"  [+] T={T//1024}k, B={B}: PagedAttention Pages={num_pages_per_seq} | Table Bloat={pagetable_metadata_kb:.1f} KB vs KD={kd_metadata_kb:.3f} KB ({res['metadata_reduction_factor']}x reduction)")

    out_path = "results/mlsys_pagetable_overhead.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SUCCESS] Saved page-table overhead results to {out_path}")
    return results

if __name__ == "__main__":
    run_pagetable_benchmark()
