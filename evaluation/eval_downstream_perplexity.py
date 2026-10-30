r"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: evaluation.eval_downstream_perplexity
==============================================================
Downstream Causal Language Modeling Perplexity Evaluator.
Loads pre-trained converged checkpoints (Scale-35M, Scale-70M, Scale-125M)
and evaluates Cross-Entropy Loss and Perplexity (PPL) on FineWeb-Edu validation text
across evaluation context horizons T in [128, 256, 512, 1024, 2048].
"""

import os
import json
import math
import time
from typing import Dict, List
import torch
import torch.nn as nn
import numpy as np

from models.kernel_deltanet_lm import KernelDeltaNetLM
from baselines.standard_transformer import StandardTransformerLM
from training.run_converged_pretraining import load_fineweb_byte_corpus


def evaluate_checkpoint_across_horizons(
    checkpoint_path: str,
    model_family: str,
    d_model: int,
    num_layers: int,
    d_rkhs: int,
    val_bytes: torch.Tensor,
    horizons: List[int] = [128, 256, 512, 1024, 2048],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Dict]:
    vocab_size = 256
    num_heads = max(2, d_model // 64)

    if model_family == "kernel_deltanet":
        rkhs_factor = d_rkhs // d_model
        model = KernelDeltaNetLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            rkhs_factor=rkhs_factor,
            chunk_size=64
        ).to(device)
    else:
        model = StandardTransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads
        ).to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    val_data = val_bytes.to(device)
    total_val_len = val_data.numel()

    results = {}
    for h in horizons:
        if total_val_len < (h + 1) * 8:
            continue
        batch_size = 4 if h <= 1024 else 2
        num_batches = 8

        losses = []
        latencies = []
        with torch.no_grad():
            for b in range(num_batches):
                start = (b * batch_size * (h + 1)) % (total_val_len - batch_size * (h + 1))
                inputs = torch.stack([val_data[start + i * h : start + (i + 1) * h] for i in range(batch_size)])
                targets = torch.stack([val_data[start + i * h + 1 : start + (i + 1) * h + 1] for i in range(batch_size)])

                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(inputs, targets=targets)
                if device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                losses.append(loss.item())
                latencies.append((t1 - t0) * 1000.0)

        mean_loss = float(np.mean(losses))
        ppl = float(math.exp(min(20.0, mean_loss)))
        mean_lat = float(np.mean(latencies))

        results[f"T_{h}"] = {
            "eval_seq_len": h,
            "cross_entropy_loss": mean_loss,
            "perplexity": ppl,
            "eval_latency_ms": mean_lat
        }

    return results


def run_perplexity_benchmark(
    save_path: str = "results/downstream_perplexity_benchmark.json"
) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 100)
    print(f"CONVERGED CHECKPOINT PERPLEXITY (PPL) BENCHMARK ON FINEWEB-EDU VAL DATA ({device})")
    print("=" * 100)

    _, val_bytes = load_fineweb_byte_corpus()

    scale_configs = [
        {"name": "Scale-35M", "d_model": 384, "num_layers": 8, "d_rkhs": 1536},
        {"name": "Scale-70M", "d_model": 512, "num_layers": 10, "d_rkhs": 2048},
        {"name": "Scale-125M", "d_model": 768, "num_layers": 12, "d_rkhs": 3072}
    ]

    all_results = {}
    for sc in scale_configs:
        scale_name = sc["name"]
        print(f"\nEvaluating Scale: {scale_name}...")
        all_results[scale_name] = {}

        # 1. Kernel-DeltaNet Checkpoint
        kd_ckpt = f"checkpoints/kernel_deltanet_{scale_name.lower()}_converged.pt"
        if os.path.exists(kd_ckpt):
            print(f"  -> Evaluating Kernel-DeltaNet Checkpoint ({kd_ckpt})...")
            kd_res = evaluate_checkpoint_across_horizons(
                checkpoint_path=kd_ckpt,
                model_family="kernel_deltanet",
                d_model=sc["d_model"],
                num_layers=sc["num_layers"],
                d_rkhs=sc["d_rkhs"],
                val_bytes=val_bytes,
                horizons=[128, 256, 512, 1024, 2048],
                device=device
            )
            all_results[scale_name]["Kernel-DeltaNet"] = kd_res
            for k, v in kd_res.items():
                print(f"     {k:<8} | Loss: {v['cross_entropy_loss']:.4f} | PPL: {v['perplexity']:>6.2f} | Latency: {v['eval_latency_ms']:>5.2f} ms")

        # 2. Standard Transformer Checkpoint
        tf_ckpt = f"checkpoints/transformer_{scale_name.lower()}_converged.pt"
        if os.path.exists(tf_ckpt):
            print(f"  -> Evaluating Standard Transformer Checkpoint ({tf_ckpt})...")
            tf_res = evaluate_checkpoint_across_horizons(
                checkpoint_path=tf_ckpt,
                model_family="transformer",
                d_model=sc["d_model"],
                num_layers=sc["num_layers"],
                d_rkhs=sc["d_rkhs"],
                val_bytes=val_bytes,
                horizons=[128, 256, 512, 1024, 2048],
                device=device
            )
            all_results[scale_name]["Transformer"] = tf_res
            for k, v in tf_res.items():
                print(f"     {k:<8} | Loss: {v['cross_entropy_loss']:.4f} | PPL: {v['perplexity']:>6.2f} | Latency: {v['eval_latency_ms']:>5.2f} ms")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[+] Converged downstream perplexity results saved to {save_path}")
    return all_results


if __name__ == "__main__":
    run_perplexity_benchmark()
