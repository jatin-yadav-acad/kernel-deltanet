"""
Kernel-DeltaNet Engine
Module: training.scaling_harness
=================================================
Empirical Chinchilla Compute Scaling Suite on FineWeb-Edu.

Fits power-law scaling relationships:
    L(C) = E + A / (C^gamma)
where C is training compute in FLOPs (approx 6 * P_non_embed * Tokens),
E is the irreducible entropy floor, and gamma is the scaling exponent.
"""

import os
import json
import math
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.kernel_deltanet_lm import KernelDeltaNetLM
from baselines.standard_transformer import StandardTransformerLM
from data.prepare_fineweb_edu import prepare_fineweb_slice
from .optimizer_utils import configure_optimizers, CosineWarmupLRScheduler


class FineWebEduDataLoader:
    """
    High-throughput zero-overhead batch loader for pre-tokenized FineWeb-Edu tensors.
    """
    def __init__(
        self,
        cache_dir: str = "data/fineweb_cache",
        batch_size: int = 4,
        seq_len: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

        train_path = os.path.join(cache_dir, "fineweb_train.pt")
        val_path = os.path.join(cache_dir, "fineweb_val.pt")

        if not os.path.exists(train_path) or not os.path.exists(val_path):
            print("[*] Cache missing, triggering FineWeb-Edu download...")
            train_t, val_t = prepare_fineweb_slice(save_dir=cache_dir)
        else:
            train_t = torch.load(train_path, weights_only=True)
            val_t = torch.load(val_path, weights_only=True)

        self.train_tokens = train_t.to(device)
        self.val_tokens = val_t.to(device)
        self.n_train = self.train_tokens.numel()
        self.n_val = self.val_tokens.numel()

    def sample_train_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        max_idx = self.n_train - self.seq_len - 1
        starts = torch.randint(0, max_idx, (self.batch_size,), device=self.device)
        inputs = torch.stack([self.train_tokens[s : s + self.seq_len] for s in starts])
        targets = torch.stack([self.train_tokens[s + 1 : s + self.seq_len + 1] for s in starts])
        return inputs, targets

    def evaluate_val_loss(self, model: nn.Module, max_eval_batches: int = 10) -> float:
        model.eval()
        total_loss = 0.0
        batches = 0
        chunk_len = self.batch_size * self.seq_len
        with torch.no_grad():
            for i in range(0, min(self.n_val - self.seq_len - 1, max_eval_batches * chunk_len), chunk_len):
                if i + self.batch_size * self.seq_len + 1 > self.n_val:
                    break
                sub = self.val_tokens[i : i + self.batch_size * self.seq_len + 1]
                inputs = sub[:-1].view(self.batch_size, self.seq_len)
                targets = sub[1:].view(self.batch_size, self.seq_len)
                _, loss = model(inputs, targets=targets)
                total_loss += loss.item()
                batches += 1
        model.train()
        return float(total_loss / max(1, batches))


def train_scaling_checkpoint(
    model_family: str,
    scale_name: str,
    d_model: int,
    num_layers: int,
    d_rkhs: Optional[int],
    target_tokens: int,
    loader: FineWebEduDataLoader,
    vocab_size: int = 50304,
    batch_size: int = 4,
    seq_len: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: str = "checkpoints"
) -> Dict:
    """
    Trains a single architecture checkpoint on FineWeb-Edu tokens.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    num_heads = max(2, d_model // 64)

    if model_family == "kernel_ttt":
        rkhs_factor = d_rkhs // d_model if d_rkhs else 4
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

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_embed_params = model.get_num_params(non_embedding=True)
    total_flops = 6 * non_embed_params * target_tokens

    print(f"\n==========================================================================")
    print(f"[*] Launching {model_family.upper()} [{scale_name}] on FineWeb-Edu")
    print(f"    Params: {param_count/1e6:.2f}M (Non-embed: {non_embed_params/1e6:.2f}M) | Target Tokens: {target_tokens:,}")
    print(f"    Target Compute: {total_flops:.2e} FLOPs | Device: {device}")
    print(f"==========================================================================")

    optimizer = configure_optimizers(model, weight_decay=0.01, learning_rate=8e-4)
    tokens_per_step = batch_size * seq_len
    total_steps = max(1, target_tokens // tokens_per_step)
    warmup_steps = max(5, total_steps // 10)
    scheduler = CosineWarmupLRScheduler(optimizer, warmup_steps=warmup_steps, max_steps=total_steps)

    tokens_processed = 0
    step = 0
    t0 = time.time()
    train_losses = []
    val_loss = loader.evaluate_val_loss(model, max_eval_batches=5)
    print(f"    [Step 0000] Initial Validation Loss: {val_loss:.4f}")

    model.train()
    while tokens_processed < target_tokens:
        inputs, targets = loader.sample_train_batch()
        optimizer.zero_grad()
        logits, loss = model(inputs, targets=targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tokens_processed += tokens_per_step
        step += 1
        train_losses.append(loss.item())

        if step % 25 == 0 or tokens_processed >= target_tokens:
            val_loss = loader.evaluate_val_loss(model, max_eval_batches=5)
            elapsed = time.time() - t0
            tok_per_sec = tokens_processed / (elapsed + 1e-7)
            vram_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            print(f"    [Step {step:04d}/{total_steps:04d}] Tokens: {tokens_processed/1e3:.1f}k | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | {tok_per_sec:.0f} tok/s | Peak VRAM: {vram_mb:.0f} MB")

    total_time = time.time() - t0
    final_val_loss = loader.evaluate_val_loss(model, max_eval_batches=15)
    final_train_loss = float(np.mean(train_losses[-10:]))
    throughput = tokens_processed / (total_time + 1e-7)

    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"{model_family}_{scale_name.lower()}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_family": model_family,
        "scale_name": scale_name,
        "d_model": d_model,
        "num_layers": num_layers,
        "d_rkhs": d_rkhs,
        "final_val_loss": final_val_loss,
        "total_flops": total_flops
    }, ckpt_path)
    print(f"[+] Saved Checkpoint -> {ckpt_path}")
    print(f"[+] Completed {scale_name}: Final Val Loss = {final_val_loss:.4f} in {total_time:.2f}s ({throughput:.0f} tok/s)\n")

    return {
        "model_family": model_family,
        "scale_name": scale_name,
        "params": int(param_count),
        "non_embed_params": int(non_embed_params),
        "target_tokens": int(target_tokens),
        "total_flops": float(total_flops),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "throughput_tok_per_sec": float(throughput),
        "elapsed_seconds": float(total_time),
        "checkpoint_path": ckpt_path
    }


def fit_scaling_power_law(points: List[Dict]) -> Dict:
    """
    Fits: log10(Val_Loss) = a + b * log10(FLOPs)
    => L(C) = A * C^(-gamma), where gamma = -b.
    """
    if len(points) < 2:
        return {"gamma": 0.0, "A": 0.0, "r2": 0.0}

    flops = np.array([p["total_flops"] for p in points])
    losses = np.array([p["final_val_loss"] for p in points])

    log_c = np.log10(flops)
    log_l = np.log10(losses)

    poly = np.polyfit(log_c, log_l, 1)
    slope = poly[0]
    intercept = poly[1]

    gamma = float(-slope)
    A = float(10 ** intercept)

    preds = np.polyval(poly, log_c)
    ss_tot = np.sum((log_l - np.mean(log_l)) ** 2)
    ss_res = np.sum((log_l - preds) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-9))) if ss_tot > 0 else 1.0

    return {
        "scaling_exponent_gamma": gamma,
        "coefficient_A": A,
        "r2_fit": r2
    }


def run_full_pretraining_scaling_suite(
    save_path: str = "results/empirical_scaling_laws.json"
) -> Dict:
    """
    Runs multi-scale pretraining for Kernel-DeltaNet and Transformer across scale points
    on authentic FineWeb-Edu tokens.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = FineWebEduDataLoader(device=device)

    # Scale Point Definitions:
    # Scale-15M: d=256, L=6, D_RKHS=1024
    # Scale-35M: d=384, L=8, D_RKHS=1536
    # Scale-70M: d=512, L=10, D_RKHS=2048
    # Scale-125M: d=768, L=12, D_RKHS=3072

    scale_configs = [
        {"name": "Scale-15M", "d": 256, "layers": 6, "d_rkhs": 1024, "tokens": 40_000},
        {"name": "Scale-35M", "d": 384, "layers": 8, "d_rkhs": 1536, "tokens": 60_000},
        {"name": "Scale-70M", "d": 512, "layers": 10, "d_rkhs": 2048, "tokens": 80_000},
        {"name": "Scale-125M", "d": 768, "layers": 12, "d_rkhs": 3072, "tokens": 100_000},
    ]

    kernel_ttt_points = []
    transformer_points = []

    print("\n" + "#" * 80)
    print("PART 1: KERNEL-DELTANET PRE-TRAINING SCALING RUNS")
    print("#" * 80)
    for cfg in scale_configs:
        pt = train_scaling_checkpoint(
            model_family="kernel_ttt",
            scale_name=cfg["name"],
            d_model=cfg["d"],
            num_layers=cfg["layers"],
            d_rkhs=cfg["d_rkhs"],
            target_tokens=cfg["tokens"],
            loader=loader,
            device=device
        )
        kernel_ttt_points.append(pt)

    print("\n" + "#" * 80)
    print("PART 2: STANDARD TRANSFORMER PRE-TRAINING BASELINE RUNS")
    print("#" * 80)
    for cfg in scale_configs:
        pt = train_scaling_checkpoint(
            model_family="transformer",
            scale_name=cfg["name"],
            d_model=cfg["d"],
            num_layers=cfg["layers"],
            d_rkhs=None,
            target_tokens=cfg["tokens"],
            loader=loader,
            device=device
        )
        transformer_points.append(pt)

    k_fit = fit_scaling_power_law(kernel_ttt_points)
    t_fit = fit_scaling_power_law(transformer_points)

    print("\n" + "=" * 80)
    print("CHINCHILLA EMPIRICAL SCALING LAWS SUMMARY:")
    print(f"  Kernel-DeltaNet:   gamma = {k_fit['scaling_exponent_gamma']:.4f} (R^2 = {k_fit['r2_fit']:.4f})")
    print(f"  Transformer:  gamma = {t_fit['scaling_exponent_gamma']:.4f} (R^2 = {t_fit['r2_fit']:.4f})")
    print(f"  Slope Ratio (Kernel-DeltaNet / Transformer): {k_fit['scaling_exponent_gamma'] / (t_fit['scaling_exponent_gamma'] + 1e-9):.2f}x")
    print("=" * 80)

    results = {
        "kernel_ttt_points": kernel_ttt_points,
        "transformer_points": transformer_points,
        "kernel_ttt_fit": k_fit,
        "transformer_fit": t_fit,
        "summary": {
            "kernel_ttt_gamma": k_fit["scaling_exponent_gamma"],
            "transformer_gamma": t_fit["scaling_exponent_gamma"],
            "scaling_advantage": bool(k_fit["scaling_exponent_gamma"] >= t_fit["scaling_exponent_gamma"])
        }
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[*] Complete empirical scaling results saved to {save_path}")

    return results


run_scaling_suite = run_full_pretraining_scaling_suite
compute_scaling_point = train_scaling_checkpoint
fit_chinchilla_scaling_law = fit_scaling_power_law

if __name__ == "__main__":
    run_full_pretraining_scaling_suite()


