"""
Comprehensive Empirical Study: Pure Kernel-DeltaNet v2
======================================================
Investigating the transition to a 100% Pure Recurrence Architecture (22:0 / 32:0)
without Softmax anchor layers by evaluating:
  - Strategy A: Spectral Mean Deflation (Theorem 6.4)
  - Strategy B: Capacity Scaling (D=512 -> D=1024)
  - Strategy C: Persistent Register Heads (lambda = 1.0)
  - Strategy D: Second-Order (Momentum) Delta Rule

Benchmarks:
  Part 1: Multi-Step Symbolic Variable Binding over N in [16..2048]
  Part 2: Microscopic Angular Discrimination (Delta_cos in [0.01..0.30])
  Part 3: MQAR Associative Recall Capacity (K in [16..1024])
  Part 4: Extreme-Context Passkey Retrieval to 64k (T in [4k..64k])
"""

import os
import sys
import math
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Force unbuffered output so logs update immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Setup paths
repo_root = Path(__file__).resolve().parent.parent
parent_root = repo_root.parent
for p in [str(repo_root), str(parent_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.pure_v2_core import PureKernelDeltaNetV2Core

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# PART 1: Multi-Step Symbolic Variable Binding Drift over N Steps
# ==============================================================================

def run_part1_symbolic_drift(num_trials: int = 50) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  PART 1: MULTI-STEP SYMBOLIC VARIABLE BINDING DRIFT (N = 16 to 2048)")
    print("=" * 80)

    d_k = 64
    d_v = 64
    dtype = torch.float32
    unroll_steps = [16, 32, 64, 128, 256, 512, 1024, 2048]

    # Model variants to evaluate:
    variants = {
        "1. Naive Pure (D=512, Uncent, Decaying)": {
            "D": 512, "use_mean_deflation": False, "num_persistent_heads": 0, "momentum_mu": 0.0
        },
        "2. Pure + Strategy A (Spectral Deflation)": {
            "D": 512, "use_mean_deflation": True, "num_persistent_heads": 0, "momentum_mu": 0.0
        },
        "3. Pure + Strategy B (Capacity D=1024)": {
            "D": 1024, "use_mean_deflation": False, "num_persistent_heads": 0, "momentum_mu": 0.0
        },
        "4. Pure + Strategy C (Persistent Heads)": {
            "D": 512, "use_mean_deflation": False, "num_persistent_heads": 2, "momentum_mu": 0.0
        },
        "5. Pure + Strategy D (Momentum Delta mu=0.85)": {
            "D": 512, "use_mean_deflation": False, "num_persistent_heads": 0, "momentum_mu": 0.85
        },
        "6. Pure + Strat A+B (Deflation + D=1024)": {
            "D": 1024, "use_mean_deflation": True, "num_persistent_heads": 0, "momentum_mu": 0.0
        },
        "7. Pure + Strat A+B+C (Defl+D1024+Persist)": {
            "D": 1024, "use_mean_deflation": True, "num_persistent_heads": 2, "momentum_mu": 0.0
        },
        "8. Full Pure v2 (Strat A+B+C+D)": {
            "D": 1024, "use_mean_deflation": True, "num_persistent_heads": 2, "momentum_mu": 0.85
        }
    }

    models = {}
    for name, cfg in variants.items():
        torch.manual_seed(42)
        models[name] = PureKernelDeltaNetV2Core(
            d_k=d_k,
            d_v=d_v,
            num_heads=8,
            D=cfg["D"],
            use_mean_deflation=cfg["use_mean_deflation"],
            num_persistent_heads=cfg["num_persistent_heads"],
            momentum_mu=cfg["momentum_mu"],
            device=str(DEVICE),
            dtype=dtype
        ).to(DEVICE)

    part1_results = {"unroll_steps": unroll_steps, "evaluations": {}}

    for N in unroll_steps:
        print(f"\n--- Testing Distractor Length N = {N} ---")
        part1_results["evaluations"][f"N_{N}"] = {}

        # Pre-generate batched test trials
        k_targets = []
        v_targets = []
        target_indices = []

        for trial in range(num_trials):
            g = torch.Generator(device=DEVICE).manual_seed(10000 + trial * 7 + N)
            k_candidates = F.normalize(torch.randn(4, d_k, device=DEVICE, dtype=dtype, generator=g), dim=-1)
            v_candidates = torch.eye(4, d_v, device=DEVICE, dtype=dtype)
            target_idx = trial % 4
            k_target = k_candidates[target_idx:target_idx+1] # (1, d_k)
            v_target = v_candidates[target_idx:target_idx+1] # (1, d_v)

            k_dist = F.normalize(torch.randn(N, d_k, device=DEVICE, dtype=dtype, generator=g), dim=-1)
            v_dist = torch.randn(N, d_v, device=DEVICE, dtype=dtype, generator=g) * 0.1

            k_seq_single = torch.cat([k_target, k_dist, k_target], dim=0) # (N+2, d_k)
            v_seq_single = torch.cat([v_target, v_dist, torch.zeros(1, d_v, device=DEVICE, dtype=dtype)], dim=0) # (N+2, d_v)

            k_targets.append(k_seq_single)
            v_targets.append(v_seq_single)
            target_indices.append(target_idx)

        k_batch = torch.stack(k_targets, dim=0).unsqueeze(1).repeat(1, 8, 1, 1) # (B, 8, N+2, d_k)
        v_batch = torch.stack(v_targets, dim=0).unsqueeze(1).repeat(1, 8, 1, 1) # (B, 8, N+2, d_v)
        targets_tensor = torch.tensor(target_indices, device=DEVICE, dtype=torch.long)
        v_one_hot = F.one_hot(targets_tensor, num_classes=4).float() # (B, 4)

        # Evaluate each recurrent variant
        for m_name, model in models.items():
            with torch.no_grad():
                out, S_final, aux = model.forward_recurrent(k_batch, v_batch, k_batch, return_last_only=True)

            y_query = out[:, :, 0, :4].mean(dim=1) # (B, 4)
            preds = torch.argmax(y_query, dim=-1) # (B,)

            acc = round((preds == targets_tensor).float().mean().item() * 100.0, 1)
            mean_mse = round(torch.mean(torch.sum((y_query - v_one_hot)**2, dim=-1)).item(), 4)

            # Target logit vs best competitor logit
            target_logits = torch.gather(y_query, 1, targets_tensor.unsqueeze(1)).squeeze(1) # (B,)
            mask = torch.ones_like(y_query, dtype=torch.bool)
            mask.scatter_(1, targets_tensor.unsqueeze(1), False)
            competitor_max = y_query[mask].view(num_trials, 3).max(dim=-1).values # (B,)
            mean_margin = round(torch.mean(target_logits - competitor_max).item(), 4)
            mean_drift = round(aux["state_norms"][-1], 2)

            part1_results["evaluations"][f"N_{N}"][m_name] = {
                "accuracy_pct": acc,
                "mse": mean_mse,
                "margin": mean_margin,
                "state_norm": mean_drift
            }
            print(f"  {m_name:<45} | Acc: {acc:5.1f}% | Margin: {mean_margin:+6.3f} | MSE: {mean_mse:.4f}")

        # Baseline: Hybrid 20:2 (Softmax boundary anchor)
        q_last = k_batch[:, 0, -1:, :] # (B, 1, d_k)
        k_all = k_batch[:, 0, :-1, :]  # (B, N+1, d_k)
        v_all = v_batch[:, 0, :-1, :4] # (B, N+1, 4)
        attn_scores = torch.bmm(q_last, k_all.transpose(-1, -2)) / math.sqrt(d_k) # (B, 1, N+1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        y_soft = torch.bmm(attn_weights, v_all).squeeze(1) # (B, 4)

        preds_hyb = torch.argmax(y_soft, dim=-1)
        acc_hyb = round((preds_hyb == targets_tensor).float().mean().item() * 100.0, 1)
        mean_mse_hyb = round(torch.mean(torch.sum((y_soft - v_one_hot)**2, dim=-1)).item(), 4)

        part1_results["evaluations"][f"N_{N}"]["9. Hybrid 20:2 (Softmax Anchor)"] = {
            "accuracy_pct": acc_hyb, "mse": mean_mse_hyb, "margin": 0.0, "state_norm": 0.0
        }
        print(f"  {'9. Hybrid 20:2 (Softmax Anchor)':<45} | Acc: {acc_hyb:5.1f}% | MSE: {mean_mse_hyb:.4f}")

    return part1_results


# ==============================================================================
# PART 2: Microscopic Angular Discrimination Stress Test (Delta_cos)
# ==============================================================================

def run_part2_angular_discrimination(num_trials: int = 50) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  PART 2: MICROSCOPIC ANGULAR DISCRIMINATION STRESS TEST (Delta_cos -> 0.01)")
    print("=" * 80)

    d_k = 64
    d_v = 64
    dtype = torch.float32
    N_distractors = 64
    delta_cos_levels = [0.30, 0.15, 0.08, 0.04, 0.02, 0.01]

    models_to_test = {
        "Naive Pure (D=512, Uncent)": PureKernelDeltaNetV2Core(d_k=d_k, d_v=d_v, num_heads=8, D=512, use_mean_deflation=False, num_persistent_heads=0).to(DEVICE),
        "Pure + Deflation (D=512)": PureKernelDeltaNetV2Core(d_k=d_k, d_v=d_v, num_heads=8, D=512, use_mean_deflation=True, num_persistent_heads=0).to(DEVICE),
        "Pure + Defl + D=1024": PureKernelDeltaNetV2Core(d_k=d_k, d_v=d_v, num_heads=8, D=1024, use_mean_deflation=True, num_persistent_heads=0).to(DEVICE),
        "Full Pure v2 (A+B+C+D)": PureKernelDeltaNetV2Core(d_k=d_k, d_v=d_v, num_heads=8, D=1024, use_mean_deflation=True, num_persistent_heads=2, momentum_mu=0.85).to(DEVICE)
    }

    part2_results = {"delta_cos_levels": delta_cos_levels, "evaluations": {}}

    for delta_cos in delta_cos_levels:
        print(f"\n--- Testing Angular Separation Delta_cos = {delta_cos:.3f} ---")
        part2_results["evaluations"][f"delta_{delta_cos}"] = {}

        k_seq_list = []
        v_seq_list = []

        for trial in range(num_trials):
            g = torch.Generator(device=DEVICE).manual_seed(20000 + trial * 13)
            u = F.normalize(torch.randn(1, d_k, device=DEVICE, dtype=dtype, generator=g), dim=-1)
            u_ortho = F.normalize(torch.randn(1, d_k, device=DEVICE, dtype=dtype, generator=g) - torch.sum(u * u, dim=-1, keepdim=True) * u, dim=-1)

            k_target = u
            v_target = torch.tensor([[1.0, 0.0, 0.0, 0.0] + [0.0]*(d_v-4)], device=DEVICE, dtype=dtype)

            cos_theta = 1.0 - delta_cos
            sin_theta = math.sqrt(max(0.0, 1.0 - cos_theta**2))
            k_competitor = cos_theta * u + sin_theta * u_ortho
            v_competitor = torch.tensor([[0.0, 1.0, 0.0, 0.0] + [0.0]*(d_v-4)], device=DEVICE, dtype=dtype)

            k_dist = F.normalize(torch.randn(N_distractors, d_k, device=DEVICE, dtype=dtype, generator=g), dim=-1)
            v_dist = torch.randn(N_distractors, d_v, device=DEVICE, dtype=dtype, generator=g) * 0.05

            k_seq = torch.cat([k_target, k_dist[:N_distractors//2], k_competitor, k_dist[N_distractors//2:], k_target], dim=0)
            v_seq = torch.cat([v_target, v_dist[:N_distractors//2], v_competitor, v_dist[N_distractors//2:], torch.zeros(1, d_v, device=DEVICE, dtype=dtype)], dim=0)

            k_seq_list.append(k_seq)
            v_seq_list.append(v_seq)

        k_batch = torch.stack(k_seq_list, dim=0).unsqueeze(1).repeat(1, 8, 1, 1) # (B, 8, T, d_k)
        v_batch = torch.stack(v_seq_list, dim=0).unsqueeze(1).repeat(1, 8, 1, 1) # (B, 8, T, d_v)

        for m_name, model in models_to_test.items():
            with torch.no_grad():
                out, _, _ = model.forward_recurrent(k_batch, v_batch, k_batch, return_last_only=True)

            y_query = out[:, :, 0, :2].mean(dim=1) # (B, 2)
            correct = (y_query[:, 0] > y_query[:, 1]).float().mean().item()
            acc = round(correct * 100.0, 1)
            part2_results["evaluations"][f"delta_{delta_cos}"][m_name] = acc
            print(f"  {m_name:<35} | Acc: {acc:5.1f}%")

    return part2_results


# ==============================================================================
# PART 3: MQAR Associative Recall Capacity Scaling (K = 16 to 1024)
# ==============================================================================

def run_part3_mqar_capacity(num_trials: int = 30) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  PART 3: MQAR ASSOCIATIVE RECALL CAPACITY BENCHMARK (K = 16 to 1024)")
    print("=" * 80)

    class InlineMQARGenerator:
        def __init__(self, d_model=64, device="cuda"):
            self.d_model = d_model
            self.d_sub = d_model // 2
            self.device = device
        def generate_batch(self, batch_size=1, seq_len=4096, num_kv_pairs=64, target_turn="earliest", seed=42):
            g = torch.Generator(device=self.device).manual_seed(seed)
            keys = F.normalize(torch.randn(batch_size, num_kv_pairs, self.d_sub, device=self.device, generator=g), dim=-1)
            values = F.normalize(torch.randn(batch_size, num_kv_pairs, self.d_sub, device=self.device, generator=g), dim=-1)
            target_idx = 0 if target_turn == "earliest" else (num_kv_pairs // 2)
            seq = torch.zeros(batch_size, seq_len, self.d_model, device=self.device)
            indices = np.linspace(0, seq_len - 10, num_kv_pairs, dtype=int)
            for i, idx in enumerate(indices):
                seq[:, idx, :self.d_sub] = keys[:, i]
                seq[:, idx, self.d_sub:] = values[:, i]
            seq[:, -1, :self.d_sub] = keys[:, target_idx]
            return {
                "sequences": seq,
                "target_values": values[:, target_idx], # (B, d_sub)
                "all_values": values # (B, num_kv_pairs, d_sub)
            }

    generator = InlineMQARGenerator(d_model=64, device=str(DEVICE))
    k_scales = [16, 32, 64, 128, 256, 512, 1024]
    T_seq = 4096

    models_to_test = {
        "Naive Pure (D=512)": PureKernelDeltaNetV2Core(d_k=32, d_v=32, num_heads=8, D=512, use_mean_deflation=False, num_persistent_heads=0).to(DEVICE),
        "Pure v2 (D=1024, Defl, Persist, Mom)": PureKernelDeltaNetV2Core(d_k=32, d_v=32, num_heads=8, D=1024, use_mean_deflation=True, num_persistent_heads=2, momentum_mu=0.85).to(DEVICE)
    }

    part3_results = {"K_scales": k_scales, "evaluations": {}}

    for K in k_scales:
        print(f"\n--- Testing MQAR Capacity K = {K} pairs ---")
        part3_results["evaluations"][f"K_{K}"] = {}

        batch = generator.generate_batch(batch_size=num_trials, seq_len=T_seq, num_kv_pairs=K, seed=30000 + K)
        seq = batch["sequences"] # (B, T, 64)
        target_val = batch["target_values"] # (B, 32)
        all_vals = batch["all_values"] # (B, K, 32)

        q_seq = seq[:, :, :32].unsqueeze(1).repeat(1, 8, 1, 1) # (B, 8, T, 32)
        k_seq = q_seq
        v_seq = seq[:, :, 32:].unsqueeze(1).repeat(1, 8, 1, 1) # (B, 8, T, 32)

        for m_name, model in models_to_test.items():
            with torch.no_grad():
                out, _, _ = model.forward_recurrent(k_seq, v_seq, q_seq, return_last_only=True)

            pred_val = out[:, :, 0, :].mean(dim=1) # (B, 32)
            # Batched cosine similarity with all K candidates: (B, K)
            sims = torch.bmm(all_vals, pred_val.unsqueeze(-1)).squeeze(-1) # (B, K)
            correct = (sims.argmax(dim=-1) == 0).float().mean().item()
            acc = round(correct * 100.0, 1)

            noise = pred_val - target_val
            snr = 10.0 * torch.log10(torch.norm(target_val, dim=-1)**2 / (torch.norm(noise, dim=-1)**2 + 1e-9))
            mean_snr = round(snr.mean().item(), 2)

            part3_results["evaluations"][f"K_{K}"][m_name] = {"acc": acc, "snr": mean_snr}
            print(f"  {m_name:<40} | Recall: {acc:5.1f}% | SNR: {mean_snr:+6.2f} dB")

        # Softmax Attention Oracle
        q_last = seq[:, -1:, :32] # (B, 1, 32)
        k_all = seq[:, :-1, :32]  # (B, T-1, 32)
        v_all = seq[:, :-1, 32:]  # (B, T-1, 32)
        w = F.softmax(torch.bmm(q_last, k_all.transpose(-1, -2)) / math.sqrt(32), dim=-1) # (B, 1, T-1)
        pred_soft = torch.bmm(w, v_all).squeeze(1) # (B, 32)

        sims_soft = torch.bmm(all_vals, pred_soft.unsqueeze(-1)).squeeze(-1) # (B, K)
        correct_soft = (sims_soft.argmax(dim=-1) == 0).float().mean().item()
        acc_soft = round(correct_soft * 100.0, 1)

        noise_soft = pred_soft - target_val
        snr_soft = 10.0 * torch.log10(torch.norm(target_val, dim=-1)**2 / (torch.norm(noise_soft, dim=-1)**2 + 1e-9))
        mean_snr_soft = round(snr_soft.mean().item(), 2)

        part3_results["evaluations"][f"K_{K}"]["Softmax Attention"] = {"acc": acc_soft, "snr": mean_snr_soft}
        print(f"  {'Softmax Attention':<40} | Recall: {acc_soft:5.1f}% | SNR: {mean_snr_soft:+6.2f} dB")

    return part3_results


# ==============================================================================
# PART 4: Extreme-Context Passkey Retrieval up to 64k Tokens
# ==============================================================================

def run_part4_long_context_passkey(num_trials: int = 10) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  PART 4: EXTREME-CONTEXT PASSKEY RETRIEVAL UP TO 64k TOKENS")
    print("=" * 80)

    context_lengths = [4096, 8192, 16384, 32768, 65536]
    depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    d_k = 64
    d_v = 64
    dtype = torch.float32

    models = {
        "Naive Pure (D=512, Decaying lambda=0.999)": PureKernelDeltaNetV2Core(d_k=d_k, d_v=d_v, num_heads=8, D=512, use_mean_deflation=False, num_persistent_heads=0, decay_init=0.999).to(DEVICE),
        "Pure v2 (D=1024, Deflated + Persistent lambda=1.0)": PureKernelDeltaNetV2Core(d_k=d_k, d_v=d_v, num_heads=8, D=1024, use_mean_deflation=True, num_persistent_heads=2, momentum_mu=0.85).to(DEVICE)
    }

    part4_results = {"context_lengths": context_lengths, "evaluations": {}}

    for T in context_lengths:
        print(f"\n--- Testing Context Window T = {T} Tokens ---")
        part4_results["evaluations"][f"T_{T}"] = {}

        for m_name, model in models.items():
            depth_scores = []
            for depth in depths:
                k_seq_list = []
                v_seq_list = []
                v_needle_list = []

                for trial in range(num_trials):
                    g = torch.Generator(device=DEVICE).manual_seed(40000 + trial * 17 + int(depth * 100))
                    k_needle = F.normalize(torch.randn(1, d_k, device=DEVICE, dtype=dtype, generator=g), dim=-1)
                    v_needle = torch.randn(1, d_v, device=DEVICE, dtype=dtype, generator=g)

                    pos = int(T * depth)
                    k_bg = F.normalize(torch.randn(T, d_k, device=DEVICE, dtype=dtype, generator=g), dim=-1)
                    v_bg = torch.randn(T, d_v, device=DEVICE, dtype=dtype, generator=g) * 0.05

                    k_bg[pos:pos+1] = k_needle
                    v_bg[pos:pos+1] = v_needle

                    k_seq = torch.cat([k_bg, k_needle], dim=0) # (T+1, d_k)
                    v_seq = torch.cat([v_bg, torch.zeros(1, d_v, device=DEVICE, dtype=dtype)], dim=0) # (T+1, d_v)

                    k_seq_list.append(k_seq)
                    v_seq_list.append(v_seq)
                    v_needle_list.append(v_needle)

                k_batch = torch.stack(k_seq_list, dim=0).unsqueeze(1).repeat(1, 8, 1, 1) # (B, 8, T+1, d_k)
                v_batch = torch.stack(v_seq_list, dim=0).unsqueeze(1).repeat(1, 8, 1, 1) # (B, 8, T+1, d_v)
                v_needle_batch = torch.stack(v_needle_list, dim=0).squeeze(1) # (B, d_v)

                with torch.no_grad():
                    out, _, _ = model.forward_recurrent(k_batch, v_batch, k_batch, return_last_only=True)

                y_retrieved = out[:, :, 0, :].mean(dim=1) # (B, d_v)
                sims = F.cosine_similarity(y_retrieved, v_needle_batch, dim=-1) # (B,)
                correct = (sims > 0.70).float().mean().item()
                depth_scores.append(correct)

            mean_acc = round(float(np.mean(depth_scores)) * 100.0, 1)
            part4_results["evaluations"][f"T_{T}"][m_name] = mean_acc
            print(f"  {m_name:<50} | Recall: {mean_acc:5.1f}% across depths {depths}")

    return part4_results


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def main():
    start_time = time.time()
    all_results = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Run Part 1: Multi-Step Symbolic Variable Binding
    all_results["part1_symbolic_drift"] = run_part1_symbolic_drift(num_trials=50)

    # Run Part 2: Microscopic Angular Discrimination
    all_results["part2_angular_discrimination"] = run_part2_angular_discrimination(num_trials=50)

    # Run Part 3: MQAR Capacity Saturation
    all_results["part3_mqar_capacity"] = run_part3_mqar_capacity(num_trials=30)

    # Run Part 4: Extreme Context Passkey
    all_results["part4_long_context_passkey"] = run_part4_long_context_passkey(num_trials=10)

    out_file = repo_root / "results" / "pure_kernel_deltanet_v2_study.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    elapsed = round(time.time() - start_time, 2)
    print("\n" + "=" * 80)
    print(f"[COMPLETE] Pure Kernel-DeltaNet v2 Empirical Study finished in {elapsed}s.")
    print(f"Saved results ledger to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
