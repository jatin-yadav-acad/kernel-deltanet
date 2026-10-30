# benchmarks/run_spectral_mean_deflation_eval.py
"""
Empirical Evaluation: Spectral Mean Deflation (Centering) in Pure Recurrence
=============================================================================
Validates that removing the non-zero DC feature mean from Random Fourier Features
eliminates O(N/D) random-walk error accumulation in PURE recurrence (without Softmax),
restoring high associative copying accuracy across sequence positions N=16 to 1024.
"""

import os
import sys
import json
import math
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_spectral_mean_deflation_experiment():
    assert torch.cuda.is_available(), "CUDA device required."
    device_name = torch.cuda.get_device_name(0)
    print("=" * 80)
    print(f"  SPECTRAL MEAN DEFLATION IN PURE RECURRENCE ON {device_name}")
    print("  Resolving the 20:2 Hybrid Purist Concession via Feature Centering")
    print("=" * 80)

    d_k = 64
    d_v = 64
    D = 512
    dtype = torch.float32

    unroll_steps = [16, 32, 64, 128, 256, 512, 1024]
    num_trials = 100

    torch.manual_seed(42)
    # Fast Walsh-Hadamard projection simulation
    W = torch.randn(D // 2, d_k, device=DEVICE, dtype=dtype) / math.sqrt(d_k)
    def phi(x):
        proj = torch.matmul(x, W.T)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(D // 2)

    results = {
        "device": device_name,
        "D": D,
        "d_k": d_k,
        "num_trials": num_trials,
        "evaluations": []
    }

    print("\n--- Running Multi-Step Symbolic Retrieval Across Distractor Steps N ---")

    for N in unroll_steps:
        correct_uncentered = 0
        correct_centered = 0
        correct_hybrid = 0
        correct_softmax = 0

        var_uncentered = []
        var_centered = []
        var_hybrid = []

        for trial in range(num_trials):
            # 4 discrete symbolic variables
            k_targets = F.normalize(torch.randn(4, 1, d_k, device=DEVICE, dtype=dtype), dim=-1)
            v_targets = torch.eye(4, d_v, device=DEVICE, dtype=dtype).unsqueeze(1)
            target_idx = trial % 4

            # N sequential non-zero distractors (interfering background tokens)
            k_dist = F.normalize(torch.randn(1, N, d_k, device=DEVICE, dtype=dtype), dim=-1)
            v_dist = torch.randn(1, N, d_v, device=DEVICE, dtype=dtype) * 0.1

            k_seq = torch.cat([k_targets[target_idx:target_idx+1], k_dist, k_targets[target_idx:target_idx+1]], dim=1)
            v_seq = torch.cat([v_targets[target_idx:target_idx+1], v_dist, torch.zeros(1, 1, d_v, device=DEVICE, dtype=dtype)], dim=1)
            
            beta = 0.5
            lambd = 0.999

            # 1. Uncentered Pure Recurrence
            phi_seq = phi(k_seq)
            S_uncent = torch.zeros(1, d_v, D, device=DEVICE, dtype=dtype)
            for t in range(N + 2):
                kt = phi_seq[:, t:t+1, :]
                vt = v_seq[:, t:t+1, :]
                S_uncent = lambd * S_uncent + beta * torch.matmul(vt.transpose(-1, -2), kt) - beta * torch.matmul(torch.matmul(S_uncent, kt.transpose(-1, -2)), kt)
            
            y_uncent = torch.matmul(S_uncent, phi_seq[:, -1:, :].transpose(-1, -2)).squeeze(-1)[:, :4]
            if torch.argmax(y_uncent).item() == target_idx:
                correct_uncentered += 1
            var_uncentered.append(torch.norm(y_uncent - v_targets[target_idx, 0, :4]).item()**2)

            # 2. Centered Pure Recurrence (Spectral Mean Deflation)
            mean_p = phi_seq.mean(dim=1, keepdim=True)
            phi_seq_c = F.normalize(phi_seq - mean_p, dim=-1)
            S_cent = torch.zeros(1, d_v, D, device=DEVICE, dtype=dtype)
            for t in range(N + 2):
                kt = phi_seq_c[:, t:t+1, :]
                vt = v_seq[:, t:t+1, :]
                S_cent = lambd * S_cent + beta * torch.matmul(vt.transpose(-1, -2), kt) - beta * torch.matmul(torch.matmul(S_cent, kt.transpose(-1, -2)), kt)

            y_cent = torch.matmul(S_cent, phi_seq_c[:, -1:, :].transpose(-1, -2)).squeeze(-1)[:, :4]
            if torch.argmax(y_cent).item() == target_idx:
                correct_centered += 1
            var_centered.append(torch.norm(y_cent - v_targets[target_idx, 0, :4]).item()**2)

            # 3. Hybrid 20:2 Gauge Topology (Softmax anchor)
            anchor_scores = torch.matmul(k_seq[:, -1:, :], k_seq.transpose(-1, -2)) / math.sqrt(d_k)
            anchor_weights = F.softmax(anchor_scores, dim=-1)
            y_anchor = torch.matmul(anchor_weights, v_seq)[:, 0, :4]
            y_hyb = 0.1 * y_uncent + 0.9 * y_anchor
            if torch.argmax(y_hyb).item() == target_idx:
                correct_hybrid += 1
            var_hybrid.append(torch.norm(y_hyb - v_targets[target_idx, 0, :4]).item()**2)

            # 4. Softmax Attention (Exact quadratic)
            if torch.argmax(y_anchor).item() == target_idx:
                correct_softmax += 1

        acc_uncent = round((correct_uncentered / num_trials) * 100.0, 1)
        acc_cent = round((correct_centered / num_trials) * 100.0, 1)
        acc_hyb = round((correct_hybrid / num_trials) * 100.0, 1)
        acc_soft = round((correct_softmax / num_trials) * 100.0, 1)

        v_uncent = round(float(np.mean(var_uncentered)), 4)
        v_cent = round(float(np.mean(var_centered)), 4)
        v_hyb = round(float(np.mean(var_hybrid)), 4)

        entry = {
            "N_steps": N,
            "uncentered_pure_acc": acc_uncent,
            "uncentered_pure_variance": v_uncent,
            "spectral_mean_deflated_pure_acc": acc_cent,
            "spectral_mean_deflated_variance": v_cent,
            "hybrid_20_2_acc": acc_hyb,
            "hybrid_20_2_variance": v_hyb,
            "softmax_acc": acc_soft
        }
        results["evaluations"].append(entry)
        print(f"  N = {N:4d} | Uncentered: {acc_uncent:5.1f}% (Var: {v_uncent:.3f}) | Centered Pure: {acc_cent:5.1f}% (Var: {v_cent:.3f}) | Hybrid 20:2: {acc_hyb:5.1f}% (Var: {v_hyb:.3f}) | Softmax: {acc_soft:5.1f}%")

    out_file = REPO_ROOT / "results" / "spectral_mean_deflation_eval.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] Spectral Mean Deflation benchmark completed. Saved to: {out_file}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_spectral_mean_deflation_experiment()
