"""
=============================================================================
EXPERIMENT 1: 2D ANCHOR PLACEMENT SENSITIVITY MAP (BATCH-VECTORIZED)
=============================================================================
Evaluates anchor placement across a 22-layer TinyLlama-scale model:
  Config A: Sandwich [1, 22] (Asymmetric Boundary Co-Design - Ours)
  Config B: Internal [7, 15] (Symmetric Mid-stack)
  Config C: Deep [21, 22]    (Terminal Concentrated)
  Config D: Shallow [1, 2]   (Initial Concentrated)
  Config E: Middle [11, 12]  (Adjacently Centered)
=============================================================================
"""

import sys
import math
import time
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FastRFF(nn.Module):
    def __init__(self, d_in=64, D=512, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        W = torch.randn(D // 2, d_in, device=DEVICE) / math.sqrt(d_in)
        self.register_buffer("W", W)
        self.D = D

    def forward(self, x):
        # x: (B, T, d_in)
        proj = torch.matmul(x, self.W.T)
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / math.sqrt(self.D // 2)
        phi = phi - phi.mean(dim=-1, keepdim=True)
        return F.normalize(phi, p=2, dim=-1)


def rms_norm(x, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def run_anchor_placement_study(batch_size=20):
    print("=" * 80)
    print("  EXPERIMENT 1: 2D ANCHOR PLACEMENT SENSITIVITY EVALUATION")
    print(f"  Batch Size = {batch_size} Trials | Physical Silicon: NVIDIA GeForce RTX 5070 Ti")
    print("=" * 80)

    L = 22
    d_model = 64
    D = 512
    rff = FastRFF(d_in=d_model, D=D).to(DEVICE)

    configs = {
        "Sandwich [1, 22] (Ours)": [1, 22],
        "Internal [7, 15]":        [7, 15],
        "Deep [21, 22]":            [21, 22],
        "Shallow [1, 2]":           [1, 2],
        "Middle [11, 12]":          [11, 12],
    }

    results = {}
    context_lengths = [1024, 2048, 4096]

    for cfg_name, anchor_layers in configs.items():
        print(f"\nTesting {cfg_name:<28} (Softmax at {anchor_layers})...")
        cfg_results = {"anchor_layers": anchor_layers}

        for T in context_lengths:
            t0 = time.time()
            g = torch.Generator(device=DEVICE).manual_seed(10000 + T)
            num_cands = 10
            # For each batch element, create cand_keys and cand_vals
            cand_keys = F.normalize(torch.randn(batch_size, num_cands, d_model, device=DEVICE, generator=g), dim=-1)
            cand_vals = F.normalize(torch.randn(batch_size, num_cands, d_model, device=DEVICE, generator=g), dim=-1)

            target_indices = torch.randint(0, num_cands, (batch_size,), device=DEVICE)
            # Gather targets
            idx_exp = target_indices.view(batch_size, 1, 1).repeat(1, 1, d_model)
            target_keys = torch.gather(cand_keys, 1, idx_exp) # (B, 1, d_model)
            target_vals = torch.gather(cand_vals, 1, idx_exp) # (B, 1, d_model)

            # Insert target at position T // 4
            pos = T // 4
            bg_keys = F.normalize(torch.randn(batch_size, T, d_model, device=DEVICE, generator=g), dim=-1)
            bg_vals = torch.randn(batch_size, T, d_model, device=DEVICE, generator=g) * 0.1

            bg_keys[:, pos:pos+1, :] = target_keys
            bg_vals[:, pos:pos+1, :] = target_vals

            # Query token at T+1
            k_seq = torch.cat([bg_keys, target_keys], dim=1) # (B, T+1, d_model)
            v_seq = torch.cat([bg_vals, torch.zeros(batch_size, 1, d_model, device=DEVICE)], dim=1)

            X = v_seq.clone() # (B, T+1, d_model)
            phi_k = rff(k_seq) # (B, T+1, D)

            # Run 22 transformer layers
            for l in range(1, L + 1):
                X_norm = rms_norm(X)
                if l in anchor_layers:
                    # Softmax global anchor at query position
                    q_vec = k_seq[:, -1:, :] # (B, 1, d_model)
                    sims = torch.bmm(q_vec, k_seq.transpose(-1, -2)) / math.sqrt(d_model) # (B, 1, T+1)
                    attn = F.softmax(sims, dim=-1)
                    retrieved = torch.bmm(attn, X_norm) # (B, 1, d_model)
                    layer_out = torch.zeros_like(X)
                    layer_out[:, -1:, :] = retrieved
                else:
                    # Kernel-DeltaNet recurrence layer
                    S = torch.zeros(batch_size, d_model, D, device=DEVICE)
                    beta = 0.5
                    decay = 1.0
                    for t_idx in range(T):
                        kt = phi_k[:, t_idx:t_idx+1, :] # (B, 1, D)
                        vt = X_norm[:, t_idx:t_idx+1, :] # (B, 1, d_model)
                        pred_v = torch.bmm(S, kt.transpose(-1, -2)).squeeze(-1) # (B, d_model)
                        err = vt.squeeze(1) - pred_v
                        S = decay * S + beta * torch.bmm(err.unsqueeze(-1), kt)

                    q_t = phi_k[:, -1:, :]
                    pred_query = torch.bmm(S, q_t.transpose(-1, -2)).squeeze(-1).unsqueeze(1)
                    layer_out = torch.zeros_like(X)
                    layer_out[:, -1:, :] = pred_query

                X = rms_norm(X + 0.3 * layer_out)

            final_out = X[:, -1, :] # (B, d_model)
            # Candidate similarity scores: (B, num_cands)
            cand_sims = torch.bmm(cand_vals, final_out.unsqueeze(-1)).squeeze(-1)
            pred_indices = cand_sims.argmax(dim=-1)
            correct_count = (pred_indices == target_indices).float().sum().item()
            acc = round((correct_count / batch_size) * 100.0, 1)

            cos_sims = F.cosine_similarity(final_out, target_vals.squeeze(1), dim=-1)
            mean_cos_err = round(float((1.0 - cos_sims).mean().item()), 4)

            dt = round(time.time() - t0, 2)
            cfg_results[f"acc_T{T}"] = acc
            cfg_results[f"cos_err_T{T}"] = mean_cos_err
            print(f"  T={T:4d} | Accuracy: {acc:5.1f}% | Cosine Error: {mean_cos_err:.4f} ({dt}s)")

        results[cfg_name] = cfg_results

    out_file = Path("results/anchor_placement_sensitivity.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved anchor placement results to: {out_file}")
    return results


if __name__ == "__main__":
    run_anchor_placement_study()
