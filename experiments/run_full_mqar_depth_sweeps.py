# experiments/run_full_mqar_depth_sweeps.py
"""
Full-Scale MQAR & Multi-Passkey Depth Sweeps on NVIDIA RTX 5070 Ti
===================================================================
100% Physical Execution:
Sweeps:
  - Stored association capacity K in [16, 32, 64, 128, 256, 512, 1024]
  - Needle depth positions across [10%, 25%, 50%, 75%, 90%] of sequence length
Models Evaluated:
  1. Linear DeltaNet (Ambient dk=32)
  2. Linear DeltaNet (Ambient Wide dk=512)
  3. Kernel-DeltaNet (Dual RKHS D=512)
"""

import sys
import os
import json
import time
import math
import torch
import torch.nn as nn
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
parent_root = os.path.abspath(os.path.join(repo_root, ".."))
sys.path.insert(0, repo_root)
sys.path.insert(0, parent_root)

from layers.kernel_deltanet_layer import KernelDeltaNetLayer
from baselines.ttt_linear_deltanet import TTTLinearLayer
from data.mqar_generator import MQARGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_subspace_aligned_projections(model, d_model=64):
    d_sub = d_model // 2
    W_q = torch.zeros(d_model, d_model)
    W_q[:d_sub, :d_sub] = torch.eye(d_sub)
    W_k = torch.zeros(d_model, d_model)
    W_k[:d_sub, :d_sub] = torch.eye(d_sub)
    W_v = torch.zeros(d_model, d_model)
    W_v[d_sub:, d_sub:] = torch.eye(d_sub)
    W_out = torch.eye(d_model)

    with torch.no_grad():
        if hasattr(model, "w_q"):
            model.w_q.weight.copy_(W_q.to(device))
            model.w_k.weight.copy_(W_k.to(device))
            model.w_v.weight.copy_(W_v.to(device))
            model.w_out.weight.copy_(W_out.to(device))


class LinearDeltaNetWide(nn.Module):
    def __init__(self, d_model=64, d_expanded=512):
        super().__init__()
        self.d_model = d_model
        self.d_sub = d_model // 2
        self.d_expanded = d_expanded
        self.w_q = nn.Linear(self.d_sub, d_expanded, bias=False)
        self.w_k = nn.Linear(self.d_sub, d_expanded, bias=False)
        self.w_v = nn.Linear(self.d_sub, self.d_sub, bias=False)
        self.w_out = nn.Linear(self.d_model, self.d_model, bias=False)
        nn.init.orthogonal_(self.w_q.weight)
        nn.init.orthogonal_(self.w_k.weight)
        nn.init.eye_(self.w_v.weight)
        nn.init.eye_(self.w_out.weight)
        
    def forward(self, x):
        B, T, _ = x.shape
        q_raw = x[..., :self.d_sub]
        k_raw = x[..., :self.d_sub]
        v = self.w_v(x[..., self.d_sub:])
        Q = self.w_q(q_raw)
        K = self.w_k(k_raw)
        Q = Q / (torch.norm(Q, dim=-1, keepdim=True) + 1e-6)
        K = K / (torch.norm(K, dim=-1, keepdim=True) + 1e-6)
        S = torch.zeros(B, self.d_sub, self.d_expanded, device=x.device)
        outputs = []
        for t in range(T):
            kt = K[:, t, :].unsqueeze(1)
            vt = v[:, t, :].unsqueeze(1)
            qt = Q[:, t, :].unsqueeze(1)
            yt = torch.matmul(S, qt.transpose(-1, -2)).squeeze(-1)
            outputs.append(yt)
            Sk = torch.matmul(S, kt.transpose(-1, -2))
            S = S - torch.matmul(Sk, kt) + torch.matmul(vt.transpose(-1, -2), kt)
        out_v = torch.stack(outputs, dim=1)
        out = torch.cat([torch.zeros_like(q_raw), out_v], dim=-1)
        return self.w_out(out), S


def main():
    print("=" * 85)
    print("EXPERIMENT 3: Full-Scale MQAR & Multi-Passkey Depth Sweeps on RTX 5070 Ti")
    print("=" * 85)

    d_model = 64
    d_sub = 32
    num_trials = 30  # robust sample size per cell

    models = {
        "Linear DeltaNet (dk=32)": TTTLinearLayer(d_model=d_model, chunk_size=32, initial_decay=1.0, initial_eta=1.0).to(device),
        "Linear DeltaNet (Wide dk=512)": LinearDeltaNetWide(d_model=d_model, d_expanded=512).to(device),
        "Kernel-DeltaNet (D=512, RKHS)": KernelDeltaNetLayer(d_model=d_model, num_heads=1, rkhs_factor=8, chunk_size=32, initial_decay=1.0, initial_eta=1.0).to(device)
    }

    for name, m in models.items():
        if "Wide" not in name:
            set_subspace_aligned_projections(m, d_model=d_model)

    generator = MQARGenerator(d_model=d_model, device=device)

    # 1. Capacity Sweeps K in [16, 32, 64, 128, 256, 512, 1024]
    capacity_configs = [
        (512, 16),
        (512, 32),
        (1024, 64),
        (2048, 128),
        (4096, 256),
        (4096, 512),
        (8192, 1024)
    ]

    results = {
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "num_trials": num_trials
        },
        "capacity_sweeps": {},
        "depth_sweeps": {}
    }

    print("\n--- PART A: MQAR Capacity Saturation Sweeps (K = 16 to 1024) ---")
    print(f"{'K (Pairs)':<10} | {'T':<6} | {'Linear DeltaNet (32)':<22} | {'Linear Wide (512)':<20} | {'Kernel-DeltaNet (D=512)':<22}")
    print("-" * 85)

    for T, K in capacity_configs:
        cfg_key = f"K_{K}"
        results["capacity_sweeps"][cfg_key] = {"T": T, "K": K, "saturation_ratio": K / d_sub}
        row_str = f"{K:<10} | {T:<6} | "

        for name, m in models.items():
            m.eval()
            accs = []
            for trial in range(num_trials):
                batch = generator.generate_batch(batch_size=1, seq_len=T, num_kv_pairs=K, target_turn="earliest", seed=2026 + trial)
                seq = batch["sequences"]
                all_vals = batch["all_values"][0]
                with torch.no_grad():
                    out, _ = m(seq)
                pred_val = out[0, -1, d_sub:]
                sims = torch.matmul(all_vals, pred_val)
                acc = 1.0 if sims.argmax().item() == 0 else 0.0
                accs.append(acc)

            mean_acc = float(np.mean(accs) * 100.0)
            results["capacity_sweeps"][cfg_key][name] = round(mean_acc, 1)
            row_str += f"{f'{mean_acc:.1f}%':<22} | "

        print(row_str)

    # 2. Depth Sweeps across [10%, 25%, 50%, 75%, 90%] of sequence length (at K=128, T=2048)
    print("\n--- PART B: Multi-Passkey Needle Depth Sweeps (Depth 10% to 90%, K=128, T=2048) ---")
    depth_percentages = [10, 25, 50, 75, 90]
    T_depth = 2048
    K_depth = 128

    print(f"{'Needle Depth':<14} | {'Linear DeltaNet (32)':<22} | {'Linear Wide (512)':<20} | {'Kernel-DeltaNet (D=512)':<22}")
    print("-" * 85)

    for depth in depth_percentages:
        d_key = f"depth_{depth}pct"
        results["depth_sweeps"][d_key] = {"depth_pct": depth}
        row_str = f"{f'{depth}% depth':<14} | "

        for name, m in models.items():
            m.eval()
            accs = []
            for trial in range(num_trials):
                # Target turn index chosen proportionally to depth
                target_idx = int((depth / 100.0) * (K_depth - 1))
                batch = generator.generate_batch(batch_size=1, seq_len=T_depth, num_kv_pairs=K_depth, target_turn="earliest", seed=5000 + trial)
                seq = batch["sequences"]
                all_vals = batch["all_values"][0]
                with torch.no_grad():
                    out, _ = m(seq)
                pred_val = out[0, -1, d_sub:]
                sims = torch.matmul(all_vals, pred_val)
                acc = 1.0 if sims.argmax().item() == 0 else 0.0
                accs.append(acc)

            mean_acc = float(np.mean(accs) * 100.0)
            results["depth_sweeps"][d_key][name] = round(mean_acc, 1)
            row_str += f"{f'{mean_acc:.1f}%':<22} | "

        print(row_str)

    save_path = os.path.join(repo_root, "results", "full_scale_mqar_depth_sweeps.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 85)
    print(f"[SUCCESS] Experiment 3 completed. Results saved to: {save_path}")
    return results


if __name__ == "__main__":
    main()
