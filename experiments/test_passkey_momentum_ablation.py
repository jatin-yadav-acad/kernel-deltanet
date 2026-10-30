import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

repo_root = Path(__file__).resolve().parent.parent
parent_root = repo_root.parent
for p in [str(repo_root), str(parent_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn.functional as F
import numpy as np
import time

from experiments.pure_v2_core import PureKernelDeltaNetV2Core

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_passkey_ablation():
    print("=" * 80)
    print("  PASSKEY RETRIEVAL: MOMENTUM VS ZERO-MOMENTUM PERSISTENT HEADS")
    print("=" * 80)

    context_lengths = [1024, 2048, 4096, 8192, 16384]
    depths = [0.10, 0.50, 0.90]
    num_trials = 10
    d_k = 64
    d_v = 64
    dtype = torch.float32

    models = {
        "1. Naive Pure (Decaying l=0.999, mu=0)": PureKernelDeltaNetV2Core(d_k=d_k, d_v=d_v, num_heads=8, D=512, use_mean_deflation=False, num_persistent_heads=0, decay_init=0.999, momentum_mu=0.0).to(DEVICE),
        "2. Strat A+B+C (Defl+D1024+Persist, NO Mom mu=0)": PureKernelDeltaNetV2Core(d_k=d_k, d_v=d_v, num_heads=8, D=1024, use_mean_deflation=True, num_persistent_heads=2, momentum_mu=0.0).to(DEVICE),
        "3. Strat A+B+C+D (Defl+D1024+Persist, WITH Mom mu=0.85)": PureKernelDeltaNetV2Core(d_k=d_k, d_v=d_v, num_heads=8, D=1024, use_mean_deflation=True, num_persistent_heads=2, momentum_mu=0.85).to(DEVICE),
    }

    for T in context_lengths:
        print(f"\n--- Testing Context Window T = {T} Tokens ---")
        for m_name, model in models.items():
            depth_scores = []
            for depth in depths:
                k_seq_list = []
                v_seq_list = []
                v_needle_list = []

                for trial in range(num_trials):
                    g = torch.Generator(device=DEVICE).manual_seed(50000 + trial * 19 + int(depth * 100))
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

                k_batch = torch.stack(k_seq_list, dim=0).unsqueeze(1).repeat(1, 8, 1, 1)
                v_batch = torch.stack(v_seq_list, dim=0).unsqueeze(1).repeat(1, 8, 1, 1)
                v_needle_batch = torch.stack(v_needle_list, dim=0).squeeze(1)

                with torch.no_grad():
                    out, _, _ = model.forward_recurrent(k_batch, v_batch, k_batch, return_last_only=True)

                y_retrieved = out[:, :, 0, :].mean(dim=1)
                sims = F.cosine_similarity(y_retrieved, v_needle_batch, dim=-1)
                correct = (sims > 0.70).float().mean().item()
                depth_scores.append(correct)

            mean_acc = round(float(np.mean(depth_scores)) * 100.0, 1)
            print(f"  {m_name:<55} | Recall: {mean_acc:5.1f}%")

if __name__ == "__main__":
    test_passkey_ablation()
