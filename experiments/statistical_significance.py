r"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: experiments.statistical_significance
==============================================================
Statistical Significance, Hypothesis Testing, and Non-Parametric Bootstrap CIs.
Runs 50 independent trials with distinct random seeds on the NVIDIA RTX 5070 Ti.
Measures empirical associative recall accuracy, cosine similarity, SNR, and step latency.
Computes:
1. Paired Student's t-tests
2. Wilcoxon signed-rank tests
3. Cohen's d effect sizes
4. Non-parametric Bootstrap 95% Confidence Intervals (10,000 resamples)
"""

import os
import json
import math
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from layers.kernel_deltanet_layer import KernelDeltaNetLayer
from baselines.standard_transformer import CausalSoftmaxAttention
from baselines.ttt_linear_deltanet import TTTLinearLayer
from data.mqar_generator import MQARGenerator


class SignificanceTester:
    r"""
    Executes rigorous statistical hypothesis testing between baseline and proposed methods.
    """
    @staticmethod
    def compute_paired_comparison(
        sample_baseline: List[float],
        sample_proposed: List[float],
        name_baseline: str = "Baseline",
        name_proposed: str = "Kernel-DeltaNet"
    ) -> Dict[str, float]:
        r"""
        Computes paired t-test, Wilcoxon signed-rank, Cohen's d, and bootstrap CI on difference.
        """
        a = np.array(sample_baseline, dtype=np.float64)
        b = np.array(sample_proposed, dtype=np.float64)
        diff = b - a
        n = len(diff)

        # 1. Paired Student's t-test
        if np.all(diff == 0):
            t_stat = 0.0
            p_val_t = 1.0
        else:
            t_res = stats.ttest_rel(b, a)
            t_stat = float(t_res.statistic)
            p_val_t = float(t_res.pvalue)

        # 2. Wilcoxon Signed-Rank Test
        if np.all(diff == 0):
            w_stat = 0.0
            p_val_w = 1.0
        else:
            try:
                w_res = stats.wilcoxon(b, a, zero_method="pratt")
                w_stat = float(w_res.statistic)
                p_val_w = float(w_res.pvalue)
            except Exception:
                w_stat = 0.0
                p_val_w = 1.0

        # 3. Cohen's d Effect Size
        diff_mean = float(np.mean(diff))
        diff_std = float(np.std(diff, ddof=1)) if n > 1 else 1e-9
        cohens_d = diff_mean / (diff_std + 1e-9)

        # 4. Bootstrap 95% Confidence Interval (10,000 resamples)
        rng = np.random.default_rng(seed=42)
        boot_means = []
        for _ in range(10000):
            resample = rng.choice(diff, size=n, replace=True)
            boot_means.append(np.mean(resample))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        return {
            "name_baseline": name_baseline,
            "name_proposed": name_proposed,
            "sample_size": n,
            "mean_baseline": float(np.mean(a)),
            "std_baseline": float(np.std(a, ddof=1)) if n > 1 else 0.0,
            "mean_proposed": float(np.mean(b)),
            "std_proposed": float(np.std(b, ddof=1)) if n > 1 else 0.0,
            "mean_difference": diff_mean,
            "t_statistic": t_stat,
            "p_value_student_t": p_val_t,
            "wilcoxon_statistic": w_stat,
            "p_value_wilcoxon": p_val_w,
            "cohens_d": cohens_d,
            "bootstrap_ci_95_lower": ci_lower,
            "bootstrap_ci_95_upper": ci_upper,
            "is_statistically_significant_p001": bool(p_val_t < 0.001 or p_val_w < 0.001)
        }


def run_statistical_significance_suite(
    d_model: int = 64,
    K: int = 64,
    T: int = 1024,
    num_trials: int = 50,
    save_path: str = "results/statistical_significance.json"
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 100)
    print(f"KERNEL-DELTANET: 50-TRIAL STATISTICAL SIGNIFICANCE EXPERIMENT ON {device}")
    print("=" * 100)

    # Initialize models
    d_sub = d_model // 2
    models = {
        "Kernel-DeltaNet": KernelDeltaNetLayer(d_model=d_model, num_heads=1, rkhs_factor=8, chunk_size=32, initial_decay=1.0, initial_eta=1.0).to(device),
        "Linear DeltaNet": TTTLinearLayer(d_model=d_model, chunk_size=32, initial_decay=1.0, initial_eta=1.0).to(device),
        "Softmax Attention": CausalSoftmaxAttention(d_model=d_model, num_heads=1).to(device)
    }

    # Set aligned projection weights
    W_q = torch.zeros(d_model, d_model, device=device)
    W_q[:d_sub, :d_sub] = torch.eye(d_sub, device=device)
    W_k = torch.zeros(d_model, d_model, device=device)
    W_k[:d_sub, :d_sub] = torch.eye(d_sub, device=device)
    W_v = torch.zeros(d_model, d_model, device=device)
    W_v[d_sub:, d_sub:] = torch.eye(d_sub, device=device)
    W_out = torch.eye(d_model, device=device)

    for m in models.values():
        with torch.no_grad():
            if hasattr(m, "w_q"):
                m.w_q.weight.copy_(W_q)
                m.w_k.weight.copy_(W_k)
                m.w_v.weight.copy_(W_v)
                m.w_out.weight.copy_(W_out)
        m.eval()

    gen = MQARGenerator(d_model=d_model, device=device)

    metrics = {
        name: {"acc": [], "cossim": [], "snr": [], "latency_ms": []}
        for name in models.keys()
    }

    print(f"[*] Executing {num_trials} independent trials (K={K}, T={T}, Saturation={K/d_sub:.1f}x)...")
    for trial in range(num_trials):
        seed = 1000 + trial
        batch = gen.generate_batch(batch_size=1, seq_len=T, num_kv_pairs=K, seed=seed)
        seq = batch["sequences"]
        target_val = batch["target_values"][0]
        all_vals = batch["all_values"][0]

        for name, m in models.items():
            t0 = time.perf_counter()
            with torch.no_grad():
                out, _ = m(seq)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            lat = (t1 - t0) * 1000.0

            pred_val = out[0, -1, d_sub:]
            csim = torch.cosine_similarity(pred_val.unsqueeze(0), target_val.unsqueeze(0)).item()
            sims = torch.matmul(all_vals, pred_val)
            acc = 1.0 if sims.argmax().item() == 0 else 0.0

            crosstalk = pred_val - target_val
            snr = 10.0 * np.log10((torch.norm(target_val)**2 / (torch.norm(crosstalk)**2 + 1e-9)).item())

            metrics[name]["acc"].append(acc * 100.0)
            metrics[name]["cossim"].append(csim)
            metrics[name]["snr"].append(snr)
            metrics[name]["latency_ms"].append(lat)

    print("\n[+] 50 Trials Complete. Summary Distributions:")
    for name in models.keys():
        acc_m, acc_s = np.mean(metrics[name]["acc"]), np.std(metrics[name]["acc"], ddof=1)
        cos_m, cos_s = np.mean(metrics[name]["cossim"]), np.std(metrics[name]["cossim"], ddof=1)
        snr_m, snr_s = np.mean(metrics[name]["snr"]), np.std(metrics[name]["snr"], ddof=1)
        lat_m, lat_s = np.mean(metrics[name]["latency_ms"]), np.std(metrics[name]["latency_ms"], ddof=1)
        print(f"  -> {name:<20}: Acc = {acc_m:>5.1f}% (±{acc_s:>4.1f}) | CosSim = {cos_m:>+.3f} (±{cos_s:>.3f}) | SNR = {snr_m:>+5.1f} dB (±{snr_s:>4.1f}) | Lat = {lat_m:>5.2f} ms")

    # Statistical hypothesis tests
    print("\n--- Formal Hypothesis Testing: Kernel-DeltaNet vs Linear DeltaNet ---")
    comp_acc = SignificanceTester.compute_paired_comparison(
        metrics["Linear DeltaNet"]["acc"],
        metrics["Kernel-DeltaNet"]["acc"],
        name_baseline="Linear DeltaNet",
        name_proposed="Kernel-DeltaNet"
    )
    print(f"  Recall Accuracy Diff: +{comp_acc['mean_difference']:.1f}% | t = {comp_acc['t_statistic']:.3f} | p(t) = {comp_acc['p_value_student_t']:.2e} | p(wilcoxon) = {comp_acc['p_value_wilcoxon']:.2e} | Cohen's d = {comp_acc['cohens_d']:.2f}")
    print(f"  95% Bootstrap CI:     [{comp_acc['bootstrap_ci_95_lower']:.2f}%, {comp_acc['bootstrap_ci_95_upper']:.2f}%]")

    comp_snr = SignificanceTester.compute_paired_comparison(
        metrics["Linear DeltaNet"]["snr"],
        metrics["Kernel-DeltaNet"]["snr"],
        name_baseline="Linear DeltaNet",
        name_proposed="Kernel-DeltaNet"
    )
    print(f"  Retrieved SNR Diff:   +{comp_snr['mean_difference']:.2f} dB | t = {comp_snr['t_statistic']:.3f} | p = {comp_snr['p_value_student_t']:.2e} | Cohen's d = {comp_snr['cohens_d']:.2f}")

    comp_cossim = SignificanceTester.compute_paired_comparison(
        metrics["Linear DeltaNet"]["cossim"],
        metrics["Kernel-DeltaNet"]["cossim"],
        name_baseline="Linear DeltaNet",
        name_proposed="Kernel-DeltaNet"
    )

    all_significance = {
        "empirical_metrics": {
            name: {
                "acc_mean": float(np.mean(metrics[name]["acc"])),
                "acc_std": float(np.std(metrics[name]["acc"], ddof=1)),
                "cos_mean": float(np.mean(metrics[name]["cossim"])),
                "cos_std": float(np.std(metrics[name]["cossim"], ddof=1)),
                "snr_mean": float(np.mean(metrics[name]["snr"])),
                "snr_std": float(np.std(metrics[name]["snr"], ddof=1)),
                "latency_mean": float(np.mean(metrics[name]["latency_ms"])),
                "latency_std": float(np.std(metrics[name]["latency_ms"], ddof=1)),
                "raw_acc": metrics[name]["acc"],
                "raw_snr": metrics[name]["snr"],
                "raw_cossim": metrics[name]["cossim"]
            }
            for name in models.keys()
        },
        "hypothesis_tests": {
            "accuracy_vs_linear_deltanet": comp_acc,
            "snr_vs_linear_deltanet": comp_snr,
            "cossim_vs_linear_deltanet": comp_cossim
        }
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_significance, f, indent=2)

    print(f"\n[+] Statistical significance report saved to {save_path}")
    return all_significance


if __name__ == "__main__":
    run_statistical_significance_suite()
