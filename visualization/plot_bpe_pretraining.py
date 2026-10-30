"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: visualization.plot_bpe_pretraining
============================================================
Generates 300 DPI publication figure for Milestone 1:
Converged pretraining dynamics on FineWeb-Edu BPE tokens comparing
Kernel-DeltaNet vs Standard Transformer across 35M and 70M scales.
"""

import os
import json
import matplotlib.pyplot as plt


def plot_bpe_pretraining_curves(
    json_path: str = "results/bpe_pretraining_convergence.json",
    output_dir: str = "artifacts"
):
    if not os.path.exists(json_path):
        print(f"[!] JSON not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    results = data["results"]

    # Subplot 1: 35M scale
    res_35 = results["35m"]
    k_35_steps = [h["step"] for h in res_35["Kernel-DeltaNet"]["history"]]
    k_35_val = [h["val_loss"] for h in res_35["Kernel-DeltaNet"]["history"]]
    t_35_steps = [h["step"] for h in res_35["Standard Transformer"]["history"]]
    t_35_val = [h["val_loss"] for h in res_35["Standard Transformer"]["history"]]

    ax1.plot(k_35_steps, k_35_val, label="Kernel-DeltaNet (Ours)", color="#1f77b4", linewidth=2.4, marker="o", markersize=6)
    ax1.plot(t_35_steps, t_35_val, label="Standard Transformer", color="#d62728", linewidth=2.4, marker="s", markersize=6)
    ax1.set_title("Scale-35M Validation Loss (FineWeb-Edu BPE)")
    ax1.set_xlabel("Pre-Training Optimization Steps")
    ax1.set_ylabel("Validation Cross-Entropy Loss")
    ax1.legend(frameon=True, facecolor="#f8f9fa")

    # Subplot 2: 70M scale
    res_70 = results["70m"]
    k_70_steps = [h["step"] for h in res_70["Kernel-DeltaNet"]["history"]]
    k_70_val = [h["val_loss"] for h in res_70["Kernel-DeltaNet"]["history"]]
    t_70_steps = [h["step"] for h in res_70["Standard Transformer"]["history"]]
    t_70_val = [h["val_loss"] for h in res_70["Standard Transformer"]["history"]]

    ax2.plot(k_70_steps, k_70_val, label="Kernel-DeltaNet (Ours)", color="#1f77b4", linewidth=2.4, marker="o", markersize=6)
    ax2.plot(t_70_steps, t_70_val, label="Standard Transformer", color="#d62728", linewidth=2.4, marker="s", markersize=6)
    ax2.set_title("Scale-70M Validation Loss (FineWeb-Edu BPE)")
    ax2.set_xlabel("Pre-Training Optimization Steps")
    ax2.set_ylabel("Validation Cross-Entropy Loss")
    ax2.legend(frameon=True, facecolor="#f8f9fa")

    plt.suptitle("Real-Token Pre-Training Convergence on RTX 5070 Ti (50k BPE Vocab)", y=0.99)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fig12_bpe_pretraining_curves.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_file}")


if __name__ == "__main__":
    plot_bpe_pretraining_curves()
