"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: visualization.plot_downstream_nlp
===========================================================
Generates 300 DPI publication figure for Milestone 2:
Zero-shot downstream NLP benchmark comparison (HellaSwag, PIQA, ARC-Easy)
between Kernel-DeltaNet and Standard Transformer across 35M and 70M scales.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_downstream_nlp_figure(
    json_path: str = "results/downstream_nlp_benchmarks.json",
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
    tasks = ["HellaSwag", "PIQA", "ARC-Easy"]
    task_keys = ["hellaswag_acc", "piqa_acc", "arc_easy_acc"]

    results = data["results"]

    # Subplot 1: Scale 35M
    k_35 = [results["Kernel-DeltaNet_35m"][k] for k in task_keys]
    t_35 = [results["Standard Transformer_35m"][k] for k in task_keys]

    x = np.arange(len(tasks))
    width = 0.32

    ax1.bar(x - width/2, k_35, width, label="Kernel-DeltaNet (Ours)", color="#1f77b4", alpha=0.9, edgecolor="none")
    ax1.bar(x + width/2, t_35, width, label="Standard Transformer", color="#d62728", alpha=0.9, edgecolor="none")

    for i in range(len(tasks)):
        ax1.text(x[i] - width/2, k_35[i] + 1.0, f"{k_35[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#1f77b4")
        ax1.text(x[i] + width/2, t_35[i] + 1.0, f"{t_35[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#d62728")

    ax1.set_title("Scale-35M Configuration (d=384, L=8, D=1536)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks)
    ax1.set_ylabel("Zero-Shot Accuracy (%)")
    ax1.set_ylim(0, 65)
    ax1.legend(frameon=True, facecolor="#f8f9fa", loc="upper left")

    # Subplot 2: Scale 70M
    k_70 = [results["Kernel-DeltaNet_70m"][k] for k in task_keys]
    t_70 = [results["Standard Transformer_70m"][k] for k in task_keys]

    ax2.bar(x - width/2, k_70, width, label="Kernel-DeltaNet (Ours)", color="#1f77b4", alpha=0.9, edgecolor="none")
    ax2.bar(x + width/2, t_70, width, label="Standard Transformer", color="#d62728", alpha=0.9, edgecolor="none")

    for i in range(len(tasks)):
        ax2.text(x[i] - width/2, k_70[i] + 1.0, f"{k_70[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#1f77b4")
        ax2.text(x[i] + width/2, t_70[i] + 1.0, f"{t_70[i]:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#d62728")

    ax2.set_title("Scale-70M Configuration (d=512, L=10, D=2048)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks)
    ax2.set_ylabel("Zero-Shot Accuracy (%)")
    ax2.set_ylim(0, 65)
    ax2.legend(frameon=True, facecolor="#f8f9fa", loc="upper left")

    plt.suptitle("Standard Zero-Shot Downstream NLP Benchmarks on RTX 5070 Ti", y=0.99)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fig11_downstream_nlp_comparison.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_file}")


if __name__ == "__main__":
    plot_downstream_nlp_figure()
