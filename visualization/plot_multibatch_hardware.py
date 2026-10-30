"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: visualization.plot_multibatch_hardware
================================================================
Generates publication-ready 300 DPI grouped bar chart for Milestone 3:
Multi-batch throughput across B in [1, 4, 8, 16] and T in [1024, 16384] on RTX 5070 Ti.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_multibatch_throughput_chart(
    json_path: str = "results/multibatch_hardware_benchmark.json",
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
        'legend.fontsize': 9.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    batch_sizes = [1, 4, 8, 16]
    seq_lens = [1024, 2048, 4096, 8192, 16384]

    models = [
        ("Kernel-DeltaNet (Ours, C=128)", "#1f77b4"),
        ("FlashAttention-2 (PyTorch SDPA)", "#d62728"),
        ("Chunk-Parallel DeltaNet", "#2ca02c"),
        ("Mamba-2 / SSD", "#ff7f0e")
    ]

    for idx, b in enumerate(batch_sizes):
        ax = axes[idx]
        b_key = f"B_{b}"
        b_data = data["matrix"].get(b_key, {})

        x = np.arange(len(seq_lens))
        width = 0.20

        for m_idx, (model_name, color) in enumerate(models):
            throughputs = []
            for t in seq_lens:
                t_key = f"T_{t}"
                entry = b_data.get(t_key, {}).get(model_name)
                if entry and entry["status"] == "SUCCESS":
                    throughputs.append(entry["throughput_tok_per_sec"] / 1e6) # Millions
                else:
                    throughputs.append(0.0)

            ax.bar(x + (m_idx - 1.5) * width, throughputs, width, label=model_name, color=color, alpha=0.9, edgecolor="none")

        ax.set_title(f"Production Batch Size B = {b}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t//1024}k" if t >= 1024 else str(t) for t in seq_lens])
        ax.set_xlabel("Sequence Length Horizon (T)")
        ax.set_ylabel("Throughput (Million Tokens / Sec)")
        ax.set_ylim(0, 22)
        if idx == 0:
            ax.legend(frameon=True, facecolor="#f8f9fa", loc="upper right")

    plt.suptitle("Multi-Batch Hardware Throughput Scaling on NVIDIA GeForce RTX 5070 Ti (BF16 AMP)", y=0.99)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fig9_multibatch_throughput.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_file}")


if __name__ == "__main__":
    plot_multibatch_throughput_chart()
