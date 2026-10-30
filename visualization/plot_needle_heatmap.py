"""
Kernel-DeltaNet Engine
Module: visualization.plot_needle_heatmap
===========================================================
Generates 300 DPI publication figure:
2D Needle-In-A-Haystack (NIAH) Passkey Retrieval Heatmap up to 32k Tokens.
Compares Linear-TTT / DeltaNet vs Kernel-DeltaNet across context depths.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_needle_heatmap(
    benchmark_path: str = None,
    save_path: str = "artifacts/fig5_needle_in_a_haystack_heatmap.png"
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    if benchmark_path is None or not os.path.exists(benchmark_path):
        candidates = [
            os.path.join(repo_root, "results", "needle_benchmark.json"),
            "results/needle_benchmark.json",
            "results/needle_benchmark.json"
        ]
        for c in candidates:
            if os.path.exists(c):
                benchmark_path = c
                break

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(benchmark_path, "r") as f:
        data = json.load(f)

    lengths = [str(l) for l in data["context_lengths"]]
    depth_labels = [f"{int(round(d*100))}%" for d in data["depths"]]

    k_grid = data["kernel_ttt_matrix"]
    l_grid = data["linear_matrix"]

    k_mat = np.zeros((len(depth_labels), len(lengths)))
    l_mat = np.zeros((len(depth_labels), len(lengths)))

    for j, l in enumerate(lengths):
        for i, d in enumerate(depth_labels):
            k_mat[i, j] = k_grid[l][d] * 100.0
            l_mat[i, j] = l_grid[l][d] * 100.0

    length_labels = [f"{int(int(l)/1024)}k" for l in lengths]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300, sharey=True)

    # Colormap: YlGn (Yellow to Green)
    cmap = sns.color_palette("YlGn", as_cmap=True)

    # Panel 1: Linear-TTT Baseline
    sns.heatmap(
        l_mat,
        ax=axes[0],
        cmap=cmap,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        cbar=False,
        xticklabels=length_labels,
        yticklabels=depth_labels,
        linewidths=0.8,
        linecolor="white"
    )
    axes[0].set_title(r"$\mathbf{(a)\ Baseline:\ Linear\ DeltaNet}$" + "\n(Subspace Rank Saturation)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Context Window Length $T$ (Tokens)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Needle Document Depth (%)", fontsize=11, fontweight='bold')

    # Panel 2: Kernel-DeltaNet
    sns_k = sns.heatmap(
        k_mat,
        ax=axes[1],
        cmap=cmap,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        cbar_kws={"label": "Passkey Retrieval Accuracy (%)"},
        xticklabels=length_labels,
        yticklabels=depth_labels,
        linewidths=0.8,
        linecolor="white"
    )
    axes[1].set_title(r"$\mathbf{(b)\ Ours:\ Kernel\text{-}DeltaNet}$" + "\n(Bochner Exponential Crosstalk Decoupling)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Context Window Length $T$ (Tokens)", fontsize=11, fontweight='bold')

    cbar = sns_k.figure.axes[-1]
    cbar.yaxis.label.set_size(11)
    cbar.yaxis.label.set_weight("bold")

    plt.suptitle(
        r"32,768-Token Needle-In-A-Haystack (NIAH) Passkey Retrieval Heatmap",
        fontsize=15,
        fontweight='bold',
        y=0.98
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    
    ms_path = os.path.join(repo_root, "manuscript", "figures", "fig5_needle_in_a_haystack_heatmap.png")
    os.makedirs(os.path.dirname(ms_path), exist_ok=True)
    plt.savefig(ms_path, dpi=300)
    plt.close()
    print(f"[+] NIAH 2D Heatmap figure generated and saved to {save_path} and {ms_path}")


if __name__ == "__main__":
    plot_needle_heatmap()
