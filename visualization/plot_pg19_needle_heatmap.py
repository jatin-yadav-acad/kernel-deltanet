"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: visualization.plot_pg19_needle_heatmap
================================================================
Generates 300 DPI comparative 2D heatmaps for Milestone 4:
Real-document long-context needle retrieval on PG-19 prose.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_pg19_needle_heatmap(
    json_path: str = "results/pg19_needle_benchmark.json",
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
        'figure.dpi': 300,
        'savefig.dpi': 300
    })

    seq_lens = data["sequence_lengths"]
    depths = data["depths"]
    results = data["results"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Grid 1: Kernel-DeltaNet log10(Target Prob)
    k_grid = np.zeros((len(depths), len(seq_lens)))
    l_grid = np.zeros((len(depths), len(seq_lens)))

    for j, T in enumerate(seq_lens):
        t_key = f"T_{T}"
        for i, d in enumerate(depths):
            d_key = f"depth_{d}%"
            k_entry = results.get("Kernel-DeltaNet (Ours)", {}).get(t_key, {}).get(d_key, {})
            l_entry = results.get("Linear DeltaNet", {}).get(t_key, {}).get(d_key, {})
            
            k_prob = k_entry.get("target_prob", 1e-45)
            l_prob = l_entry.get("target_prob", 1e-45)

            k_grid[i, j] = np.log10(max(1e-45, k_prob))
            l_grid[i, j] = np.log10(max(1e-45, l_prob))

    # Common color scale: -45 to 0
    vmin, vmax = -45, 0

    im1 = ax1.imshow(k_grid, cmap="Blues", aspect="auto", vmin=vmin, vmax=vmax, origin="lower")
    ax1.set_title("Kernel-DeltaNet (Ours): Stable Retention")
    ax1.set_xticks(range(len(seq_lens)))
    ax1.set_xticklabels([f"{t//1024}k" for t in seq_lens])
    ax1.set_yticks(range(len(depths)))
    ax1.set_yticklabels([f"{d}%" for d in depths])
    ax1.set_xlabel("Context Length Horizon (T)")
    ax1.set_ylabel("Needle Depth Position (%)")

    # Annotate values
    for i in range(len(depths)):
        for j in range(len(seq_lens)):
            ax1.text(j, i, f"{k_grid[i, j]:.1f}", ha="center", va="center", color="black" if k_grid[i, j] < -20 else "white", fontsize=9)

    im2 = ax2.imshow(l_grid, cmap="Reds_r", aspect="auto", vmin=vmin, vmax=vmax, origin="lower")
    ax2.set_title("Linear DeltaNet: Catastrophic Rank Saturation")
    ax2.set_xticks(range(len(seq_lens)))
    ax2.set_xticklabels([f"{t//1024}k" for t in seq_lens])
    ax2.set_yticks(range(len(depths)))
    ax2.set_yticklabels([f"{d}%" for d in depths])
    ax2.set_xlabel("Context Length Horizon (T)")
    ax2.set_ylabel("Needle Depth Position (%)")

    for i in range(len(depths)):
        for j in range(len(seq_lens)):
            ax2.text(j, i, f"{l_grid[i, j]:.1f}", ha="center", va="center", color="black" if l_grid[i, j] < -20 else "white", fontsize=9)

    cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation="horizontal", fraction=0.07, pad=0.18)
    cbar.set_label("Log10 Target Token Probability [log10(P)] (Higher / Closer to 0 is Better)")

    plt.suptitle("PG-19 Natural Narrative Haystack: Target Probability Retention Across Context Horizons", y=0.98)
    
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fig10_pg19_needle_heatmap.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_file}")


if __name__ == "__main__":
    plot_pg19_needle_heatmap()
