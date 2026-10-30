"""
Kernel-DeltaNet MLSys 2026
Module: visualization.generate_all_figures
=========================================
Master Figure Generator producing 300 DPI publication-ready figures.
All figures comply with strict conference submission standards:
- No embedded 'Figure X:' title strings in bitmaps.
- Subpanel labels match paper text.
"""

import os
import shutil
import json
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .plot_throughput_cliff import plot_throughput_cliff
from .plot_mqar_capacity_curves import plot_mqar_capacity_curves
from .plot_generation_memory_flat import plot_generation_memory_flat
from .plot_needle_heatmap import plot_needle_heatmap


def build_master_overview_figure(
    fig1_path: str,
    fig2_path: str,
    fig3_path: str,
    fig4_path: str,
    save_path: str
):
    """
    Combines all four experimental figures into a unified 2x2 presentation grid:
    (A) Throughput Step Latency
    (B) MQAR Capacity Saturation
    (C) Strict O(1) Autoregressive Memory
    (D) 32k Needle-in-a-Haystack Passkey Retrieval
    """
    if not (os.path.exists(fig1_path) and os.path.exists(fig2_path) and os.path.exists(fig3_path) and os.path.exists(fig4_path)):
        print(f"[-] Skipping master overview: missing one of {fig1_path}, {fig2_path}, {fig3_path}, {fig4_path}")
        return

    img1 = mpimg.imread(fig1_path)
    img2 = mpimg.imread(fig2_path)
    img3 = mpimg.imread(fig3_path)
    img4 = mpimg.imread(fig4_path)

    fig, axes = plt.subplots(2, 2, figsize=(22, 14), dpi=300)
    
    axes[0, 0].imshow(img1)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(img2)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(img3)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(img4)
    axes[1, 1].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Master 4-panel overview figure generated and saved to {save_path}")


def generate_all_figures():
    print("=" * 90)
    print("KERNEL-DELTANET: GENERATING ALL PUBLICATION FIGURES (300 DPI)")
    print("=" * 90)
    t0 = time.time()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    artifacts_dir = os.path.join(repo_root, "artifacts")
    manuscript_fig_dir = os.path.join(repo_root, "figures")
    
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(manuscript_fig_dir, exist_ok=True)
    
    fig1 = os.path.join(artifacts_dir, "fig1_throughput_cliff.png")
    fig2 = os.path.join(artifacts_dir, "fig2_mqar_capacity_curves.png")
    fig3 = os.path.join(artifacts_dir, "fig3_generation_memory_flat.png")
    fig4 = os.path.join(artifacts_dir, "fig5_needle_in_a_haystack_heatmap.png")
    master_fig = os.path.join(artifacts_dir, "master_overview.png")

    print("[1/4] Generating Figure 1: Fair Hardware Throughput Cliff ($B=8$)...")
    plot_throughput_cliff(save_path=fig1)

    print("[2/4] Generating Figure 2: MQAR Capacity Saturation Curves...")
    plot_mqar_capacity_curves(save_path=fig2)

    print("[3/4] Generating Figure 3: Strict O(1) Generation Memory...")
    plot_generation_memory_flat(save_path=fig3)

    print("[4/4] Generating Figure 4: 32k Needle-in-a-Haystack Passkey Retrieval Heatmap...")
    plot_needle_heatmap(save_path=fig4)

    print("[*] Generating Master 4-Panel Overview Figure...")
    build_master_overview_figure(fig1, fig2, fig3, fig4, save_path=master_fig)

    # Sync to figures directory
    target_files = [fig1, fig2, fig3, fig4, master_fig]
    for src in target_files:
        if os.path.exists(src):
            basename = os.path.basename(src)
            dst_ms = os.path.join(manuscript_fig_dir, basename)
            shutil.copyfile(src, dst_ms)

    elapsed = time.time() - t0
    print(f"\n[+] All figures successfully generated and synced in {elapsed:.2f}s!\n")


if __name__ == "__main__":
    generate_all_figures()
