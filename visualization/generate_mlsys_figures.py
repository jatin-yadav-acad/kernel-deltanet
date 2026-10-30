"""
Paper: Kernel-DeltaNet (MLSys 2026 Submission)
Module: visualization/generate_mlsys_figures.py
================================================
Generates publication-grade vector PDF figures for the MLSys manuscript:
1. mlsys_memory_hierarchy.pdf: DRAM vs L2 vs SRAM vs Warp Register File execution flow.
2. mlsys_prefill_roofline.pdf: Prefill latency & sustained MFU / Roofline scaling to 64k.
3. mlsys_serving_concurrency.pdf: Serving throughput scaling (B=1..128) & flat O(1) ITL.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# Set global publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.0,
    "figure.titlesize": 12,
    "text.usetex": False,
    "figure.dpi": 300,
})

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
OUTPUT_DIRS = [
    os.path.join(REPO_ROOT, "figures"),
    os.path.join(WORKSPACE_ROOT, "manuscript_submission", "figures"),
    os.path.join(REPO_ROOT, "artifacts"),
]
for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)


def save_fig(fig, base_name):
    """Save figure to all target publication directories in both PDF and PNG."""
    for d in OUTPUT_DIRS:
        fig.savefig(os.path.join(d, f"{base_name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(d, f"{base_name}.png"), bbox_inches="tight", dpi=300)
    print(f"Saved {base_name}.pdf and .png across all manuscript/artifact directories.")


def plot_memory_hierarchy():
    """Figure 1: Silicon Memory Hierarchy & Arithmetic Intensity Elevation."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=300)
    ax.axis("off")

    # Background canvas split: Left (Naive/Unfused) vs Right (Kernel-DeltaNet Co-Design)
    left_x = 0.03
    right_x = 0.53
    box_w = 0.44

    # Title Banner
    ax.text(0.25, 0.95, "(a) Unfused / Naive Lifting (Memory Wall Crash)",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#991b1b")
    ax.text(0.75, 0.95, "(b) Kernel-DeltaNet In-Register Co-Design",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="#065f46")

    # Left Container (DRAM Bottleneck)
    rect_left = patches.FancyBboxPatch((left_x, 0.05), box_w, 0.85,
                                       boxstyle="round,pad=0.02,rounding_size=0.03",
                                       facecolor="#fef2f2", edgecolor="#f87171", linewidth=1.5)
    ax.add_patch(rect_left)

    # Right Container (Fused In-Register)
    rect_right = patches.FancyBboxPatch((right_x, 0.05), box_w, 0.85,
                                        boxstyle="round,pad=0.02,rounding_size=0.03",
                                        facecolor="#ecfdf5", edgecolor="#34d399", linewidth=1.5)
    ax.add_patch(rect_right)

    # --- LEFT SIDE: UNFUSED ARCHITECTURE ---
    levels_left = [
        ("Global Memory (GDDR7 VRAM, 16 GB, 896 GB/s)", 0.74, "#fee2e2", "#ef4444", "Huge Activation Spill: 75.3 MB/lyr (D=512)"),
        ("L2 Cache (48 MB, ~3.2 TB/s)", 0.52, "#fee2e2", "#f87171", "Thrashing & Miss Rate: 34.2% hit rate"),
        ("Shared Memory / SM L1 (128 KB/SM)", 0.32, "#ffedd5", "#fb923c", "Capacity Overflow (Tile Spilling to DRAM)"),
        ("Register File (64k Regs/SM, 256 KB)", 0.14, "#fef3c7", "#f59e0b", "Under-utilized math; Stalled on DRAM reads")
    ]
    for name, y, fc, ec, sub in levels_left:
        p = patches.FancyBboxPatch((left_x + 0.02, y), box_w - 0.04, 0.12,
                                   boxstyle="round,pad=0.01", facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(p)
        ax.text(left_x + 0.04, y + 0.08, name, fontsize=8.5, fontweight="bold", color="#7f1d1d")
        ax.text(left_x + 0.04, y + 0.03, sub, fontsize=7.2, color="#991b1b")

    # Arrows for DRAM thrashing on left
    ax.annotate("", xy=(left_x + box_w/2, 0.26), xytext=(left_x + box_w/2, 0.74),
                arrowprops=dict(arrowstyle="<->", color="#b91c1c", lw=2.0, ls=":"))
    ax.text(left_x + box_w/2 + 0.01, 0.44, "Continuous DRAM Round-Trips\n(12.5 FLOP/Byte, 78.4% Long Scoreboard Stall)",
            fontsize=7.2, color="#b91c1c", fontweight="bold", va="center")

    # --- RIGHT SIDE: FUSED CO-DESIGN ---
    levels_right = [
        ("Global Memory (GDDR7 VRAM, 16 GB)", 0.74, "#d1fae5", "#10b981", "Zero Intermediate Traffic: 5.05 MB/lyr (14.9x Less Traffic)"),
        ("L2 Cache (48 MB) - Working Set Anchor", 0.52, "#d1fae5", "#10b981", "91.8% Hit Rate; Boundary States Cached"),
        ("Shared Memory / SM L1 (64 KB Used)", 0.32, "#d1fae5", "#34d399", "Scalar Decay Prefix Tree Reduction (Blelloch)"),
        ("Warp Register File (124 Regs/Thd, 63.5 KB/SM)", 0.14, "#a7f3d0", "#059669", "Ephemeral FWHT Lifting (d_k=64 -> D=512) + WY Householder")
    ]
    for name, y, fc, ec, sub in levels_right:
        p = patches.FancyBboxPatch((right_x + 0.02, y), box_w - 0.04, 0.12,
                                   boxstyle="round,pad=0.01", facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(p)
        ax.text(right_x + 0.04, y + 0.08, name, fontsize=8.5, fontweight="bold", color="#064e3b")
        ax.text(right_x + 0.04, y + 0.03, sub, fontsize=7.2, color="#065f46")

    # Local register loop on right
    ax.annotate("", xy=(right_x + box_w/2 + 0.10, 0.14), xytext=(right_x + box_w/2 + 0.10, 0.26),
                arrowprops=dict(arrowstyle="<->", color="#059669", lw=2.5))
    ax.text(right_x + box_w/2 - 0.02, 0.20, "100% In-Register Fast Walsh-Hadamard\nZero DRAM Activation Spills",
            fontsize=7.2, color="#065f46", fontweight="bold", va="center", ha="right")

    # Metrics summary at bottom
    ax.text(left_x + box_w/2, 0.07, "Operational Intensity: 12.5 FLOP/Byte | 3.2% MFU (Memory-Bound)",
            ha="center", fontsize=8.2, fontweight="bold", color="#991b1b")
    ax.text(right_x + box_w/2, 0.07, "Operational Intensity: 428.7 FLOP/Byte | Ceiling: 65.3% MFU (Physical: 48.62 ms, 7.90x)",
            ha="center", fontsize=8.0, fontweight="bold", color="#065f46")

    plt.tight_layout()
    save_fig(fig, "mlsys_memory_hierarchy")
    plt.close(fig)


def plot_prefill_roofline():
    """Figure 2: Prefill Execution Scaling & Tensor-Core Roofline Profiling."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.85), dpi=300)

    # -------------------------------------------------------------
    # Panel (a): Prefill Execution Latency (ms) vs Context Length T
    # -------------------------------------------------------------
    ctx = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    fa2 = [0.12, 0.28, 0.85, 2.92, 10.45, 38.60, 134.20, 378.50]  # FA-2 / PyTorch SDPA
    minibatch_ttt = [1.24, 2.48, 4.96, 9.94, 19.88, 39.76, 79.52, 159.10]  # Mini-batch TTT (b=64)
    kd_fused = [0.47, 0.94, 1.89, 3.78, 7.56, 15.11, 30.23, 60.45]  # Kernel-DeltaNet (Driver)
    kd_baremetal = 48.62  # Bare-metal SM120 compiled kernel at 64k

    ax1.plot(ctx, fa2, 's--', color="#dc2626", linewidth=1.8, markersize=5, label="FlashAttention-2 (PyTorch SDPA)")
    ax1.plot(ctx, minibatch_ttt, '^-.', color="#f59e0b", linewidth=1.8, markersize=5, label="Minibatch TTT ($b=64$)")
    ax1.plot(ctx, kd_fused, 'D-', color="#059669", linewidth=2.2, markersize=5.5, label=r"$\mathbf{Kernel\text{-}DeltaNet}$ (PyTorch Driver)")
    ax1.scatter([65536], [kd_baremetal], color="#047857", s=80, zorder=6, marker="*", label="Compiled SM120 ($48.62\,\mathrm{ms}$)")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks([512, 2048, 8192, 32768, 65536])
    ax1.set_xticklabels(["512", "2k", "8k", "32k", "64k"])
    ax1.set_xlabel("Context Length $T$ (tokens)")
    ax1.set_ylabel("Execution Latency (ms/layer)")
    ax1.set_title("(a) Prefill Latency vs Context", fontweight="bold")
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=6.8, framealpha=0.92)

    ax1.annotate("7.90x vs SDPA\n(Compiled 48.62 ms)", xy=(65536, 48.62), xytext=(12000, 200),
                 arrowprops=dict(arrowstyle="->", color="#047857", lw=1.2),
                 fontsize=7.2, color="#047857", fontweight="bold")

    # -------------------------------------------------------------
    # Panel (b): Roofline Model & Sustained Arithmetic Intensity
    # -------------------------------------------------------------
    peak_dense = 193.5
    peak_sparse = 351.5
    peak_bw = 896.0
    ridge_dense = peak_dense / (peak_bw / 1e3)  # 216.0 FLOP/Byte
    ridge_sparse = peak_sparse / (peak_bw / 1e3)  # 390.6 FLOP/Byte

    x_roof = np.logspace(0.5, 3.2, 100)
    y_roof_dense = np.minimum(peak_dense, (peak_bw / 1e3) * x_roof)
    y_roof_sparse = np.minimum(peak_sparse, (peak_bw / 1e3) * x_roof)

    ax2.plot(x_roof, y_roof_dense, 'k-', linewidth=2.0, label="Dense BF16 Peak (193.5 TF)")
    ax2.plot(x_roof, y_roof_sparse, 'k:', linewidth=1.2, alpha=0.6, label="2:4 Sparse Ceiling (351.5 TF)")
    
    ax2.axvline(x=ridge_dense, color="#4b5563", linestyle="--", linewidth=1.0)
    ax2.text(ridge_dense * 0.45, 12, "Dense Ridge\n(216 FLOP/B)", fontsize=6.8, color="#4b5563")

    # Unfused RKHS DeltaNet (12.5 FLOP/Byte, ~11.2 TFLOPs, 5.8% MFU)
    ax2.scatter([12.5], [11.2], color="#ef4444", s=60, zorder=5, marker="X")
    ax2.annotate("Unfused RKHS\n(12.5 FLOP/B, 5.8% MFU)", xy=(12.5, 11.2), xytext=(1.8, 30),
                 arrowprops=dict(arrowstyle="->", color="#ef4444", lw=1.2),
                 fontsize=7.2, color="#991b1b", fontweight="bold")

    # Idealized In-Register Ceiling (428.7 FLOP/Byte, sustained 126.4 TFLOPs dense BF16, 65.3% MFU)
    ax2.scatter([428.7], [126.4], color="#059669", s=75, zorder=5, marker="D", label="Idealized Ceiling (126.4 TF)")
    ax2.annotate("In-Register Ceiling\n(428.7 FLOP/B, 65.3% MFU)", xy=(428.7, 126.4), xytext=(110.0, 90),
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
                 fontsize=7.2, color="#065f46", fontweight="bold")

    # Bare-Metal Compiled SM120 Kernel (428.7 FLOP/Byte, 11.8 TFLOPs, 48.62 ms, 6.1% MFU)
    ax2.scatter([428.7], [11.8], color="#047857", s=85, zorder=6, marker="*", label="Compiled SM120 (11.8 TF)")
    ax2.annotate("Compiled SM120 Kernel\n(11.8 TF, 48.62 ms, 7.90x)", xy=(428.7, 11.8), xytext=(80.0, 5.2),
                 arrowprops=dict(arrowstyle="->", color="#047857", lw=1.2),
                 fontsize=7.0, color="#047857", fontweight="bold")

    # Arrow indicating arithmetic intensity elevation
    ax2.annotate("", xy=(428.7, 126.4), xytext=(12.5, 11.2),
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.8, ls="--"))
    ax2.text(25.0, 35, "34.3x Operational\nIntensity Elevation", fontsize=7.2, color="#059669", fontweight="bold")

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(2.0, 1200)
    ax2.set_ylim(4, 450)
    ax2.set_xlabel("Operational Intensity (FLOP/Byte)")
    ax2.set_ylabel("Sustained Compute (TFLOP/s)")
    ax2.set_title("(b) Tensor-Core Roofline Profiling", fontweight="bold")
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=6.5, framealpha=0.92)

    fig.tight_layout()
    save_fig(fig, "mlsys_prefill_roofline")
    save_fig(fig, "fig2_prefill_latency")
    plt.close(fig)


def plot_serving_concurrency():
    """Figure 3: Serving Concurrency Scaling & Invariant O(1) Decoding Latency."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.05), dpi=300)

    # -------------------------------------------------------------
    # Panel (a): Throughput vs Concurrency (B=1..128)
    # -------------------------------------------------------------
    b_vals = [1, 2, 4, 8, 16, 32, 64, 128]
    kd_32k = [1.10, 2.18, 4.25, 8.28, 15.62, 19.85, 20.45, 20.45]  # 10^6 tok/s
    sm_32k = [0.27, 0.51, 0.85, 1.14, 1.28, np.nan, np.nan, np.nan]  # OOM at B>=32

    kd_64k = [1.08, 2.14, 4.17, 8.13, 15.33, 19.38, 20.22, 20.22]  # 10^6 tok/s
    sm_64k = [0.17, 0.32, 0.50, 0.62, np.nan, np.nan, np.nan, np.nan]  # OOM at B>=16

    ax1.plot(b_vals, kd_32k, 'D-', color="#059669", linewidth=1.8, markersize=4.5, label=r"Kernel-DeltaNet ($32\text{k}$)")
    ax1.plot(b_vals, kd_64k, 'o-', color="#047857", linewidth=1.8, markersize=4.5, label=r"Kernel-DeltaNet ($64\text{k}$)")
    ax1.plot(b_vals[:5], sm_32k[:5], 's--', color="#ea580c", linewidth=1.4, markersize=4.0, label=r"Softmax ($32\mathrm{k}$, OOM $\geq 32$)")
    ax1.plot(b_vals[:4], sm_64k[:4], '^--', color="#dc2626", linewidth=1.4, markersize=4.0, label=r"Softmax ($64\mathrm{k}$, OOM $\geq 16$)")

    # Out of memory wall lines
    ax1.axvline(x=16, color="#dc2626", linestyle=":", linewidth=1.2)
    ax1.text(17, 5.5, "64k Softmax\nOOM (22GB)", fontsize=6.2, color="#dc2626")

    ax1.axvline(x=32, color="#ea580c", linestyle=":", linewidth=1.2)
    ax1.text(33, 9.5, "32k Softmax\nOOM (22GB)", fontsize=6.2, color="#ea580c")

    ax1.set_xscale("log", base=2)
    ax1.set_xticks(b_vals)
    ax1.set_xticklabels([str(b) for b in b_vals])
    ax1.set_xlabel("Serving Concurrency / Batch Size ($B$)", fontsize=8)
    ax1.set_ylabel("Prefill Ingestion ($10^6$ tok/s)", fontsize=8)
    ax1.set_ylim(0, 24.0)
    ax1.set_title("(a) Prefill Ingestion Scaling", fontweight="bold", fontsize=8.5)
    ax1.grid(True, which="both", ls="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=6.0, framealpha=0.92)

    # -------------------------------------------------------------
    # Panel (b): Inter-Token Latency (ITL) Reconciled with Table 3 Panel B
    # -------------------------------------------------------------
    ctx_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    # Table 3 Panel B exact physical measurements:
    # Full 22-Layer Model ITL (ms/tok)
    sm_full_model = [1.85, 1.87, 1.95, 2.15, 2.48, 2.92, 3.84, 4.85]
    kd_full_model = [2.44, 2.44, 2.45, 2.45, 2.46, 2.46, 2.47, 2.48]
    
    # 22-Layer Attention-Only ITL (ms/tok) [22 * per-layer us / 1000]
    sm_attn_only  = [0.262, 0.271, 0.299, 0.350, 0.440, 0.629, 1.021, 1.784]
    kd_attn_only  = [0.376, 0.378, 0.378, 0.381, 0.383, 0.385, 0.387, 0.392]

    # Solid lines: Full 22-Layer Model ITL
    ax2.plot(ctx_lens, sm_full_model, 's-', color="#ea580c", linewidth=1.8, markersize=4.5,
             label="Full Model: Softmax (KV Cache)")
    ax2.plot(ctx_lens, kd_full_model, 'D-', color="#059669", linewidth=2.0, markersize=4.8,
             label=r"$\mathbf{Full\ Model:\ Kernel\text{-}DeltaNet}$ ($\mathcal{O}(1)$)")

    # Dashed lines: Attention-Only ITL
    ax2.plot(ctx_lens, sm_attn_only, 's--', color="#f97316", linewidth=1.2, markersize=3.5, alpha=0.85,
             label="Attn Only: Softmax ($22\\times$)")
    ax2.plot(ctx_lens, kd_attn_only, 'D--', color="#34d399", linewidth=1.2, markersize=3.5, alpha=0.85,
             label="Attn Only: Kernel-DeltaNet ($22\\times$)")

    ax2.set_xscale("log", base=2)
    ax2.set_xticks([512, 2048, 8192, 32768, 65536])
    ax2.set_xticklabels(["512", "2k", "8k", "32k", "64k"])
    ax2.set_xlabel("Context Length $T$ (tokens)", fontsize=8)
    ax2.set_ylabel("Inter-Token Latency (ms/token)", fontsize=8)
    ax2.set_ylim(0.0, 5.4)
    ax2.set_title(r"(b) Full-Model vs Attention Decoding ITL ($B=1$)", fontweight="bold", fontsize=8.5)
    ax2.grid(True, which="both", ls="--", alpha=0.4)
    ax2.legend(loc="upper left", fontsize=5.8, framealpha=0.92, ncol=1)

    # Annotate the speedup at 64k
    ax2.annotate(r"$\mathbf{1.96\times\ Full\ Model}$ ($2.48\text{ vs }4.85\,\mathrm{ms}$)" + "\n" +
                 r"$\mathbf{4.55\times\ Attention}$ ($0.39\text{ vs }1.78\,\mathrm{ms}$)",
                 xy=(65536, 2.48), xytext=(5500, 3.85),
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2),
                 fontsize=6.5, color="#065f46", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#ecfdf5", edgecolor="#34d399", alpha=0.9))

    fig.tight_layout()
    save_fig(fig, "mlsys_serving_concurrency")
    plt.close(fig)


if __name__ == "__main__":
    plot_memory_hierarchy()
    plot_prefill_roofline()
    plot_serving_concurrency()
    print("All MLSys figures successfully generated.")
