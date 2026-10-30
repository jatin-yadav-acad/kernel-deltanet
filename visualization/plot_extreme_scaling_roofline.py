"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: visualization.plot_extreme_scaling_roofline
=====================================================================
Publication-Ready 300 DPI Visualizations for:
1. Fig 6: Extreme Context Scaling (2k to 32k tokens: Throughput & VRAM scaling)
2. Fig 7: Physical Hardware Roofline Model on NVIDIA GeForce RTX 5070 Ti
3. Fig 8: Algorithmic Fine-Tuning Convergence & Task Accuracy

Constraints:
- 300 DPI resolution
- Clean styling with NO hardcoded 'Figure X:' in bitmap text
- High contrast, colorblind-friendly academic palettes
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def setup_matplotlib_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.grid': True,
        'grid.alpha': 0.35,
        'grid.linestyle': '--',
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def plot_extreme_context_scaling(
    json_path: str = "results/extreme_context_benchmark.json",
    output_dir: str = "artifacts"
):
    if not os.path.exists(json_path):
        print(f"[!] File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    setup_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors = {
        "Kernel-DeltaNet (Chunked)": "#1f77b4",
        "Linear DeltaNet": "#2ca02c",
        "Standard Attention (SDPA)": "#d62728"
    }
    markers = {
        "Kernel-DeltaNet (Chunked)": "o",
        "Linear DeltaNet": "s",
        "Standard Attention (SDPA)": "^"
    }

    scaling = data["scaling_results"]

    for model_name, res in scaling.items():
        lengths = res["lengths"]
        throughputs = res["throughput_tok_per_sec"]
        vrams = res["peak_vram_mb"]
        statuses = res["status"]

        # Filter valid success points
        valid_idx = [i for i, s in enumerate(statuses) if s == "SUCCESS"]
        valid_lens = [lengths[i] for i in valid_idx]
        valid_tps = [throughputs[i] for i in valid_idx]
        valid_vram = [vrams[i] for i in valid_idx]

        c = colors.get(model_name, "#7f7f7f")
        m = markers.get(model_name, "x")

        # Subplot 1: Throughput
        ax1.plot(valid_lens, valid_tps, marker=m, linewidth=2.2, markersize=7, label=model_name, color=c)
        
        # Subplot 2: VRAM
        ax2.plot(valid_lens, valid_vram, marker=m, linewidth=2.2, markersize=7, label=model_name, color=c)

        # Plot OOM indicator if any
        oom_idx = [i for i, s in enumerate(statuses) if s == "OOM"]
        for oi in oom_idx:
            ax1.scatter([lengths[oi]], [1000], color="#d62728", marker="X", s=140, zorder=5)
            ax1.annotate("OOM", (lengths[oi], 1200), color="#d62728", fontweight="bold", ha="center")
            ax2.scatter([lengths[oi]], [16000], color="#d62728", marker="X", s=140, zorder=5)
            ax2.annotate("OOM", (lengths[oi], 16200), color="#d62728", fontweight="bold", ha="center")

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Context Length (Tokens)")
    ax1.set_ylabel("Inference Throughput (Tokens / Second)")
    ax1.set_title("Throughput Scaling: Extreme Sequence Horizons")
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.legend(frameon=True, facecolor="#f8f9fa", edgecolor="none")

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Context Length (Tokens)")
    ax2.set_ylabel("Peak Allocated VRAM (MB)")
    ax2.set_title("Memory Allocation: Sub-Linear vs Quadratic Footprint")
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.legend(frameon=True, facecolor="#f8f9fa", edgecolor="none")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fig6_extreme_context_scaling.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_file}")


def plot_hardware_roofline(
    json_path: str = "results/hardware_roofline_benchmark.json",
    output_dir: str = "artifacts"
):
    if not os.path.exists(json_path):
        print(f"[!] File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(9, 6))

    specs = data["hardware_specs"]
    peak_compute = specs["peak_dense_bf16_tflops"]
    peak_bw = specs["peak_memory_bandwidth_gb_s"]
    ridge = specs["ridge_point_flops_per_byte"]

    # Generate theoretical roofline ceiling curve
    x_ai = np.logspace(0, 3, 500) # 1 to 1000 FLOPs/Byte
    # Roofline: min(peak_compute, AI * peak_bw / 1000)
    y_roof = np.minimum(peak_compute, x_ai * (peak_bw / 1000.0))

    # Plot theoretical hardware boundary
    ax.plot(x_ai, y_roof, color="#111111", linewidth=2.5, linestyle="-", label=f"RTX 5070 Ti Theoretical Ceiling ({peak_compute} TFLOP/s, {peak_bw} GB/s)")
    ax.fill_between(x_ai, 0.01, y_roof, color="#e9ecef", alpha=0.45)

    # Vertical dashed line at ridge point
    ax.axvline(ridge, color="#e65100", linestyle=":", linewidth=1.8, label=f"Roofline Ridge Point ({ridge:.1f} FLOP/B)")
    ax.text(ridge * 1.1, 1.5, "Compute-Bound\nRegime", color="#b71c1c", fontweight="bold", fontsize=10)
    ax.text(ridge * 0.15, 1.5, "Memory-Bound\nRegime", color="#0d47a1", fontweight="bold", fontsize=10)

    # Plot Chunk Size sweep points
    chunk_points = data["chunk_size_sweep"]
    c_x = [p["arithmetic_intensity_flops_per_byte"] for p in chunk_points]
    c_y = [p["achieved_tflops"] for p in chunk_points]
    c_labels = [f"C={p['chunk_size']}" for p in chunk_points]

    ax.scatter(c_x, c_y, color="#1f77b4", s=90, zorder=5, label="Kernel-DeltaNet (Chunk Size C in [16, 512])")
    for i, txt in enumerate(c_labels):
        ax.annotate(txt, (c_x[i], c_y[i]), textcoords="offset points", xytext=(6, -2), fontsize=9, color="#0d47a1")

    # Plot RKHS Factor sweep points
    rkhs_points = data["rkhs_factor_sweep"]
    r_x = [p["arithmetic_intensity_flops_per_byte"] for p in rkhs_points]
    r_y = [p["achieved_tflops"] for p in rkhs_points]
    r_labels = [f"r={p['rkhs_factor']}" for p in rkhs_points]

    ax.scatter(r_x, r_y, color="#2ca02c", marker="D", s=80, zorder=5, label="Kernel-DeltaNet (RKHS Factor r in [1, 16])")
    for i, txt in enumerate(r_labels):
        ax.annotate(txt, (r_x[i], r_y[i]), textcoords="offset points", xytext=(-28, 4), fontsize=9, color="#1b5e20")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.5, peak_compute * 1.3)
    ax.set_xlabel("Operational Arithmetic Intensity (FLOPs / Byte)")
    ax.set_ylabel("Achieved Performance (TFLOP/s)")
    ax.set_title("NVIDIA GeForce RTX 5070 Ti Roofline Analysis for Kernel-DeltaNet")
    ax.legend(frameon=True, facecolor="#f8f9fa", loc="upper left", fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fig7_hardware_roofline_curve.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_file}")


def plot_algorithmic_finetuning(
    json_path: str = "results/algorithmic_finetuning_results.json",
    output_dir: str = "artifacts"
):
    if not os.path.exists(json_path):
        print(f"[!] File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    setup_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    tasks = data["tasks"]
    task_labels = {
        "multihop_2hop": "Multi-Hop (2-Hop)",
        "multihop_3hop": "Multi-Hop (3-Hop)",
        "dynamic_overwrite": "Dynamic Overwrite",
        "dyck3_stack": "Dyck-3 Stack"
    }

    models = ["Kernel-DeltaNet (Ours)", "Linear DeltaNet", "Standard Transformer"]
    colors = {
        "Kernel-DeltaNet (Ours)": "#1f77b4",
        "Linear DeltaNet": "#2ca02c",
        "Standard Transformer": "#d62728"
    }

    # Subplot 1: Bar chart of Final Accuracies across tasks
    x_pos = np.arange(len(tasks))
    width = 0.26

    for i, model in enumerate(models):
        accs = [data["results"][t][model]["final_accuracy"] for t in tasks]
        ax1.bar(x_pos + (i - 1) * width, accs, width=width, label=model, color=colors[model], alpha=0.9, edgecolor="none")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([task_labels.get(t, t) for t in tasks])
    ax1.set_ylabel("Final Evaluation Accuracy (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Algorithmic Fine-Tuning Task Accuracy")
    ax1.legend(frameon=True, facecolor="#f8f9fa")

    # Subplot 2: Learning curve comparison on Multi-Hop 3-hop
    focus_task = "multihop_3hop"
    if focus_task in data["results"]:
        for model in models:
            acc_curve = data["results"][focus_task][model]["acc_curve"]
            steps = [s * 10 for s in range(len(acc_curve))]
            ax2.plot(steps, acc_curve, linewidth=2.2, label=model, color=colors[model], marker="o", markersize=4)

        ax2.set_xlabel("Fine-Tuning Training Steps")
        ax2.set_ylabel("Validation Accuracy (%)")
        ax2.set_title("Sample Efficiency: Multi-Hop 3-Hop Reasoning")
        ax2.set_ylim(0, 105)
        ax2.legend(frameon=True, facecolor="#f8f9fa")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "fig8_algorithmic_finetuning_curves.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved {out_file}")


def generate_extended_figures():
    plot_extreme_context_scaling()
    plot_hardware_roofline()
    plot_algorithmic_finetuning()


if __name__ == "__main__":
    generate_extended_figures()
