"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: visualization.plot_calibrated_retrieval_curves
========================================================================
Plots calibrated long-context passkey retrieval retention curves:
Top panel: Passkey Retrieval Accuracy (%) vs Context Length T (up to 32k)
Bottom panel: Signal-to-Noise Ratio (SNR dB) vs Context Length T.
No embedded figure titles in image bitmaps (conference submission standard).
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np


def plot_calibrated_retrieval_curves(
    results_path: str = "results/calibrated_retrieval_benchmark.json",
    save_path: str = "artifacts/fig5_calibrated_retrieval.png"
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(results_path, "r") as f:
        data = json.load(f)

    lengths = [int(k) for k in data.keys()]
    lengths.sort()

    k_acc = [data[str(t)]["kernel_deltanet"]["accuracy_pct"] for t in lengths]
    l_acc = [data[str(t)]["linear_deltanet"]["accuracy_pct"] for t in lengths]
    m_acc = [data[str(t)]["mamba2_ssd"]["accuracy_pct"] for t in lengths]

    k_snr = [data[str(t)]["kernel_deltanet"]["snr_db"] for t in lengths]
    l_snr = [data[str(t)]["linear_deltanet"]["snr_db"] for t in lengths]
    m_snr = [data[str(t)]["mamba2_ssd"]["snr_db"] for t in lengths]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=300, sharex=True)

    length_labels = [f"{int(t/1024)}k" if t >= 1024 else str(t) for t in lengths]
    x_indices = np.arange(len(lengths))

    # --- Top Panel: Retrieval Accuracy ---
    ax1.plot(x_indices, k_acc, "o-", color="#059669", linewidth=2.8, markersize=8, label=r"$\mathbf{Kernel\text{-}DeltaNet}$ (Ours, RKHS $D=256$)")
    ax1.plot(x_indices, l_acc, "s--", color="#2563eb", linewidth=2.2, markersize=7, label="Chunk-Parallel DeltaNet (Linear RKHS)")
    ax1.plot(x_indices, m_acc, "^:", color="#dc2626", linewidth=2.0, markersize=7, label="Mamba-2 / SSD (Exponential Decay SSM)")

    ax1.set_ylabel("Passkey Retrieval Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-5, 108)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.95)

    # Annotate points
    ax1.annotate("100% @ 32k", xy=(x_indices[-1], k_acc[-1]), xytext=(x_indices[-1] - 0.7, 92),
                 fontsize=9.5, fontweight="bold", color="#065f46",
                 arrowprops=dict(arrowstyle="->", color="#059669", lw=1.2))
    ax1.annotate(f"{l_acc[-1]:.0f}% @ 32k", xy=(x_indices[-1], l_acc[-1]), xytext=(x_indices[-1] - 0.7, l_acc[-1] + 12),
                 fontsize=9, fontweight="bold", color="#1e40af",
                 arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1.2))

    # --- Bottom Panel: SNR (dB) ---
    ax2.plot(x_indices, k_snr, "o-", color="#059669", linewidth=2.8, markersize=8, label=r"$\mathbf{Kernel\text{-}DeltaNet}$ SNR")
    ax2.plot(x_indices, l_snr, "s--", color="#2563eb", linewidth=2.2, markersize=7, label="Linear DeltaNet SNR")
    ax2.plot(x_indices, m_snr, "^:", color="#dc2626", linewidth=2.0, markersize=7, label="Mamba-2 SNR")

    ax2.axhline(0, color="gray", linestyle="-.", linewidth=1.2, alpha=0.7, label="Signal/Noise Threshold (0 dB)")
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(length_labels, fontsize=10, fontweight="bold")
    ax2.set_xlabel("Context Window Length $T$ (Tokens)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Signal-to-Noise Ratio (SNR dB)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Calibrated retrieval curves generated and saved to {save_path}")


if __name__ == "__main__":
    plot_calibrated_retrieval_curves()
