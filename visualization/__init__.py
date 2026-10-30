"""
Kernel-DeltaNet Engine
Package: visualization
========================================
Publication-quality visualization and figure generation suite.
"""

from .plot_throughput_cliff import plot_throughput_cliff
from .plot_mqar_capacity_curves import plot_mqar_capacity_curves
from .plot_generation_memory_flat import plot_generation_memory_flat
from .plot_scaling_laws import plot_scaling_laws
from .generate_all_figures import generate_all_figures

__all__ = [
    "plot_throughput_cliff",
    "plot_mqar_capacity_curves",
    "plot_generation_memory_flat",
    "plot_scaling_laws",
    "generate_all_figures",
]
