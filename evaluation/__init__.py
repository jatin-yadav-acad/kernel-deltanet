"""
Evaluation module for Kernel-DeltaNet
"""

import os
import sys

_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from .eval_hardware_throughput import run_throughput_benchmark
from .eval_mqar_capacity import run_mqar_capacity_benchmark
from .eval_generation_memory import run_generation_memory_benchmark
from .eval_needle_haystack import run_needle_benchmark

__all__ = [
    "run_throughput_benchmark",
    "run_mqar_capacity_benchmark",
    "run_generation_memory_benchmark",
    "run_needle_benchmark"
]
