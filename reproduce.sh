#!/usr/bin/env bash
set -e
echo "========================================================================"
echo "  Kernel-DeltaNet: MLSys 2026 Reproduction Harness"
echo "========================================================================"
pip install -e .
python benchmarks/reproduce_all_mlsys_results.py --task eval_all
pytest tests/ -q
echo "========================================================================"
echo "  Reproduction Complete!"
echo "========================================================================"
