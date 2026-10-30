@echo off
echo ========================================================================
echo   Kernel-DeltaNet: MLSys 2026 Reproduction Harness
echo ========================================================================
python -m pip install -e .
python benchmarks/reproduce_all_mlsys_results.py --task eval_all
python -m pytest tests/ -q
echo ========================================================================
echo   Reproduction Complete!
echo ========================================================================
