# Kernel-DeltaNet: Breaking the Client-Side Long-Context Serving Wall via In-Register Hardware-Algorithm Co-Design

[![MLSys 2026](https://img.shields.io/badge/MLSys-2026-blue.svg)](https://mlsys.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-77%2F77%20Passing-brightgreen.svg)](tests/)

Official anonymous artifact repository for the MLSys 2026 submission:  
**"Kernel-DeltaNet: Breaking the Client-Side Long-Context Serving Wall via In-Register Hardware-Algorithm Co-Design"**

---

## Overview

While datacenter clusters deploy multi-GPU nodes with TB/s HBM to absorb Key-Value (KV) cache expansion, client-side long-context serving on commodity workstations is bounded by strict **16 GB VRAM limits**. Standard Softmax Attention consumes $>44\,\mathrm{GB}$ KV cache at $64\mathrm{k}$ tokens ($B=32$), triggering fatal Out-Of-Memory (OOM) failures.

**Kernel-DeltaNet** reconciles associative capacity, mathematical associativity, and memory hierarchy efficiency via hardware-algorithm co-design:
1. **Mathematical Semigroup Associativity**: Maps representations into a Reproducing Kernel Hilbert Space (RKHS) via spherical normalization and structured Fast Walsh-Hadamard Transforms (FWHT), expanding recurrent state capacity to $D=512$ while strictly preserving affine semigroup associativity for parallel chunk scans.
2. **In-Register Hardware Co-Design**: Formulates the 3-stage FWHT butterfly directly within the Warp Register File ($\le 124$ registers/thread), elevating operational intensity by $\mathbf{34.3\times}$ to $\mathbf{428.7\,\mathrm{FLOP/Byte}}$ ($>216.0\,\mathrm{FLOP/Byte}$ hardware ridge point).
3. **Asymmetric 20:2 Hybrid Topology**: Maps $90.9\%$ of layers (20 layers) entirely into on-chip registers and SRAM ($0\,\mathrm{MB}$ KV cache), while $9.1\%$ (2 boundary anchors) fit inside the physical $48\,\mathrm{MB}$ L2 cache. Slashes $\mathbf{90.9\%\text{--}96.5\%}$ KV cache memory while matching teacher accuracy ($97.08\%$ retention; $100.0\%$ 64k passkey recall).
4. **Physical Silicon Performance**: Measured on consumer NVIDIA GeForce RTX 5070 Ti (16 GB GDDR7, $896\,\mathrm{GB/s}$), Kernel-DeltaNet delivers an isolated attention prefill speedup of $\mathbf{2.50\times}$ at $64\mathrm{k}$ ($161.26\,\mathrm{ms}$ vs $403.59\,\mathrm{ms}$ for FlashAttention-2 / PyTorch SDPA; $2.94\times$ over FA-2 eager), sustaining $B=32\text{--}64$ concurrent $64\mathrm{k}$ serving streams under a strict 16 GB budget.

---

## Silicon Benchmark Highlights (NVIDIA RTX 5070 Ti)

### Single Attention Layer Prefill Execution Latency (Table 1)

| Context Length ($T$) | 4,096 | 8,192 | 16,384 | 32,768 | 65,536 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| FlashAttention-2 (PyTorch SDPA) | 1.64 ms | 6.27 ms | 26.25 ms | 97.94 ms | 403.59 ms |
| FlashAttention-2 (Eager) | 1.82 ms | 7.15 ms | 33.20 ms | 134.5 ms | 473.5 ms |
| Parallel TTT-MLP ($b=64$) | 11.8 ms | 21.5 ms | 38.9 ms | 78.4 ms | 159.1 ms |
| Linear DeltaNet ($d_k=64$) | 1.77 ms | 3.83 ms | 8.35 ms | 17.20 ms | 34.72 ms |
| **Kernel-DeltaNet ($D=512$, Ours)** | **11.09 ms** | **20.99 ms** | **41.43 ms** | **81.93 ms** | **161.26 ms** |
| **Speedup vs SDPA** | $0.15\times$ | $0.30\times$ | $0.63\times$ | $\mathbf{1.20\times}$ | $\mathbf{2.50\times}$ |
| **Speedup vs FA-2 Eager** | $0.16\times$ | $0.34\times$ | $0.80\times$ | $\mathbf{1.64\times}$ | $\mathbf{2.94\times}$ |

*Note: At short sequences ($T \le 16\mathrm{k}$), FlashAttention-2 achieves lower latency from in-SRAM fusion with minimal framework dispatch. Beyond $25\mathrm{k}$ tokens, quadratic attention explodes while Kernel-DeltaNet scales strictly linearly, delivering a $2.50\times$ speedup at $64\mathrm{k}$ (crossover at $T^* \approx 25{,}000$ tokens).*

---

## Repository Structure

```
kernel-deltanet-MLSYS/
├── kernels/                  # Core recurrent & Triton kernels
│   ├── tiled_triton_engine.py    # Streaming & tiled chunk scan Triton kernels (SM120)
│   ├── kernel_deltanet_fused.py  # Reference PyTorch/cuSOLVER resolvent engine
│   ├── fwht.py / fwht_cuda.cpp   # In-register Fast Walsh-Hadamard Transform
│   ├── in_register_fwht.cu       # Bare-metal CUDA warp butterfly kernel
│   ├── vectorized_triangular_solve.py # In-register unipotent causal solver
│   └── blelloch_scan.py          # Parallel associative Blelloch tree scan
├── serving/                  # Production serving engine & runtime
│   ├── vllm_ragged_batching.py   # Ragged chunk prefill & decode scheduler
│   ├── vllm_hybrid_model_runner.py # Unified PagedAttention & flat slab memory pool
│   ├── demo_end_to_end_generation.py # End-to-end token generation harness
│   └── vllm_openai_server.py     # OpenAI-compatible API server & telemetry
├── layers/                   # Recurrent layers & hybrid blocks
│   ├── kernel_deltanet_layer.py  # Primary multi-head sequence layer
│   └── hybrid_kernel_deltanet_block.py # Pre-LayerNorm Transformer block
├── models/                   # Architecture definitions & upcycling bridge
│   ├── kernel_deltanet_lm.py     # Causal autoregressive language model
│   ├── long_context_kernel_deltanet.py # Streaming million-token engine
│   └── upcycling_llama_bridge.py # Surgical attention transmutation bridge
├── benchmarks/               # Complete MLSys benchmark & profiling suite
│   ├── reproduce_all_mlsys_results.py  # Master reproduction harness
│   ├── run_mlsys_serving_benchmark.py  # Concurrency, TTFT, ITL macrobenchmarks
│   ├── run_real_benchmarks.py          # Physical silicon latency microbenchmarks
│   ├── test_vllm_dropin_serving.py     # Drop-in vLLM memory plugin test
│   └── run_hardware_profiling_dump.py  # Microarchitectural roofline model
├── evaluation/               # Long-context & downstream evaluation harnesses
│   ├── evaluate_full_downstream.py     # Full PIQA, ARC, HellaSwag evaluation
│   ├── eval_dense_niah_heatmap_64k.py  # 64k dense Needle-in-a-Haystack grid
│   └── eval_mqar_capacity.py          # Multi-query associative recall stress tests
├── core/                     # Mathematical RKHS projections & Lie algebra ops
├── baselines/                # Comparative baselines (FA-2, TTT-MLP, Mamba-2)
├── tests/                    # 77 unit tests verifying math & hardware invariants
├── results/                  # Verified silicon measurement logs (JSON)
├── setup.py                  # Standard packaging definition
├── pyproject.toml            # Build specification
├── requirements.txt          # Python dependencies
└── README.md                 # This documentation
```

---

## Quickstart & Installation

### Requirements
- **OS**: Linux or Windows
- **Python**: $\ge 3.10$
- **PyTorch**: $\ge 2.2.0$ with CUDA 12.x / 13.x
- **GPU**: NVIDIA GPU with $\ge 8\,\mathrm{GB}$ VRAM (evaluated on NVIDIA GeForce RTX 5070 Ti)

### Setup
```bash
git clone https://anonymous.4open.science/r/kernel-deltanet.git
cd kernel-deltanet
pip install -e .
```

---

## One-Click Reproduction

To run the complete MLSys evaluation suite and reproduce all tables and figures:

```bash
python benchmarks/reproduce_all_mlsys_results.py --task eval_all
```

Or on Windows / Linux:
```bash
# Linux
bash reproduce.sh

# Windows
reproduce.bat
```

### Running Unit Tests
All 77 mathematical and hardware invariant unit tests can be executed via:
```bash
pytest tests/ -v
```

---

## Accompanying Paper & Supplementary Materials
This codebase accompanies the MLSys 2026 submission:
- **Main Conference Paper**: (10 pages main body + 1 page references)
- **Supplementary Material**: (33 pages containing formal proofs, hardware SASS register allocations, and microarchitectural models)
---

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
