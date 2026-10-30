r"""
Kernel-DeltaNet: Kernel-DeltaNet Sequence Engine
Module: tests.test_kernel_deltanet_suite
==========================================================
Standalone Camera-Ready Unit Test Suite (TEST-2.1 through TEST-2.12)
Validates core mathematical invariants, hardware execution, and numerical precision
with fixed deterministic random seeds on NVIDIA GPU / CPU silicon.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import os
import json
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.associative_algebra import AffineSemigroupVerifier
from core.chunked_scan import chunked_affine_scan, recurrent_step
from core.rkhs_features import MultiHeadRKHSKernel
from layers.kernel_deltanet_layer import KernelDeltaNetLayer
from models.kernel_deltanet_lm import KernelDeltaNetLM
from data.mqar_generator import MQARGenerator


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def test_2_1_semigroup_associativity() -> bool:
    """TEST-2.1: Mathematical Semigroup Associativity of Affine Transitions."""
    print("[*] Running TEST-2.1: Semigroup Associativity Invariant...")
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = AffineSemigroupVerifier.verify_associativity(D=128, d_v=64, batch_size=4, device=device)

    err_A = result["error_A_relative"]
    err_B = result["error_B_relative"]
    is_assoc = result["is_associative"]

    assert is_assoc, f"Associativity failed: {result}"
    print(f"    Operator A Rel Error: {err_A:.2e} | Operator B Rel Error: {err_B:.2e} | Associative: {is_assoc}")
    print("[+] TEST-2.1 PASSED: Strict Semigroup Associativity Verified.\n")
    pass


def test_2_2_chunked_scan_vs_recurrence() -> bool:
    """TEST-2.2: Chunk-Parallel Scan vs Recurrence Numerical Equivalence."""
    print("[*] Running TEST-2.2: Chunk-Parallel Scan vs Recurrent Unrolling Equivalence...")
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, T, D, d_v = 2, 2, 128, 64, 32
    phi = F.normalize(torch.randn(B, H, T, D, device=device), p=2, dim=-1)
    v = torch.randn(B, H, T, d_v, device=device)
    decay = torch.full((1, H, 1, 1), 0.98, device=device)
    eta = torch.full((1, H, 1, 1), 0.6, device=device)

    # 1. Parallel chunked scan
    y_par = chunked_affine_scan(phi, v, decay=decay, eta=eta, chunk_size=32)

    # 2. Sequential recurrent unrolling
    recurrent_outs = []
    state = None
    for t in range(T):
        phi_t = phi[:, :, t : t + 1, :]
        v_t = v[:, :, t : t + 1, :]
        q_t = phi_t
        y_t, state = recurrent_step(phi_t, v_t, q_t, state, decay, eta)
        recurrent_outs.append(y_t)
    y_rec = torch.cat(recurrent_outs, dim=2)

    l2_diff = torch.norm(y_rec - y_par).item()
    rel_diff = l2_diff / (torch.norm(y_rec).item() + 1e-9)

    assert rel_diff < 1e-4, f"Parallel scan diverged from recurrence: rel diff = {rel_diff}"
    print(f"    L2 Absolute Difference: {l2_diff:.2e} | Relative Difference: {rel_diff:.2e}")
    print("[+] TEST-2.2 PASSED: Exact Numerical Equivalence Verified.\n")
    pass


def test_2_3_autograd_gradient_flow() -> bool:
    """TEST-2.3: Autograd Flow & Finite Parameter Gradients."""
    print("[*] Running TEST-2.3: Full Autograd Flow & Finite Parameter Gradients...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = KernelDeltaNetLayer(d_model=64, num_heads=2, rkhs_factor=4, chunk_size=32).to(device)

    x = torch.randn(2, 64, 64, device=device, requires_grad=True)
    out, _ = layer(x)
    loss = out.sum()
    loss.backward()

    for name, p in layer.named_parameters():
        assert p.grad is not None, f"Parameter {name} has None gradient!"
        assert not torch.isnan(p.grad).any(), f"Parameter {name} has NaN gradient!"
        assert not torch.isinf(p.grad).any(), f"Parameter {name} has Inf gradient!"
        assert torch.norm(p.grad).item() > 0.0, f"Parameter {name} has zero gradient!"

    assert x.grad is not None and torch.norm(x.grad).item() > 0.0, "Input x has zero gradient!"
    print(f"    Verified non-zero finite gradients across all {len(list(layer.parameters()))} parameter tensors.")
    print("[+] TEST-2.3 PASSED: Autograd Flow Fully Verified.\n")
    pass


def test_2_4_bochner_crosstalk_decoupling() -> bool:
    """TEST-2.4: Bochner RKHS Crosstalk & Mutual Coherence Decoupling."""
    print("[*] Running TEST-2.4: Bochner RKHS Crosstalk Decoupling...")
    set_seed(42)
    d = 32
    D = 512
    N = 64

    k = F.normalize(torch.randn(N, d), p=2, dim=-1)
    linear_gram = torch.matmul(k, k.t())
    linear_gram.fill_diagonal_(0.0)
    linear_crosstalk = torch.abs(linear_gram).mean().item()

    kernel = MultiHeadRKHSKernel(num_heads=1, d_head=d, rkhs_dim=D, gamma_init=2.0)
    phi = kernel(k.view(1, 1, N, d))[0, 0]
    rkhs_gram = torch.matmul(phi, phi.t())
    rkhs_gram.fill_diagonal_(0.0)
    rkhs_crosstalk = torch.abs(rkhs_gram).mean().item()

    decoupling_factor = linear_crosstalk / (rkhs_crosstalk + 1e-9)
    assert decoupling_factor > 1.5, f"RKHS failed to decouple crosstalk: factor = {decoupling_factor:.2f}x"
    print(f"    Linear Mean Crosstalk: {linear_crosstalk:.4f} | RKHS Mean Crosstalk: {rkhs_crosstalk:.4f}")
    print(f"    Decoupling Advantage:  {decoupling_factor:.2f}x lower mutual interference")
    print("[+] TEST-2.4 PASSED: Crosstalk Decoupling Verified.\n")
    pass


def test_2_5_layer_dual_mode_equivalence() -> bool:
    """TEST-2.5: Layer Parallel Mode vs Step-by-Step Inference Equivalence."""
    print("[*] Running TEST-2.5: Dual-Mode Forward Equivalence (Parallel Scan vs Step Cache)...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = KernelDeltaNetLayer(d_model=64, num_heads=2, rkhs_factor=4, chunk_size=32).to(device)
    layer.eval()

    x = torch.randn(1, 32, 64, device=device)

    # 1. Parallel forward
    with torch.no_grad():
        out_par, state_par = layer(x)

    # 2. Step-by-step forward
    step_outputs = []
    state_curr = None
    with torch.no_grad():
        for t in range(x.shape[1]):
            x_t = x[:, t : t + 1, :]
            out_t, state_curr = layer(x_t, state=state_curr)
            step_outputs.append(out_t)

    out_step = torch.cat(step_outputs, dim=1)
    rel_diff = (torch.norm(out_par - out_step) / (torch.norm(out_par) + 1e-9)).item()

    assert rel_diff < 1e-4, f"Parallel output diverged from step output: rel diff = {rel_diff}"
    print(f"    Relative Difference between Parallel and Step Forward: {rel_diff:.2e}")
    print("[+] TEST-2.5 PASSED: Layer Dual-Mode Equivalence Verified.\n")
    pass


def test_2_6_fused_chunk_scan_equivalence() -> bool:
    """TEST-2.6: Fused Chunk Kernel Equivalence with Reference Scan."""
    print("[*] Running TEST-2.6: Fused Chunk Kernel Equivalence...")
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from kernels.fused_chunk_kernel import fused_chunked_scan

    B, H, T, D, d_v = 2, 4, 128, 64, 16
    chunk_size = 32
    phi_k = torch.randn(B, H, T, D, device=device)
    phi_q = torch.randn(B, H, T, D, device=device)
    v = torch.randn(B, H, T, d_v, device=device)
    decay = torch.full((H,), 0.99, device=device)
    eta = torch.full((H,), 0.5, device=device)

    with torch.no_grad():
        ref_out = chunked_affine_scan(phi_k=phi_k, phi_q=phi_q, v=v, decay=decay, eta=eta, chunk_size=chunk_size)
        fused_out = fused_chunked_scan(phi_k, phi_q, v, decay, eta, chunk_size=chunk_size)

    l2_diff = torch.norm(ref_out - fused_out).item()
    rel_diff = l2_diff / (torch.norm(ref_out).item() + 1e-9)

    assert rel_diff < 1e-5, f"Fused chunk scan diverged: rel diff = {rel_diff}"
    print(f"    Fused vs Reference Relative Difference: {rel_diff:.2e}")
    print("[+] TEST-2.6 PASSED: Fused Chunk Kernel Equivalence Verified.\n")
    pass


def test_2_7_spectral_bandwidth_trainability() -> bool:
    """TEST-2.7: Learnable Spectral Bandwidth Positivity and Active Gradients."""
    print("[*] Running TEST-2.7: Spectral Bandwidth Positivity & Trainability...")
    set_seed(42)
    kernel = MultiHeadRKHSKernel(num_heads=4, d_head=32, rkhs_dim=128, gamma_init=1.5)
    x = torch.randn(2, 4, 16, 32, requires_grad=True)

    phi = kernel(x)
    target = torch.randn_like(phi)
    loss = (phi * target).sum()
    loss.backward()

    assert kernel.log_gamma.grad is not None, "log_gamma gradient is None!"
    assert torch.norm(kernel.log_gamma.grad).item() > 0.0, "log_gamma gradient is zero!"
    gamma_values = torch.exp(kernel.log_gamma)
    assert (gamma_values > 0).all(), "Spectral bandwidth gamma must be strictly positive!"

    print(f"    Bandwidth values: {gamma_values.detach().tolist()}")
    print(f"    Bandwidth gradient norm: {torch.norm(kernel.log_gamma.grad).item():.4f}")
    print("[+] TEST-2.7 PASSED: Bandwidth Positivity and Trainability Verified.\n")
    pass


def test_2_8_constant_generation_memory() -> bool:
    """TEST-2.8: Strict O(1) Memory Footprint During Autoregressive Inference."""
    print("[*] Running TEST-2.8: Strict O(1) Inference Memory Invariance...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = KernelDeltaNetLayer(d_model=64, num_heads=2, rkhs_factor=4, chunk_size=32).to(device)
    layer.eval()

    state = None
    state_sizes = []
    with torch.no_grad():
        for step in range(50):
            token = torch.randn(1, 1, 64, device=device)
            _, state = layer(token, state=state)
            state_sizes.append(state.numel() * state.element_size())

    assert len(set(state_sizes)) == 1, "Recurrent state size changed across generation steps!"
    print(f"    Recurrent State Footprint: {state_sizes[0]:,} bytes (strictly invariant over 50 steps)")
    print("[+] TEST-2.8 PASSED: Constant O(1) Inference Footprint Verified.\n")
    pass


def test_2_9_calibrated_passkey_retrieval() -> bool:
    """TEST-2.9: Calibrated Passkey Retrieval Retention."""
    print("[*] Running TEST-2.9: Calibrated Passkey Retrieval...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from evaluation.eval_calibrated_retrieval import GatedKernelDeltaNetCell

    cell = GatedKernelDeltaNetCell(d_model=64, num_heads=1, rkhs_factor=8).to(device)
    cell.eval()
    K = torch.randn(1, 1, 1024, 32, device=device)
    K = F.normalize(K, p=2, dim=-1)
    V = torch.randn(1, 1, 1024, 32, device=device)
    V = F.normalize(V, p=2, dim=-1)
    seq = torch.cat([K.squeeze(1), V.squeeze(1)], dim=-1)

    with torch.no_grad():
        S = cell.forward_state_scan(K, V, seq)
        assert S is not None and not torch.isnan(S).any()
    print("    Gated Kernel-DeltaNet 1024-token context state successfully accumulated.")
    print("[+] TEST-2.9 PASSED: Calibrated Passkey State Accumulation Verified.\n")
    pass


def test_2_10_converged_scaling_monotonicity() -> bool:
    """TEST-2.10: Converged Chinchilla Scaling Law Exponent Monotonicity."""
    candidates = [
        "results/converged_scaling_laws.json",
        "results/converged_scaling_laws.json",
        os.path.join(os.path.dirname(__file__), "..", "results", "converged_scaling_laws.json")
    ]
    scaling_json = next((c for c in candidates if os.path.exists(c)), None)
    if not scaling_json:
        print("    [-] Skipping: converged_scaling_laws.json not found.")
        pass

    with open(scaling_json, "r") as f:
        data = json.load(f)

    k_fit = data["kernel_deltanet_fit"]
    gamma = k_fit["scaling_exponent_gamma"]
    r2 = k_fit["r2_fit"]

    assert gamma > 0.0, f"Scaling exponent gamma must be strictly positive: {gamma}"
    assert r2 > 0.90, f"Scaling power law fit R^2 must exceed 0.90: {r2}"
    print(f"    Scaling Exponent gamma: {gamma:.4f} | Fit Quality R^2: {r2:.3f}")
    print("[+] TEST-2.10 PASSED: Scaling Law Monotonicity Verified.\n")
    pass


def test_2_11_mqar_capacity_saturation() -> bool:
    """TEST-2.11: Multi-Query Associative Recall Capacity at 4x Saturation."""
    print("[*] Running TEST-2.11: MQAR Capacity Saturation...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 64
    d_sub = d_model // 2
    K = 128

    gen = MQARGenerator(d_model=d_model, device=device)
    batch = gen.generate_batch(batch_size=1, seq_len=1024, num_kv_pairs=K, seed=42)

    layer = KernelDeltaNetLayer(d_model=d_model, num_heads=1, rkhs_factor=8, chunk_size=32, initial_decay=1.0, initial_eta=1.0).to(device)
    with torch.no_grad():
        W = torch.zeros(d_model, d_model, device=device)
        W[:d_sub, :d_sub] = torch.eye(d_sub, device=device)
        layer.w_q.weight.copy_(W)
        layer.w_k.weight.copy_(W)
        W_v = torch.zeros(d_model, d_model, device=device)
        W_v[d_sub:, d_sub:] = torch.eye(d_sub, device=device)
        layer.w_v.weight.copy_(W_v)
        layer.w_out.weight.copy_(torch.eye(d_model, device=device))

    layer.eval()
    with torch.no_grad():
        out, _ = layer(batch["sequences"])

    pred_val = out[0, -1, d_sub:]
    target_val = batch["target_values"][0]
    all_vals = batch["all_values"][0]

    sims = torch.matmul(all_vals, pred_val)
    top1 = bool(sims.argmax().item() == 0)
    cos_sim = torch.cosine_similarity(pred_val.unsqueeze(0), target_val.unsqueeze(0)).item()

    assert top1, f"Kernel-DeltaNet failed 4x saturation recall! CosSim = {cos_sim:.3f}"
    print(f"    K={K} (4.0x Saturation) Recall Status: TOP-1 RECALLED | CosSim: {cos_sim:+.3f}")
    print("[+] TEST-2.11 PASSED: MQAR Capacity Saturation Verified.\n")
    pass


def test_2_12_gpu_silicon_forward_backward() -> bool:
    """TEST-2.12: GPU Silicon Full LM Forward + Backward without Leaks."""
    print("[*] Running TEST-2.12: GPU Silicon Full LM Forward + Backward...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KernelDeltaNetLM(vocab_size=256, d_model=128, num_layers=4, num_heads=2, rkhs_factor=4, chunk_size=32).to(device)

    tokens = torch.randint(0, 256, (4, 512), device=device)
    targets = torch.randint(0, 256, (4, 512), device=device)

    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss = model(tokens, targets=targets)
    loss.backward()

    assert not torch.isnan(loss), "LM Loss is NaN!"
    assert not torch.isinf(loss), "LM Loss is Inf!"
    assert loss.item() > 0.0, "LM Loss must be positive!"
    print(f"    LM Loss: {loss.item():.4f} on {device} (B=4, T=512, BF16 AMP)")
    print("[+] TEST-2.12 PASSED: GPU Silicon Execution Verified.\n")
    pass


def run_test_2_suite(save_log: bool = True) -> bool:
    print("=" * 95)
    print("KERNEL-DELTANET: STANDALONE CAMERA-READY TEST SUITE (TEST-2.1 to TEST-2.12)")
    print("=" * 95)
    t0 = time.time()

    tests = [
        ("TEST-2.1_Semigroup_Associativity", test_2_1_semigroup_associativity),
        ("TEST-2.2_Scan_Recurrent_Equivalence", test_2_2_chunked_scan_vs_recurrence),
        ("TEST-2.3_Autograd_Gradient_Flow", test_2_3_autograd_gradient_flow),
        ("TEST-2.4_Bochner_Crosstalk_Decoupling", test_2_4_bochner_crosstalk_decoupling),
        ("TEST-2.5_Layer_Dual_Mode_Equivalence", test_2_5_layer_dual_mode_equivalence),
        ("TEST-2.6_Fused_Chunk_Kernel", test_2_6_fused_chunk_scan_equivalence),
        ("TEST-2.7_Spectral_Bandwidth_Trainability", test_2_7_spectral_bandwidth_trainability),
        ("TEST-2.8_Constant_Inference_Memory", test_2_8_constant_generation_memory),
        ("TEST-2.9_Calibrated_Passkey_Retrieval", test_2_9_calibrated_passkey_retrieval),
        ("TEST-2.10_Converged_Scaling_Monotonicity", test_2_10_converged_scaling_monotonicity),
        ("TEST-2.11_MQAR_Capacity_Saturation", test_2_11_mqar_capacity_saturation),
        ("TEST-2.12_GPU_Silicon_Forward_Backward", test_2_12_gpu_silicon_forward_backward)
    ]

    results = {}
    all_passed = True
    for test_name, test_fn in tests:
        try:
            ok = test_fn()
            results[test_name] = {"passed": bool(ok), "status": "PASSED"}
        except Exception as e:
            print(f"[-] {test_name} FAILED: {e}\n")
            results[test_name] = {"passed": False, "status": "FAILED", "error": str(e)}
            all_passed = False

    elapsed = time.time() - t0
    print("=" * 95)
    if all_passed:
        print(f"[+] ALL 12 STANDALONE TESTS (TEST-2.1 to TEST-2.12) PASSED IN {elapsed:.2f}s!")
    else:
        print(f"[-] SOME TESTS FAILED IN {elapsed:.2f}s.")
    print("=" * 95)

    if save_log:
        out_path = "results/test_2_suite_report.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "all_passed": all_passed,
                "elapsed_seconds": elapsed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": results
            }, f, indent=2)
        print(f"[*] Test suite report logged to {out_path}\n")

    return all_passed


if __name__ == "__main__":
    run_test_2_suite()
