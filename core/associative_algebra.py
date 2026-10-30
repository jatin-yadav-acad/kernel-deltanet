"""
Kernel-DeltaNet Engine
Module: core.associative_algebra
==================================================
Theoretical and Algebraic Properties of the Affine Semigroup in RKHS.

Mathematical Invariants:
1. Semigroup Composition:
   Let M = R^{D x D} x R^{d_v x D}. Define the binary operation:
     (A_1, B_1) o (A_2, B_2) = (A_1 A_2, B_1 A_2 + B_2)
   Then (M, o) forms a monoid with identity element (I, 0).
   The state at time T satisfies:
     (A_{1:T}, B_{1:T}) = (A_1, B_1) o (A_2, B_2) o ... o (A_T, B_T)

2. Lyapunov Contraction & Spectral Stability:
   For state update S_t = S_{t-1} A_t + B_t, where
     A_t = lambda * (I - eta * phi_t * phi_t^T)
   The singular values of A_t satisfy:
     sigma_1(A_t) <= lambda * max(1, |1 - eta * ||phi_t||^2|)
   When ||phi_t|| = 1, eta in (0, 1], and lambda in (0, 1]:
     ||A_t||_2 <= lambda <= 1.0
   Thus the state operator is strictly non-expansive, guaranteeing Lyapunov stability.

3. Mutual Coherence & Welch Bound:
   The mutual coherence of a dictionary Phi = [phi_1, ..., phi_K] is:
     mu(Phi) = max_{i != j} |<phi_i, phi_j>| / (||phi_i|| * ||phi_j||)
   By the Welch bound, in linear space R^d:
     mu_linear >= sqrt((K - d) / (d * (K - 1))) ~ 1/sqrt(d)
   In RKHS feature space R^D with Gaussian RBF bandwidth gamma:
     mu_rkhs <= exp(-2 * gamma * (1 - mu_linear))
   Demonstrating exponential noise suppression as gamma scales.
"""

import math
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
import numpy as np


class AffineSemigroupVerifier:
    """
    Verifies mathematical invariants of the Affine Associative Semigroup on GPU/CPU.
    """
    @staticmethod
    def compose(
        A1: torch.Tensor, B1: torch.Tensor,
        A2: torch.Tensor, B2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Binary composition (A_1, B_1) o (A_2, B_2) = (A_1 A_2, B_1 A_2 + B_2).
        Args:
            A1, A2: (..., D, D)
            B1, B2: (..., d_v, D)
        Returns:
            A12, B12: (..., D, D) and (..., d_v, D)
        """
        A12 = torch.matmul(A1, A2)
        B12 = torch.matmul(B1, A2) + B2
        return A12, B12

    @staticmethod
    def verify_associativity(D: int = 64, d_v: int = 32, batch_size: int = 4, device: str = "cpu") -> Dict[str, float]:
        """
        Tests: [ (T1 o T2) o T3 ] == [ T1 o (T2 o T3) ]
        """
        torch.manual_seed(42)
        # Random transitions
        A1 = torch.randn(batch_size, D, D, device=device) * 0.1
        B1 = torch.randn(batch_size, d_v, D, device=device)
        A2 = torch.randn(batch_size, D, D, device=device) * 0.1
        B2 = torch.randn(batch_size, d_v, D, device=device)
        A3 = torch.randn(batch_size, D, D, device=device) * 0.1
        B3 = torch.randn(batch_size, d_v, D, device=device)

        # Left grouping: (T1 o T2) o T3
        A12, B12 = AffineSemigroupVerifier.compose(A1, B1, A2, B2)
        A_left, B_left = AffineSemigroupVerifier.compose(A12, B12, A3, B3)

        # Right grouping: T1 o (T2 o T3)
        A23, B23 = AffineSemigroupVerifier.compose(A2, B2, A3, B3)
        A_right, B_right = AffineSemigroupVerifier.compose(A1, B1, A23, B23)

        err_A = torch.norm(A_left - A_right).item() / (torch.norm(A_left).item() + 1e-9)
        err_B = torch.norm(B_left - B_right).item() / (torch.norm(B_left).item() + 1e-9)

        return {
            "error_A_relative": float(err_A),
            "error_B_relative": float(err_B),
            "is_associative": bool(err_A < 1e-6 and err_B < 1e-6)
        }

    @staticmethod
    def spectral_stability_analysis(
        phi: torch.Tensor,
        eta: float = 0.5,
        decay: float = 0.99
    ) -> Dict[str, float]:
        """
        Analyzes the spectral norm and condition number of A_t = lambda * (I - eta * phi * phi^T).
        Args:
            phi: (T, D) normalized features
        """
        T, D = phi.shape
        norms = []
        conds = []

        I = torch.eye(D, device=phi.device)
        for t in range(min(T, 100)):
            p = phi[t:t+1] # (1, D)
            At = decay * (I - eta * torch.matmul(p.t(), p))
            s = torch.linalg.svdvals(At)
            norms.append(s[0].item())
            conds.append((s[0] / (s[-1] + 1e-9)).item())

        return {
            "max_spectral_norm": float(np.max(norms)),
            "mean_spectral_norm": float(np.mean(norms)),
            "max_condition_number": float(np.max(conds)),
            "is_contractive": bool(np.max(norms) <= 1.0)
        }

    @staticmethod
    def mutual_coherence_analysis(
        keys: torch.Tensor,
        phi_keys: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compares linear mutual coherence vs RKHS mutual coherence.
        Args:
            keys: (K, d) normalized linear keys
            phi_keys: (K, D) normalized RKHS keys
        """
        K = keys.shape[0]
        # Linear Gram
        G_lin = torch.abs(torch.matmul(keys, keys.t()))
        G_lin.fill_diagonal_(0.0)
        mu_lin = torch.max(G_lin).item()
        mean_crosstalk_lin = torch.sum(G_lin).item() / (K * (K - 1))

        # RKHS Gram
        G_rkhs = torch.abs(torch.matmul(phi_keys, phi_keys.t()))
        G_rkhs.fill_diagonal_(0.0)
        mu_rkhs = torch.max(G_rkhs).item()
        mean_crosstalk_rkhs = torch.sum(G_rkhs).item() / (K * (K - 1))

        decoupling_ratio = (mean_crosstalk_lin + 1e-9) / (mean_crosstalk_rkhs + 1e-9)

        return {
            "linear_max_coherence": float(mu_lin),
            "linear_mean_crosstalk": float(mean_crosstalk_lin),
            "rkhs_max_coherence": float(mu_rkhs),
            "rkhs_mean_crosstalk": float(mean_crosstalk_rkhs),
            "noise_decoupling_factor": float(decoupling_ratio)
        }
