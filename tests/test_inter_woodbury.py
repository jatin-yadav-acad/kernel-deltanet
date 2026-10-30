import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# tests/test_inter_woodbury.py
"""
Unit test verifying exact mathematical equivalence between sequential recurrence
and the unipotent Woodbury resolvent chunked update for Kernel-DeltaNet.
Guarantees mathematical associativity down to machine epsilon (< 1e-12).
"""
import torch


def test_inter_woodbury_equivalence():
    torch.manual_seed(42)
    B, T, d_k, d_v, C = 1, 8, 4, 4, 4

    q = torch.randn(B, T, d_k, dtype=torch.float64)
    k = torch.randn(B, T, d_k, dtype=torch.float64)
    v = torch.randn(B, T, d_v, dtype=torch.float64)
    beta = torch.rand(B, T, dtype=torch.float64) * 0.4 + 0.1
    decay = torch.rand(B, T, dtype=torch.float64) * 0.05 + 0.94

    k = k / torch.norm(k, dim=-1, keepdim=True)
    q = q / torch.norm(q, dim=-1, keepdim=True)

    # Test Chunk 1 where S_start is non-zero
    S_start = torch.randn(B, d_v, d_k, dtype=torch.float64)

    # Sequential evolution of S_start alone (v = 0):
    S_seq = S_start.clone()
    y_inter_seq = torch.zeros(B, C, d_v, dtype=torch.float64)

    for i in range(C):
        qi = q[:, i]
        ki = k[:, i]
        bi = beta[:, i, None]
        li = decay[:, i, None]

        # Readout at step i
        y_inter_seq[:, i] = torch.bmm(S_seq, qi.unsqueeze(-1)).squeeze(-1)

        # Update S_seq
        S_phi = torch.bmm(S_seq, ki.unsqueeze(-1)).squeeze(-1)
        S_seq = li.unsqueeze(-1) * (S_seq - bi.unsqueeze(-1) * torch.bmm(S_phi.unsqueeze(-1), ki.unsqueeze(1)))

    K_c = k[:, :C]  # [B, C, d_k]
    Q_c = q[:, :C]  # [B, C, d_k]
    b_c = beta[:, :C]  # [B, C]
    d_c = decay[:, :C]  # [B, C]

    cum_log_d = torch.cumsum(torch.log(d_c), dim=-1)
    gamma_0_to_C = torch.exp(cum_log_d[:, -1:, None])
    gamma_inter = torch.cat([torch.ones(B, 1, dtype=torch.float64), torch.exp(cum_log_d[:, :-1])], dim=-1)

    # Exact Unipotent Woodbury Resolvent:
    G_kk = torch.bmm(K_c, K_c.transpose(-1, -2))
    G_qk = torch.bmm(Q_c, K_c.transpose(-1, -2))
    B_mat = torch.diag_embed(b_c)
    upper_mask = torch.triu(torch.ones(C, C, dtype=torch.float64), 1).unsqueeze(0)
    lower_mask = torch.tril(torch.ones(C, C, dtype=torch.float64), -1).unsqueeze(0)
    eye_C = torch.eye(C, dtype=torch.float64).unsqueeze(0)

    # Upper triangular interaction: U_start = triu(diag(beta) @ G_kk, 1)
    U_start = torch.bmm(B_mat, G_kk) * upper_mask
    W_start = torch.bmm(torch.linalg.inv(eye_C + U_start), B_mat)

    # 1. State update: S_final = gamma_{0 -> C} [ S_start - S_start @ K_c^T @ W_start @ K_c ]
    S_chunk_final = gamma_0_to_C * (S_start - torch.bmm(torch.bmm(S_start, K_c.transpose(-1, -2)), torch.bmm(W_start, K_c)))
    diff_S = torch.norm(S_seq - S_chunk_final).item()

    # 2. Readout: Y_inter = gamma_{inter} * [ Q @ S_start^T - (Q @ K^T * lower) @ (S_start @ K^T @ W_start)^T ]
    S_Q = torch.bmm(Q_c, S_start.transpose(-1, -2))
    U = torch.bmm(torch.bmm(S_start, K_c.transpose(-1, -2)), W_start).transpose(-1, -2)
    correction = torch.bmm(G_qk * lower_mask, U)
    Y_inter_chunk = gamma_inter.unsqueeze(-1) * (S_Q - correction)
    diff_y = torch.norm(y_inter_seq - Y_inter_chunk).item()

    print(f"Sequential S_final: {torch.norm(S_seq).item():.6f}")
    print(f"Diff S_final (machine precision): {diff_S:.2e}")
    print(f"Diff y_inter (machine precision): {diff_y:.2e}")

    assert diff_S < 1e-12, f"diff_S {diff_S} exceeds machine precision tolerance"
    assert diff_y < 1e-12, f"diff_y {diff_y} exceeds machine precision tolerance"


if __name__ == "__main__":
    test_inter_woodbury_equivalence()
