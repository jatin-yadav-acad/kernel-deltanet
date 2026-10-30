import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# tests/test_fused_equivalence.py
import torch
import numpy as np

def verify_algebra():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running algebraic verification on {device}...")
    
    B, T, d_k, d_v, D, C = 2, 128, 64, 128, 512, 64
    tau = (d_k) ** 0.25 # 2.828427
    
    q = torch.randn(B, T, d_k, device=device, dtype=torch.float64) # test in float64 first for exact algebraic identity
    k = torch.randn(B, T, d_k, device=device, dtype=torch.float64)
    v = torch.randn(B, T, d_v, device=device, dtype=torch.float64)
    beta = torch.rand(B, T, device=device, dtype=torch.float64) * 0.4 + 0.1
    decay = torch.rand(B, T, device=device, dtype=torch.float64) * 0.05 + 0.94
    
    Omega = torch.randn(D, d_k, device=device, dtype=torch.float64)
    b = torch.rand(D, device=device, dtype=torch.float64) * 2 * np.pi
    
    # Lift features
    phi_k = np.sqrt(2.0 / D) * torch.cos(torch.matmul(k, Omega.T) / tau + b)
    phi_k = phi_k / torch.norm(phi_k, dim=-1, keepdim=True)
    
    phi_q = np.sqrt(2.0 / D) * torch.cos(torch.matmul(q, Omega.T) / tau + b)
    phi_q = phi_q / torch.norm(phi_q, dim=-1, keepdim=True)
    
    # Sequential Recurrence:
    # y_t = S_{t-1} @ phi_q_t
    # S_t = lambda_t * S_{t-1} * (I - beta_t phi_k_t phi_k_t^T) + beta_t * v_t * phi_k_t^T
    S_seq = torch.zeros(B, d_v, D, device=device, dtype=torch.float64)
    y_seq = torch.zeros(B, T, d_v, device=device, dtype=torch.float64)
    S_chunk_ends_seq = []
    
    for t in range(T):
        pk = phi_k[:, t] # [B, D]
        pq = phi_q[:, t] # [B, D]
        vt = v[:, t]     # [B, d_v]
        bt = beta[:, t:t+1] # [B, 1]
        lt = decay[:, t:t+1] # [B, 1]
        
        # Readout
        y_seq[:, t] = torch.bmm(S_seq, pq.unsqueeze(-1)).squeeze(-1)
        
        # Update
        S_phi = torch.bmm(S_seq, pk.unsqueeze(-1)).squeeze(-1) # [B, d_v]
        S_seq = lt.unsqueeze(-1) * (S_seq - bt.unsqueeze(-1) * torch.bmm(S_phi.unsqueeze(-1), pk.unsqueeze(1))) + bt.unsqueeze(-1) * torch.bmm(vt.unsqueeze(-1), pk.unsqueeze(1))
        
        if (t + 1) % C == 0:
            S_chunk_ends_seq.append(S_seq.clone())
            
    # Chunked Recurrence
    num_chunks = T // C
    y_chunk = torch.zeros(B, T, d_v, device=device, dtype=torch.float64)
    S_start = torch.zeros(B, d_v, D, device=device, dtype=torch.float64)
    S_chunk_ends_chunk = []
    
    for c in range(num_chunks):
        c_start = c * C
        c_end = (c + 1) * C
        
        K_c = phi_k[:, c_start:c_end] # [B, C, D]
        Q_c = phi_q[:, c_start:c_end] # [B, C, D]
        V_c = v[:, c_start:c_end]     # [B, C, d_v]
        b_c = beta[:, c_start:c_end]  # [B, C]
        d_c = decay[:, c_start:c_end] # [B, C]
        
        # Decay cumulative products in chunk:
        # gamma_{i -> j} = prod_{m=i+1}^j lambda_m (for i < j)
        log_d = torch.log(d_c)
        cum_log_d = torch.cumsum(log_d, dim=-1) # [B, C]
        
        # 1. Inter-chunk Readout:
        # For token i in {0, ..., C-1}:
        # y_inter[i] = gamma_{0 -> i-1} * S_start @ Q_c[i]
        # with gamma_{0 -> -1} = 1.0
        gamma_inter = torch.cat([torch.ones(B, 1, device=device, dtype=torch.float64), torch.exp(cum_log_d[:, :-1])], dim=-1) # [B, C]
        # S_start @ Q_c.T -> [B, d_v, C] -> transpose to [B, C, d_v]
        S_Q = torch.bmm(S_start, Q_c.transpose(-1, -2)).transpose(-1, -2) # [B, C, d_v]
        Y_inter = gamma_inter.unsqueeze(-1) * S_Q # [B, C, d_v]
        
        # 2. Gram matrix G_kk = K_c @ K_c^T: [B, C, C]
        G_kk = torch.bmm(K_c, K_c.transpose(-1, -2))
        
        # 3. Upper-triangular coupling matrix U_bar:
        # (U_bar)_{ij} = - beta_j * (K_c_i . K_c_j) for i < j
        upper_mask = torch.triu(torch.ones(C, C, device=device, dtype=torch.float64), 1)
        U_bar = - (b_c.unsqueeze(1) * G_kk) * upper_mask # [B, C, C]
        
        # Invert (I - U_bar)
        I_C = torch.eye(C, device=device, dtype=torch.float64).unsqueeze(0)
        I_minus_U_inv = torch.linalg.inv(I_C - U_bar)
        
        # Resolvents:
        # W_tilde = (I - U_bar)^(-1) @ diag(beta_1 ... beta_C)
        W_tilde = I_minus_U_inv * b_c.unsqueeze(1) # broadcast beta across columns
        
        # W_c = diag(beta_i * gamma_{i -> C}) @ (I - U_bar)^(-1)
        gamma_i_to_C = torch.exp(cum_log_d[:, -1:] - cum_log_d) # [B, C]
        W_c = (b_c * gamma_i_to_C).unsqueeze(-1) * I_minus_U_inv # broadcast along rows
        
        # 4. State transition:
        # S_next = gamma_{0 -> C} [ S_start - S_start @ K_c^T @ W_tilde @ K_c ] + V_c^T @ W_c @ K_c
        gamma_0_to_C = torch.exp(cum_log_d[:, -1:, None]) # [B, 1, 1]
        S_K = torch.bmm(S_start, K_c.transpose(-1, -2)) # [B, d_v, C]
        S_K_W = torch.bmm(S_K, W_tilde)                 # [B, d_v, C]
        S_decayed = gamma_0_to_C * (S_start - torch.bmm(S_K_W, K_c))
        
        # Delta S_c = V_c^T @ W_c @ K_c
        V_W = torch.bmm(V_c.transpose(-1, -2), W_c) # [B, d_v, C]
        delta_S = torch.bmm(V_W, K_c)               # [B, d_v, D]
        
        S_next = S_decayed + delta_S
        S_chunk_ends_chunk.append(S_next)
        
        # 5. Intra-chunk Readout:
        # Lower-triangular causal key operator L_c:
        # (L_c)_{ij} = - beta_i * gamma_{j -> i} * (K_c_i . K_c_j) for i > j
        # Note: gamma_{j -> i} = exp(cum_log_d[i] - cum_log_d[j])
        log_d_diff = cum_log_d.unsqueeze(-1) - cum_log_d.unsqueeze(-2) # [B, C, C]: diff[i, j] = log_d[i] - log_d[j]
        gamma_pairwise = torch.exp(log_d_diff) # [B, C, C]
        lower_mask = torch.tril(torch.ones(C, C, device=device, dtype=torch.float64), -1)
        
        L_c = - (b_c.unsqueeze(-1) * gamma_pairwise * G_kk) * lower_mask # [B, C, C]
        I_minus_L_inv = torch.linalg.inv(I_C - L_c)
        
        # Transformed values V_tilde = (I - L_c)^(-1) @ (diag(beta) @ V_c)
        beta_V = b_c.unsqueeze(-1) * V_c # [B, C, d_v]
        V_tilde = torch.bmm(I_minus_L_inv, beta_V) # [B, C, d_v]
        
        # Intra retrieval:
        # Y_intra = [ (Q_c @ K_c^T) * P_decay * M_strict_lower ] @ V_tilde
        # where P_decay[i, j] = gamma_{j -> i-1} = exp(cum_log_d[i-1] - cum_log_d[j]) for i > j
        G_qk = torch.bmm(Q_c, K_c.transpose(-1, -2)) # [B, C, C]
        
        cum_log_d_shift = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.float64), cum_log_d[:, :-1]], dim=-1) # [B, C]
        P_decay = torch.exp(cum_log_d_shift.unsqueeze(-1) - cum_log_d.unsqueeze(-2)) # [B, C, C]
        
        A_intra = G_qk * P_decay * lower_mask # [B, C, C]
        Y_intra = torch.bmm(A_intra, V_tilde) # [B, C, d_v]
        
        # Total chunk output:
        Y_c = Y_inter + Y_intra
        y_chunk[:, c_start:c_end] = Y_c
        
        S_start = S_next

    # Compare y_seq and y_chunk
    diff_y = torch.max(torch.norm(y_chunk - y_seq, dim=-1) / (torch.norm(y_seq, dim=-1) + 1e-12)).item()
    print(f"Max relative output error (float64): {diff_y:.6e}")
    
    # Compare final states
    diff_S = torch.max(torch.norm(S_chunk_ends_chunk[-1] - S_chunk_ends_seq[-1], p='fro') / (torch.norm(S_chunk_ends_seq[-1], p='fro') + 1e-12)).item()
    print(f"Max state relative Frobenius error (float64): {diff_S:.6e}")
    
    assert diff_y < 1e-10, f"Output discrepancy in float64: {diff_y}"
    assert diff_S < 1e-10, f"State discrepancy in float64: {diff_S}"
    print("[PASS] Exact mathematical identity confirmed down to machine precision!")

if __name__ == "__main__":
    verify_algebra()
