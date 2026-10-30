import os
import json
import time
import torch
import torch.nn as nn
import numpy as np

from layers.kernel_deltanet_layer import KernelDeltaNetLayer
from baselines.ttt_linear_deltanet import TTTLinearLayer
from data.mqar_generator import MQARGenerator
from evaluation.eval_mqar_capacity import set_subspace_aligned_projections

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LinearDeltaNetState512(nn.Module):
    """
    Linear DeltaNet with key dimension expanded to dk=512 via a linear projection.
    State matrix is S in R^{dv x 512}, exactly matching the state footprint of Kernel-DeltaNet.
    """
    def __init__(self, d_model=64, d_expanded=512):
        super().__init__()
        self.d_model = d_model
        self.d_sub = d_model // 2
        self.d_expanded = d_expanded
        
        # Linear projection into R^512
        self.w_q = nn.Linear(self.d_sub, d_expanded, bias=False)
        self.w_k = nn.Linear(self.d_sub, d_expanded, bias=False)
        self.w_v = nn.Linear(self.d_sub, self.d_sub, bias=False)
        self.w_out = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Orthogonal initialization for fair comparison
        nn.init.orthogonal_(self.w_q.weight)
        nn.init.orthogonal_(self.w_k.weight)
        nn.init.eye_(self.w_v.weight)
        nn.init.eye_(self.w_out.weight)
        
    def forward(self, x):
        B, T, _ = x.shape
        q_raw = x[..., :self.d_sub]
        k_raw = x[..., :self.d_sub]
        v = self.w_v(x[..., self.d_sub:])
        
        Q = self.w_q(q_raw)
        K = self.w_k(k_raw)
        
        Q = Q / (torch.norm(Q, dim=-1, keepdim=True) + 1e-6)
        K = K / (torch.norm(K, dim=-1, keepdim=True) + 1e-6)
        
        S = torch.zeros(B, self.d_sub, self.d_expanded, device=x.device)
        outputs = []
        for t in range(T):
            kt = K[:, t, :].unsqueeze(1)
            vt = v[:, t, :].unsqueeze(1)
            qt = Q[:, t, :].unsqueeze(1)
            
            yt = torch.matmul(S, qt.transpose(-1, -2)).squeeze(-1)
            outputs.append(yt)
            
            Sk = torch.matmul(S, kt.transpose(-1, -2))
            S = S - torch.matmul(Sk, kt) + torch.matmul(vt.transpose(-1, -2), kt)
            
        out_v = torch.stack(outputs, dim=1)
        out = torch.cat([torch.zeros_like(q_raw), out_v], dim=-1)
        return self.w_out(out), S

def main():
    d_model = 64
    d_sub = d_model // 2
    configs = [
        (512, 16),
        (1024, 64),
        (2048, 128),
        (4096, 256),
        (4096, 512)
    ]

    lin32 = TTTLinearLayer(d_model=d_model, chunk_size=32, initial_decay=1.0, initial_eta=1.0).to(device)
    set_subspace_aligned_projections(lin32, d_model=d_model)

    lin512 = LinearDeltaNetState512(d_model=d_model, d_expanded=512).to(device)
    
    kernel512 = KernelDeltaNetLayer(d_model=d_model, num_heads=1, rkhs_factor=8, chunk_size=32, initial_decay=1.0, initial_eta=1.0).to(device)
    set_subspace_aligned_projections(kernel512, d_model=d_model)

    models = {
        'Linear DeltaNet (dk=32)': lin32,
        'Linear DeltaNet (dk=512, Linear)': lin512,
        'Kernel-DeltaNet (D=512, RKHS)': kernel512
    }

    generator = MQARGenerator(d_model=d_model, device=device)
    num_trials = 25

    print(f'| K (Pairs) | Linear Saturation | Linear DeltaNet (dk=32) | Linear DeltaNet (dk=512) | Kernel-DeltaNet (D=512) |')
    print('|---|---|---|---|---|')

    results_table = {}

    for T, K in configs:
        sat = K / d_sub
        row_str = f'| K={K} | {sat:.1f}x | '
        results_table[f'K_{K}'] = {}
        for name, m in models.items():
            m.eval()
            accs = []
            snrs = []
            for trial in range(num_trials):
                batch = generator.generate_batch(batch_size=1, seq_len=T, num_kv_pairs=K, target_turn='earliest', seed=400 + trial)
                seq = batch['sequences']
                target_val = batch['target_values'][0]
                all_vals = batch['all_values'][0]
                with torch.no_grad():
                    out, _ = m(seq)
                pred_val = out[0, -1, d_sub:]
                sims = torch.matmul(all_vals, pred_val)
                acc = 1.0 if sims.argmax().item() == 0 else 0.0
                accs.append(acc)
                crosstalk = pred_val - target_val
                snr = 10.0 * np.log10((torch.norm(target_val)**2 / (torch.norm(crosstalk)**2 + 1e-9)).item())
                snrs.append(snr)
            mean_acc = float(np.mean(accs) * 100.0)
            mean_snr = float(np.mean(snrs))
            results_table[f'K_{K}'][name] = {'acc': mean_acc, 'snr': mean_snr}
            row_str += f'{mean_acc:.1f}% ({mean_snr:+.1f} dB) | '
        print(row_str)

    os.makedirs('results', exist_ok=True)
    with open('results/state_dimension_control_mqar.json', 'w') as f:
        json.dump(results_table, f, indent=2)
    print('\nSaved results to kernel_deltanet/results/state_dimension_control_mqar.json')

if __name__ == '__main__':
    main()
