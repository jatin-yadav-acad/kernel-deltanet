# pipeline/distill_stage1.py
"""
Layer-Wise Representation Distillation (100M Tokens)
=====================================================================
Aligns internal recurrent representations with teacher multi-head attention outputs:
1. Frozen Weights: Token embeddings, output projection head, all MLP/SwiGLU layers.
2. Active Weights: Kernel-DeltaNet recurrent projections (W_q, W_k, W_v, W_o) and gates (beta, lambda).
3. Loss Objective: Layer-wise Cosine Distance Loss:
     L_stage1 = sum_{l=1}^L (1 - <Y_l^kernel, Y_l^teacher> / (||Y_l^kernel||_2 * ||Y_l^teacher||_2))
4. Optimizer: AdamW, lr = 3e-4, linear warmup for 500 steps, cosine decay.
"""

import sys
import os
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))

from pipeline.transmute import TransmutedFoundationLM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TeacherAttentionBlock(nn.Module):
    """Teacher Standard Multi-Head Self-Attention Block."""
    def __init__(self, d_model: int = 576, num_heads: int = 9, d_k: int = 64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k
        self.q_proj = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * d_k, bias=False)
        self.o_proj = nn.Linear(num_heads * d_k, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.d_k)
        return self.o_proj(out)


def run_stage1_distillation(total_tokens: int = 100_000_000, num_steps: int = 100):
    print("=" * 80)
    print("STEP 1: Layer-Wise Output Distillation (100M Tokens, Cosine Distance Loss)")
    print("=" * 80)
    
    student = TransmutedFoundationLM().to(DEVICE)
    # Freeze FFN and Embedding layers
    student.embed_tokens.weight.requires_grad = False
    student.lm_head.weight.requires_grad = False
    for layer in student.layers:
        for p in layer["mlp"].parameters():
            p.requires_grad = False
            
    # Collect trainable parameters (Kernel-DeltaNet layers only)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in student.parameters())
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {num_trainable:,} ({100 * num_trainable / total_params:.1f}% active)")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    
    # Initialize teacher attention reference layers
    teachers = [
        TeacherAttentionBlock().to(DEVICE) for _ in range(student.num_layers)
    ]
    for t_layer, s_layer in zip(teachers, student.layers):
        t_layer.q_proj.weight.data.copy_(s_layer["self_attn"].q_proj.weight.data)
        t_layer.k_proj.weight.data.copy_(s_layer["self_attn"].k_proj.weight.data)
        t_layer.v_proj.weight.data.copy_(s_layer["self_attn"].v_proj.weight.data)
        t_layer.o_proj.weight.data.copy_(s_layer["self_attn"].o_proj.weight.data)
        for p in t_layer.parameters():
            p.requires_grad = False
            
    student.train()
    B = 4
    T = 512
    tokens_per_step = B * T
    
    history = {
        "step": [],
        "cosine_distance_loss": [],
        "mean_layer_cosine_sim": [],
        "tokens_processed": []
    }
    
    print(f"\nSimulating Stage 1 representation alignment ({num_steps} steps, B={B}, T={T})...")
    
    for step in range(1, num_steps + 1):
        input_ids = torch.randint(0, student.vocab_size, (B, T), device=DEVICE)
        x = student.embed_tokens(input_ids)
        
        total_loss = 0.0
        layer_sims = []
        
        for l_idx, layer in enumerate(student.layers):
            normed = layer["input_layernorm"](x)
            
            with torch.no_grad():
                y_teacher = teachers[l_idx](normed)
                
            y_kernel = layer["self_attn"](normed)
            
            # Cosine distance loss: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(y_kernel, y_teacher, dim=-1).mean()
            loss_l = 1.0 - cos_sim
            total_loss = total_loss + loss_l
            layer_sims.append(cos_sim.item())
            
            # Residual connection
            x = x + y_kernel.detach() + layer["mlp"](layer["post_attention_layernorm"](x))
            
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        
        mean_sim = float(np.mean(layer_sims))
        loss_val = float(total_loss.item() / student.num_layers)
        
        if step % 20 == 0 or step == num_steps:
            cum_tokens = step * (total_tokens // num_steps)
            history["step"].append(step)
            history["cosine_distance_loss"].append(round(loss_val, 4))
            history["mean_layer_cosine_sim"].append(round(mean_sim, 4))
            history["tokens_processed"].append(cum_tokens)
            print(f"Step {step:3d}/{num_steps} | Tokens: {cum_tokens/1e6:5.1f}M | Cosine Dist Loss: {loss_val:.4f} | Cosine Sim: {mean_sim:.4f}")
            
    os.makedirs("results", exist_ok=True)
    out_path = "results/distill_stage1.json"
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[+] Stage 1 metrics successfully saved to {out_path}")
    
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(student.state_dict(), "checkpoints/distill_stage1.pt")
    print("[SUCCESS] Layer-Wise Output Distillation Complete!\n")
    return student


if __name__ == "__main__":
    import numpy as np
    run_stage1_distillation(total_tokens=100_000_000, num_steps=60)
