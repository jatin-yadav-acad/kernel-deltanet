"""
Kernel-DeltaNet Engine
Module: training.optimizer_utils
==================================================
Optimizer and Learning Rate Scheduler Configuration.
"""

import math
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.01,
    learning_rate: float = 1e-3,
    spectral_lr_multiplier: float = 0.5,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8
) -> optim.Optimizer:
    """
    Configures AdamW optimizer with parameter-grouping:
    - 2D weight matrices (linear projections, embeddings): standard lr, weight decay applied.
    - 1D bias and normalization scale vectors: standard lr, zero weight decay.
    - Spectral bandwidth parameters (log_gamma, logit_decay, raw_eta): specialized learning rate.
    """
    decay_params = []
    no_decay_params = []
    spectral_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "log_gamma" in name or "logit_decay" in name or "raw_eta" in name:
            spectral_params.append(param)
        elif param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay, "lr": learning_rate},
        {"params": no_decay_params, "weight_decay": 0.0, "lr": learning_rate},
        {"params": spectral_params, "weight_decay": 0.0, "lr": learning_rate * spectral_lr_multiplier}
    ]

    optimizer = optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    return optimizer


class CosineWarmupLRScheduler:
    """
    Cosine Annealing Learning Rate Scheduler with Linear Warmup.
    """
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        step = self.current_step

        if step < self.warmup_steps:
            # Linear warmup
            ratio = float(step) / float(max(1, self.warmup_steps))
        elif step > self.max_steps:
            ratio = self.min_lr_ratio
        else:
            # Cosine decay
            progress = float(step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            ratio = self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * ratio

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]
