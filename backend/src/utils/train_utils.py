"""
Training utilities: early stopping, warmup scheduler, seed setting.
"""

import torch
import numpy as np
import random
import os
from typing import Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4,
                 mode: str = 'min', verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch}, "
                          f"best score: {self.best_score:.6f} at epoch {self.best_epoch}")
        
        return self.should_stop


class WarmupScheduler:
    """Learning rate warmup scheduler."""
    
    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_epochs: int, base_lr: float,
                 total_epochs: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.current_epoch = 0
    
    def step(self):
        """Update learning rate based on current epoch."""
        self.current_epoch += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self) -> float:
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            return self.min_lr + (self.base_lr - self.min_lr) * \
                   (1 + np.cos(np.pi * progress)) / 2
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class AverageMeter:
    """Track running averages."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
