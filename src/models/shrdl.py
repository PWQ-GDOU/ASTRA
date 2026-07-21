"""
SHRDL: Self-Supervised Health Representation Decomposition Learning

Based on: Wang et al. "Self-Supervised Health Representation Decomposition
Based on Contrast Learning", Reliability Engineering & System Safety, 2023.

Architecture:
  Input → [WDL (Working Condition Decomposition)] 
        → [NDL (Noise Decomposition)]
        → [Transformer Layer]
        → Health Representation z
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Tuple, Optional


class WorkingConditionDecomposition(nn.Module):
    """
    WDL: Remove working condition effects from sensor signals using
    cross-attention between condition flow and performance flow.
    
    Key idea: Use working condition as Query, sensor signal as Key/Value.
    The attention output represents "how much of the sensor signal is
    explained by the working condition" → subtract it.
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Cross-attention: condition (query) attends to performance (key/value)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Condition signal encoder (compress to same dimension)
        self.cond_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, h_perf: torch.Tensor, h_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_perf: Performance signal flow (B, T, D) - sensor data
            h_cond: Working condition flow (B, C, D) - condition info
        
        Returns:
            h_perf_new: Condition-removed performance signal (B, T, D)
        """
        B, T, D = h_perf.shape
        
        # Encode condition signal
        h_cond = self.cond_encoder(h_cond)  # (B, C, D)
        
        # Multi-head cross-attention
        Q = self.W_q(h_cond).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(h_perf).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(h_perf).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = self.head_dim ** 0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # (B, n_heads, C, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, D)
        
        # Cross-attention output: mean-pool over condition dimension
        attn_output = attn_output.mean(dim=1, keepdim=True)  # (B, 1, D)
        attn_output = attn_output.expand(-1, T, -1)  # (B, T, D)
        attn_output = self.W_o(attn_output)
        
        # Residual: subtract condition-explained component
        h_perf_new = self.layer_norm(h_perf - attn_output)
        
        return h_perf_new


class NoiseDecomposition(nn.Module):
    """
    NDL: Remove noise from signals using multi-scale moving average
    with self-attention-based adaptive filtering.
    """
    
    def __init__(self, d_model: int, window_sizes: list = [2, 4, 8, 16, 32],
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.window_sizes = window_sizes
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head self-attention for adaptive window fusion
        self.W_q = nn.Linear(d_model * len(window_sizes), d_model, bias=False)
        self.W_k = nn.Linear(d_model * len(window_sizes), d_model, bias=False)
        self.W_v = nn.Linear(d_model * len(window_sizes), d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Input signal (B, T, D)
        Returns:
            h_clean: Denoised signal (B, T, D)
        """
        B, T, D = h.shape
        
        # Multi-scale average pooling with padding to preserve length
        pooled_features = []
        for ws in self.window_sizes:
            ks = min(ws, T)
            # Pad input to maintain output size same as input
            pad_left = (ks - 1) // 2
            pad_right = ks - 1 - pad_left
            
            h_padded = F.pad(h.transpose(1, 2), (pad_left, pad_right), mode='replicate')
            pooled = F.avg_pool1d(
                h_padded,  # (B, D, T + pad)
                kernel_size=ks,
                stride=1,
                padding=0,
            ).transpose(1, 2)  # (B, T, D)
            pooled_features.append(pooled)
        
        # Concatenate all window outputs
        multi_scale = torch.cat(pooled_features, dim=-1)  # (B, T, D * n_windows)
        
        # Self-attention for adaptive fusion
        Q = self.W_q(multi_scale).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(multi_scale).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(multi_scale).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        scale = self.head_dim ** 0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        attn_output = self.W_o(attn_output)
        
        # Residual: subtract noise component
        h_clean = self.layer_norm(h - attn_output)
        
        return h_clean


class ADNBlock(nn.Module):
    """
    Attention-based Decomposition Network (ADN) block.
    
    Combines WDL + NDL + Transformer layer.
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, d_feedforward: int = 256,
                 window_sizes: list = [2, 4, 8, 16, 32], dropout: float = 0.1):
        super().__init__()
        
        self.wdl = WorkingConditionDecomposition(d_model, n_heads, dropout)
        self.ndl = NoiseDecomposition(d_model, window_sizes, n_heads, dropout)
        
        # Standard Transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
    
    def forward(self, h: torch.Tensor, h_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h: Performance signal (B, T, D)
            h_cond: Working condition signal (B, C, D), optional
        
        Returns:
            h_s: Health-related representation (B, T, D)
        """
        if h_cond is not None:
            h = self.wdl(h, h_cond)
        h = self.ndl(h)
        h = self.transformer_layer(h)
        return h


class MoCoMemoryBank:
    """
    Momentum Contrast (MoCo) memory bank for efficient contrastive learning.
    
    Maintains a queue of encoded features from the momentum encoder
    to serve as negative samples.
    """
    
    def __init__(self, feature_dim: int, queue_size: int = 4096):
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        
        self.register_buffer = None  # Will be set when used with PyTorch
        
        # Queue stored on CPU by default
        self.queue = torch.zeros(queue_size, feature_dim)
        self.queue_ptr = 0
        self.queue_full = False
    
    @torch.no_grad()
    def enqueue(self, keys: torch.Tensor):
        """Add keys to the queue."""
        batch_size = keys.shape[0]
        
        # Handle batch larger than queue
        ptr_end = self.queue_ptr + batch_size
        
        if ptr_end <= self.queue_size:
            self.queue[self.queue_ptr:ptr_end] = keys.cpu()
        else:
            # Wrap around
            overflow = ptr_end - self.queue_size
            self.queue[self.queue_ptr:] = keys[:batch_size - overflow].cpu()
            self.queue[:overflow] = keys[batch_size - overflow:].cpu()
            self.queue_full = True
        
        self.queue_ptr = (self.queue_ptr + batch_size) % self.queue_size
        if self.queue_ptr == 0:
            self.queue_full = True
    
    def get_queue(self) -> torch.Tensor:
        """Get current queue contents."""
        if self.queue_full:
            return self.queue.detach()
        return self.queue[:self.queue_ptr].detach()


class SHRDL(nn.Module):
    """
    SHRDL: Self-Supervised Health Representation Decomposition Learning.
    
    Architecture:
      - Input projection
      - N stacked ADN blocks (WDL + NDL + Transformer)
      - Projection head (for contrastive learning)
      - Momentum encoder (for MoCo)
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_adn_blocks: Number of ADN stacking blocks
        d_feedforward: Feedforward dimension in Transformer
        n_features: Number of input sensor channels
        feature_dim: Output embedding dimension for contrastive learning
        momentum: MoCo momentum coefficient
        memory_bank_size: Size of MoCo memory queue
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_adn_blocks: int = 2,
        d_feedforward: int = 256,
        n_features: int = 21,
        feature_dim: int = 64,
        window_sizes: list = [2, 4, 8, 16, 32],
        dropout: float = 0.1,
        momentum: float = 0.999,
        memory_bank_size: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        
        # ADN blocks
        self.adn_blocks = nn.ModuleList([
            ADNBlock(d_model, n_heads, d_feedforward, window_sizes, dropout)
            for _ in range(n_adn_blocks)
        ])
        
        # Projection head (for contrastive learning)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, feature_dim),
        )
        
        # Condition encoder (optional, for WDL)
        self.cond_encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Momentum encoder (for MoCo)
        self.momentum_encoder = copy.deepcopy(self._get_encoder())
        self._freeze_momentum_encoder()
        
        # Memory bank
        self.memory_bank = MoCoMemoryBank(feature_dim, memory_bank_size)
    
    def _get_encoder(self):
        """Return a module that encodes input to projection features."""
        return nn.Sequential(
            self.input_proj,
            *self.adn_blocks,
            self.projection_head,
        )
    
    @torch.no_grad()
    def _freeze_momentum_encoder(self):
        """Disable gradients for momentum encoder."""
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        """Update momentum encoder parameters."""
        for param_q, param_k in zip(
            self._get_encoder().parameters(),
            self.momentum_encoder.parameters()
        ):
            param_k.data = self.momentum * param_k.data + \
                           (1 - self.momentum) * param_q.data
    
    def encode(self, x: torch.Tensor, h_cond: Optional[torch.Tensor] = None,
               project: bool = False) -> torch.Tensor:
        """
        Encode input to health representation.
        
        Args:
            x: Input (B, T, F)
            h_cond: Working condition (B, C, F), optional
            project: Whether to apply projection head
        
        Returns:
            Encoded features (B, T, D) or (B, feature_dim)
        """
        h = self.input_proj(x)
        
        for block in self.adn_blocks:
            h = block(h, h_cond)
        
        if project:
            # Global average pooling + projection head
            h_pool = h.mean(dim=1)  # (B, D)
            h = self.projection_head(h_pool)
        
        return h
    
    @torch.no_grad()
    def encode_momentum(self, x: torch.Tensor) -> torch.Tensor:
        """Encode using momentum encoder (with global pooling)."""
        self.momentum_encoder.eval()
        # Manually apply encoder parts: input_proj -> ADN blocks -> pool -> projection head
        h = self.momentum_encoder[0](x)  # input_proj
        for block_idx in range(1, len(self.momentum_encoder) - 1):
            h = self.momentum_encoder[block_idx](h)  # ADN blocks
        h_pool = h.mean(dim=1)  # Global average pooling
        h = self.momentum_encoder[-1](h_pool)  # projection_head
        return h
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                h_cond: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for contrastive learning.
        
        Args:
            x1, x2: Two augmented views of the same sample
            h_cond: Working condition signal
        
        Returns:
            q: Query features (from online encoder)
            k1, k2: Key features (from momentum encoder, two views)
        """
        # Online encoder: query
        q = self.encode(x1, h_cond, project=True)
        q = F.normalize(q, dim=-1)
        
        # Momentum encoder: keys
        k1 = self.encode_momentum(x1)
        k1 = F.normalize(k1, dim=-1)
        k2 = self.encode_momentum(x2)
        k2 = F.normalize(k2, dim=-1)
        
        # Update momentum encoder
        self._momentum_update()
        
        # Enqueue keys to memory bank
        self.memory_bank.enqueue(k2)
        
        return q, k1, k2


class CIMCLLoss(nn.Module):
    """
    Cycle Information Modified Contrastive Loss (CIMCL).
    
    Extends InfoNCE with cycle-difference-based dynamic weights:
    - Larger cycle gap → higher weight (harder negative, push harder)
    - Smaller cycle gap → lower weight (semantically similar, gentler)
    
    weight = 1 / (1 + beta * exp(-gamma * |cycle_i - cycle_j|))
    """
    
    def __init__(self, temperature: float = 0.07, beta: float = 400.0,
                 gamma: float = 100.0):
        super().__init__()
        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, q: torch.Tensor, k_plus: torch.Tensor,
                queue: torch.Tensor, cycle_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            q: Query features (B, D), normalized
            k_plus: Positive key features (B, D), normalized
            queue: Memory queue (Q, D), normalized
            cycle_indices: Cycle indices for each query sample (B,)
        
        Returns:
            CIMCL loss (scalar)
        """
        B = q.shape[0]
        
        # Positive logits
        l_pos = torch.sum(q * k_plus, dim=-1, keepdim=True) / self.temperature
        
        # Negative logits (from memory queue)
        l_neg = torch.matmul(q, queue.T) / self.temperature  # (B, Q)
        
        # Apply cycle-based weights to negatives if available
        if cycle_indices is not None and hasattr(self, 'queue_cycles'):
            weight_matrix = 1.0 / (1.0 + self.beta * torch.exp(
                -self.gamma * torch.abs(
                    cycle_indices.unsqueeze(1) - self.queue_cycles.unsqueeze(0).to(q.device)
                )
            ))
            l_neg = l_neg * weight_matrix
        
        # InfoNCE loss
        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=q.device)
        
        return F.cross_entropy(logits, labels)
    
    def update_queue_cycles(self, cycles: torch.Tensor):
        """Store cycle indices for the memory queue."""
        self.queue_cycles = cycles


def ntxent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent loss for self-supervised pretraining (PEUDA stage 0).
    
    Normalized temperature-scaled cross-entropy loss.
    Used for target-domain-only contrastive pretraining.
    """
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    
    # Concatenate all embeddings
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    
    # Compute similarity matrix
    sim = torch.matmul(z, z.T) / temperature  # (2B, 2B)
    
    # Mask out self-similarity
    mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, float('-inf'))
    
    # Positive pairs: (i, i+B) and (i+B, i)
    pos_indices = torch.cat([
        torch.arange(B, 2*B),
        torch.arange(B),
    ]).to(z.device)
    
    # Cross-entropy
    labels = pos_indices
    loss = F.cross_entropy(sim, labels)
    
    return loss
