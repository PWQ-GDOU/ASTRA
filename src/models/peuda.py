"""
PEUDA: Pre-training Enhanced Unsupervised Contrastive Domain Adaptation

Based on: Li H. et al. "Pre-training Enhanced Unsupervised Contrastive
Domain Adaptation for Industrial Equipment RUL Prediction",
Advanced Engineering Informatics, 2024.

Architecture:
  1. Dual parallel time-frequency feature extractor
  2. Frequency-domain enhanced adversarial training (Gradient Reversal)
  3. Momentum Contrastive Learning (MCL) module for negative transfer prevention
  
Training:
  Stage 0: Target-only SSL pretraining (NT-Xent)
  Stage 1: Source+Target adversarial domain adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from typing import Tuple, Optional


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial domain adaptation.
    
    Forward: identity
    Backward: gradient × (-lambda)
    """
    
    @staticmethod
    def forward(ctx, x, lambda_val=1.0):
        ctx.lambda_val = lambda_val
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_val, None


class GRL(nn.Module):
    """Gradient Reversal Layer as nn.Module."""
    
    def __init__(self, lambda_val: float = 1.0):
        super().__init__()
        self.lambda_val = lambda_val
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_val)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time series."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeStreamEncoder(nn.Module):
    """Time-domain feature encoder using Transformer."""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x


class FrequencyStreamEncoder(nn.Module):
    """Frequency-domain feature encoder."""
    
    def __init__(self, d_model: int, n_heads: int, n_layers: int,
                 d_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T_freq, D)"""
        x = self.transformer(x)
        x = self.layer_norm(x)
        return x


class DualStreamEncoder(nn.Module):
    """
    Dual parallel time-frequency feature extractor (PEUDA).
    
    Time stream: processes raw time-domain signals
    Frequency stream: processes FFT-transformed signals (PEUDA key innovation)
    
    The frequency stream forces the model to focus on phase information,
    which is more transferable across domains.
    """
    
    def __init__(self, n_features: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, n_fft: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_fft = n_fft
        
        # Input projections
        self.time_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.freq_proj = nn.Sequential(
            nn.Linear(n_features, d_model),  # Map each freq bin's features
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        
        # Encoders
        self.time_encoder = TimeStreamEncoder(
            d_model, n_heads, n_layers, d_model * 2, dropout
        )
        self.freq_encoder = FrequencyStreamEncoder(
            d_model, n_heads, n_layers, d_model * 2, dropout
        )
        
        # Fusion layer: combine time and frequency features
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
    
    def to_frequency(self, x: torch.Tensor) -> torch.Tensor:
        """Convert time-domain signal to frequency domain."""
        B, T, F = x.shape
        x_fft = torch.fft.rfft(x, n=self.n_fft, dim=1)
        # Convert complex to magnitude
        x_mag = torch.abs(x_fft)  # (B, n_fft//2+1, F)
        return x_mag
    
    def forward(self, x_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_time: Time-domain input (B, T, F)
        
        Returns:
            fused_features: Combined time-freq features (B, T, d_model)
        """
        B, T, F = x_time.shape
        
        # Time stream
        h_time = self.time_proj(x_time)  # (B, T, D)
        h_time = self.time_encoder(h_time)
        h_time_pool = h_time.mean(dim=1)  # (B, D)
        
        # Frequency stream
        x_freq = self.to_frequency(x_time)  # (B, n_freq, F)
        h_freq = self.freq_proj(x_freq)  # (B, n_freq, D)
        h_freq = self.freq_encoder(h_freq)
        h_freq_pool = h_freq.mean(dim=1)  # (B, D)
        
        # Fusion
        h_fused = torch.cat([h_time_pool, h_freq_pool], dim=-1)  # (B, 2D)
        h_fused = self.fusion(h_fused)  # (B, D)
        
        return h_fused, h_time, h_freq


class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial training."""
    
    def __init__(self, d_model: int, hidden_dim: int = 256, n_domains: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_domains if n_domains > 2 else 1),
        )
        self.n_domains = n_domains
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RULPredictor(nn.Module):
    """RUL regression head."""
    
    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive RUL output
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PEUDA(nn.Module):
    """
    PEUDA: Pre-training Enhanced Unsupervised Contrastive Domain Adaptation.
    
    Full pipeline:
      Stage 0: Target-domain SSL with NT-Xent loss (contrastive pretraining)
      Stage 1: Joint adversarial training with:
        - MSE loss on source RUL predictions
        - Adversarial domain confusion loss (GRL-based)
        - Momentum contrastive constraint (MCL) to prevent negative transfer
    """
    
    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        n_fft: int = 64,
        dropout: float = 0.1,
        mcl_momentum: float = 0.999,
        mcl_queue_size: int = 2048,
        mcl_temperature: float = 0.07,
    ):
        super().__init__()
        self.d_model = d_model
        self.mcl_momentum = mcl_momentum
        self.mcl_temperature = mcl_temperature
        
        # Dual-stream encoder (shared between source and target)
        self.encoder = DualStreamEncoder(
            n_features, d_model, n_heads, n_layers, n_fft, dropout
        )
        
        # RUL predictor (trained on source, applied to target)
        self.rul_predictor = RULPredictor(d_model)
        
        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(d_model)
        self.grl = GRL(lambda_val=1.0)
        
        # SSL projection head (used during Stage 0 target pretraining)
        self.ssl_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 64),
        )
        
        # MCL momentum encoder and queue
        self.momentum_encoder = copy.deepcopy(self.encoder)
        self._freeze_momentum_encoder()
        
        self.register_buffer('mcl_queue', torch.zeros(mcl_queue_size, d_model))
        self.register_buffer('mcl_queue_ptr', torch.zeros(1, dtype=torch.long))
        self.mcl_queue_size = mcl_queue_size
    
    @torch.no_grad()
    def _freeze_momentum_encoder(self):
        for param in self.momentum_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        for param_q, param_k in zip(
            self.encoder.parameters(),
            self.momentum_encoder.parameters()
        ):
            param_k.data = self.mcl_momentum * param_k.data + \
                           (1 - self.mcl_momentum) * param_q.data
    
    @torch.no_grad()
    def _mcl_enqueue(self, keys: torch.Tensor):
        """Enqueue features to MCL memory queue."""
        batch_size = keys.shape[0]
        ptr = int(self.mcl_queue_ptr)
        
        if ptr + batch_size > self.mcl_queue_size:
            overflow = ptr + batch_size - self.mcl_queue_size
            self.mcl_queue[ptr:] = keys[:batch_size - overflow]
            self.mcl_queue[:overflow] = keys[batch_size - overflow:]
            self.mcl_queue_ptr[0] = overflow
        else:
            self.mcl_queue[ptr:ptr + batch_size] = keys
            self.mcl_queue_ptr[0] = ptr + batch_size
    
    def get_mcl_queue(self) -> torch.Tensor:
        """Get valid entries from MCL queue."""
        ptr = int(self.mcl_queue_ptr)
        if ptr == 0:
            return self.mcl_queue
        return self.mcl_queue[:ptr]
    
    def forward_ssl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stage 0: Target-domain SSL forward pass.
        Returns features for NT-Xent loss.
        """
        fused, _, _ = self.encoder(x)
        z = self.ssl_projection(fused)
        return F.normalize(z, dim=-1)
    
    def forward_adaptation(
        self,
        source_x: torch.Tensor,
        target_x: torch.Tensor,
    ) -> dict:
        """
        Stage 1: Adversarial domain adaptation forward pass.
        
        Returns dict with:
            - rul_pred: RUL predictions for source
            - domain_pred_s: Domain predictions for source
            - domain_pred_t: Domain predictions for target
            - mcl_features: For MCL constraint
        """
        # Source forward
        source_fused, source_time, source_freq = self.encoder(source_x)
        source_rul = self.rul_predictor(source_fused)
        
        # Target forward
        target_fused, target_time, target_freq = self.encoder(target_x)
        
        # Domain discrimination (with GRL)
        source_grl = self.grl(source_fused)
        target_grl = self.grl(target_fused)
        domain_pred_s = self.domain_discriminator(source_grl)
        domain_pred_t = self.domain_discriminator(target_grl)
        
        # MCL: momentum encoder for target (time-augmented view)
        with torch.no_grad():
            self.momentum_encoder.eval()
            target_momentum, _, _ = self.momentum_encoder(target_x)
            self._mcl_enqueue(target_momentum)
        
        # Momentum update
        self._momentum_update()
        
        return {
            "source_rul": source_rul,
            "domain_pred_s": domain_pred_s,
            "domain_pred_t": domain_pred_t,
            "source_fused": source_fused,
            "target_fused": target_fused,
            "target_momentum": target_momentum,
        }
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: predict RUL for target domain."""
        self.eval()
        with torch.no_grad():
            fused, _, _ = self.encoder(x)
            rul = self.rul_predictor(fused)
        return rul


def compute_peuda_loss(
    outputs: dict,
    source_y: torch.Tensor,
    lambda_adv: float = 1.0,
    lambda_mcl: float = 0.5,
    mcl_queue: torch.Tensor = None,
    mcl_temperature: float = 0.07,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PEUDA training losses.
    
    Returns:
        total_loss, loss_dict
    """
    # 1. Regression loss (source only, MSE)
    loss_reg = F.mse_loss(outputs["source_rul"].squeeze(), source_y.squeeze())
    
    # 2. Adversarial domain confusion loss
    # Discriminator should NOT be able to distinguish source vs target
    # So we use binary labels flipped: all labeled as 0.5 (confused)
    B_s = outputs["domain_pred_s"].shape[0]
    B_t = outputs["domain_pred_t"].shape[0]
    
    # Domain labels: source=0, target=1
    domain_labels_s = torch.zeros(B_s, 1, device=source_y.device)
    domain_labels_t = torch.ones(B_t, 1, device=source_y.device)
    
    loss_domain_s = F.binary_cross_entropy_with_logits(
        outputs["domain_pred_s"], domain_labels_s
    )
    loss_domain_t = F.binary_cross_entropy_with_logits(
        outputs["domain_pred_t"], domain_labels_t
    )
    loss_adv = loss_domain_s + loss_domain_t
    
    # 3. MCL constraint: keep target features consistent with momentum encoder
    loss_mcl = torch.tensor(0.0, device=source_y.device)
    if mcl_queue is not None and len(mcl_queue) > 0:
        target_fused = F.normalize(outputs["target_fused"], dim=-1)
        target_momentum = F.normalize(outputs["target_momentum"], dim=-1)
        
        # InfoNCE-style: positive = momentum output, negatives = queue
        l_pos = torch.sum(target_fused * target_momentum, dim=-1, keepdim=True) / mcl_temperature
        l_neg = torch.matmul(target_fused, mcl_queue.T) / mcl_temperature
        
        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(B_t, dtype=torch.long, device=source_y.device)
        loss_mcl = F.cross_entropy(logits, labels)
    
    # Total loss
    total_loss = loss_reg + lambda_adv * loss_adv + lambda_mcl * loss_mcl
    
    loss_dict = {
        "loss_reg": loss_reg.item(),
        "loss_adv": loss_adv.item(),
        "loss_mcl": loss_mcl.item(),
        "total": total_loss.item(),
    }
    
    return total_loss, loss_dict
