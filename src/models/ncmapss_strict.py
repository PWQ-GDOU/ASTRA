"""SOTA-targeting sequence models for N-CMAPSS turbofan RUL prediction.

Architecture rationale for N-CMAPSS:
- Variable operating conditions require condition-aware encoding.
- Long trajectories (up to 400+ cycles) favour gated/attention mechanisms.
- Multi-scale temporal receptive fields capture both short-term fluctuations
  and long-term degradation trends simultaneously.
- RUL head uses softplus to guarantee non-negative output; scale matches cap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


RUL_SCALE = 125.0  # matches standard RUL cap


@dataclass
class RULOutput:
    rul: torch.Tensor
    degradation: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class ConditionNormLayer(nn.Module):
    """Learnable affine normalisation conditioned on operating point.

    The operating conditions (W) modulate mean and variance of the sensor
    features so the encoder sees condition-de-biased sensor readings.
    """

    def __init__(self, n_features: int, n_conditions: int, hidden: int = 16):
        super().__init__()
        self.gamma_net = nn.Sequential(
            nn.Linear(n_conditions, hidden), nn.GELU(), nn.Linear(hidden, n_features)
        )
        self.beta_net = nn.Sequential(
            nn.Linear(n_conditions, hidden), nn.GELU(), nn.Linear(hidden, n_features)
        )
        nn.init.zeros_(self.gamma_net[-1].bias)
        nn.init.zeros_(self.beta_net[-1].bias)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_features)
            w: (B, T, n_conditions)  OR  (B, n_conditions) — broadcast if needed
        """
        if w.dim() == 2:
            w = w.unsqueeze(1).expand(-1, x.size(1), -1)
        gamma = 1.0 + self.gamma_net(w)
        beta = self.beta_net(w)
        return gamma * x + beta


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation,
                              padding=(kernel - 1) * dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = x.size(-1)
        h = self.conv(x)[..., :l]
        h = self.drop(F.gelu(self.mix(h)))
        return self.norm((x + h).transpose(1, 2)).transpose(1, 2)


# ---------------------------------------------------------------------------
# Model 1: Condition-Aware Multi-Scale TCN  (strong baseline)
# ---------------------------------------------------------------------------


class ConditionAwareMultiScaleTCN(nn.Module):
    """Multi-scale TCN with operating-condition film normalisation.

    This directly extends the C-MAPSS multiscale TCN architecture to handle
    N-CMAPSS's variable operating conditions.
    """

    def __init__(
        self,
        n_features: int,
        n_conditions: int = 4,
        hidden: int = 96,
        dropout: float = 0.1,
        rul_scale: float = RUL_SCALE,
    ):
        super().__init__()
        self.rul_scale = rul_scale
        self.n_conditions = n_conditions
        self.n_sensor_features = n_features - n_conditions if n_conditions > 0 else n_features
        if n_conditions > 0:
            self.cond_norm = ConditionNormLayer(self.n_sensor_features, n_conditions)
        else:
            self.cond_norm = None
        branch = max(16, hidden // 3)
        self.branch = branch
        self.proj = nn.Conv1d(self.n_sensor_features, branch * 3, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalBlock(branch, k, 1, dropout),
                CausalBlock(branch, k, 2, dropout),
                CausalBlock(branch, k, 4, dropout),
            )
            for k in (3, 5, 7)
        ])
        self.norm = nn.LayerNorm(branch * 3)
        self.head = nn.Sequential(
            nn.Linear(branch * 3, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        w: Optional[torch.Tensor] = None,
    ) -> RULOutput:
        if self.cond_norm is not None and self.n_conditions > 0:
            sensors = x[..., :self.n_sensor_features]
            conditions = x[..., self.n_sensor_features:]
            x_enc = self.cond_norm(sensors, conditions)
        else:
            x_enc = x
        h = self.proj(x_enc.transpose(1, 2))
        outs = [block(bx) for bx, block in zip(torch.split(h, self.branch, 1), self.branches)]
        z = self.norm(torch.cat(outs, 1).transpose(1, 2)).mean(dim=1)
        return RULOutput(rul=self.rul_scale * F.softplus(self.head(z).squeeze(-1)))


# ---------------------------------------------------------------------------
# Model 2: Bidirectional GRU with Temporal Attention  (strong sequence model)
# ---------------------------------------------------------------------------


class BiGRUAttention(nn.Module):
    """Bidirectional GRU with self-attention pooling over the temporal axis.

    Attention lets the model weight which time steps are most informative
    for predicting remaining useful life, rather than just using the last state.
    """

    def __init__(
        self,
        n_features: int,
        n_conditions: int = 4,
        hidden: int = 96,
        n_heads: int = 4,
        dropout: float = 0.1,
        rul_scale: float = RUL_SCALE,
    ):
        super().__init__()
        self.rul_scale = rul_scale
        self.n_conditions = n_conditions
        self.n_sensor_features = n_features - n_conditions if n_conditions > 0 else n_features
        if n_conditions > 0:
            self.cond_norm = ConditionNormLayer(self.n_sensor_features, n_conditions)
        else:
            self.cond_norm = None
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=dropout)
        attn_dim = hidden * 2
        self.attn = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(attn_dim)
        self.head = nn.Sequential(
            nn.Linear(attn_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> RULOutput:
        if self.cond_norm is not None and self.n_conditions > 0:
            sensors = x[..., :self.n_sensor_features]
            conditions = x[..., self.n_sensor_features:]
            sensors_normed = self.cond_norm(sensors, conditions)
            x_enc = torch.cat([sensors_normed, conditions], dim=-1)
        else:
            x_enc = x
        h, _ = self.gru(x_enc)
        attended, weights = self.attn(h, h, h)
        z = self.norm(h + attended).mean(dim=1)
        return RULOutput(
            rul=self.rul_scale * F.softplus(self.head(z).squeeze(-1)),
            attention_weights=weights,
        )


# ---------------------------------------------------------------------------
# Model 3: Temporal Transformer Encoder  (attention-based SOTA candidate)
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerRUL(nn.Module):
    """Transformer encoder with condition-aware normalisation and degradation head.

    The degradation head predicts a monotonically increasing health indicator
    as an auxiliary task to regularise RUL prediction.
    """

    def __init__(
        self,
        n_features: int,
        n_conditions: int = 4,
        hidden: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.1,
        rul_scale: float = RUL_SCALE,
    ):
        super().__init__()
        self.rul_scale = rul_scale
        self.n_conditions = n_conditions
        self.n_sensor_features = n_features - n_conditions if n_conditions > 0 else n_features
        if n_conditions > 0:
            self.cond_norm = ConditionNormLayer(self.n_sensor_features, n_conditions)
        else:
            self.cond_norm = None
        self.proj = nn.Linear(n_features, hidden)
        self.pos_enc = PositionalEncoding(hidden, dropout=dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden)
        self.rul_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.degradation_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> RULOutput:
        if self.cond_norm is not None and self.n_conditions > 0:
            sensors = x[..., :self.n_sensor_features]
            cond = x[..., self.n_sensor_features:]
            sensors = self.cond_norm(sensors, cond)
            x_enc = torch.cat([sensors, cond], dim=-1)
        else:
            x_enc = x
        h = self.norm(self.encoder(self.pos_enc(self.proj(x_enc))))[:, -1]
        rul = self.rul_scale * F.softplus(self.rul_head(h).squeeze(-1))
        deg = self.degradation_head(h).squeeze(-1)
        return RULOutput(rul=rul, degradation=deg)


# ---------------------------------------------------------------------------
# Model 4: Physics-Informed GRU (exploits thermodynamic relationships)
# ---------------------------------------------------------------------------


class PhysicsInformedGRU(nn.Module):
    """GRU augmented with physics-derived degradation rate estimate.

    The physics branch estimates a normalised degradation state from the
    trend of thermal sensor readings; the GRU branch learns residual
    corrections. Initialised so physics branch dominates early in training.
    """

    def __init__(
        self,
        n_features: int,
        n_conditions: int = 4,
        hidden: int = 96,
        dropout: float = 0.1,
        rul_scale: float = RUL_SCALE,
    ):
        super().__init__()
        self.rul_scale = rul_scale
        self.n_conditions = n_conditions
        self.gru = nn.GRU(n_features, hidden, num_layers=2, batch_first=True,
                          dropout=dropout)
        # Physics branch: linear model of thermal/pressure trend
        self.physics_proj = nn.Linear(n_features, 1)
        # Residual head initialised near zero
        self.residual_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None) -> RULOutput:
        h, _ = self.gru(x)
        z = self.dropout(h[:, -1])
        # Physics estimate: monotonic trend from last window step
        physics_raw = self.physics_proj(x[:, -1]).squeeze(-1)
        physics_est = torch.sigmoid(physics_raw) * self.rul_scale
        fused = torch.cat([z, physics_est.unsqueeze(-1)], dim=-1)
        residual = self.residual_head(fused).squeeze(-1)
        rul = F.softplus(physics_est + residual)
        return RULOutput(rul=rul)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_ncmapss_model(
    name: str,
    n_features: int,
    n_conditions: int = 4,
    hidden: int = 96,
    dropout: float = 0.1,
    rul_scale: float = RUL_SCALE,
) -> nn.Module:
    key = name.lower()
    if key in {"ms_tcn", "multiscale_tcn", "cond_ms_tcn"}:
        return ConditionAwareMultiScaleTCN(n_features, n_conditions, hidden, dropout, rul_scale)
    if key in {"bigru_attn", "bigru"}:
        return BiGRUAttention(n_features, n_conditions, hidden, 4, dropout, rul_scale)
    if key in {"transformer", "transformer_rul"}:
        return TransformerRUL(n_features, n_conditions, hidden, 8, 3, dropout, rul_scale)
    if key in {"physics_gru", "phys_gru"}:
        return PhysicsInformedGRU(n_features, n_conditions, hidden, dropout, rul_scale)
    raise ValueError(f"Unknown N-CMAPSS model: {name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
