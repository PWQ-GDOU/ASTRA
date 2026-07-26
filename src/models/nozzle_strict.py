"""Compact models for the strict single-trajectory nozzle benchmark.

The models in this module deliberately stay small: the canonical COMSOL run has
only 56 observations.  Every model predicts non-negative time-RUL in seconds.
Physics-guided variants receive a deterministic current-rate extrapolation as
an explicit input instead of manufacturing latent temperatures/pressures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NozzleModelOutput:
    rul: torch.Tensor
    rate_mm_s: Optional[torch.Tensor] = None


class PositiveHead(nn.Module):
    def __init__(self, in_features: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x)).squeeze(-1)


class TinyLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24, dropout: float = 0.1,
                 rul_scale: float = 20.0):
        super().__init__()
        self.rul_scale = rul_scale
        self.lstm = nn.LSTM(n_features, hidden, num_layers=1, batch_first=True)
        self.head = PositiveHead(hidden, max(8, hidden // 2), dropout)

    def forward(self, x: torch.Tensor, physics_rul: Optional[torch.Tensor] = None):
        del physics_rul
        h, _ = self.lstm(x)
        return NozzleModelOutput(rul=self.rul_scale * self.head(h[:, -1]))


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding
        )
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        h = self.conv(x)[..., :length]
        h = self.dropout(self.mix(F.gelu(h)))
        return self.norm((x + h).transpose(1, 2)).transpose(1, 2)


class TinyTCNEncoder(nn.Module):
    def __init__(self, n_features: int, channels: int = 24,
                 dropout: float = 0.1):
        super().__init__()
        self.project = nn.Linear(n_features, channels)
        self.blocks = nn.ModuleList([
            CausalBlock(channels, kernel_size=3, dilation=1, dropout=dropout),
            CausalBlock(channels, kernel_size=3, dilation=2, dropout=dropout),
        ])
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.project(x).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        # The last state preserves causal endpoint semantics.
        return self.norm(h.transpose(1, 2))[:, -1]


class TinyTCN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24, dropout: float = 0.1,
                 rul_scale: float = 20.0):
        super().__init__()
        self.rul_scale = rul_scale
        self.encoder = TinyTCNEncoder(n_features, hidden, dropout)
        self.head = PositiveHead(hidden, max(8, hidden // 2), dropout)

    def forward(self, x: torch.Tensor, physics_rul: Optional[torch.Tensor] = None):
        del physics_rul
        return NozzleModelOutput(rul=self.rul_scale * self.head(self.encoder(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * divisor)
        pe[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TinyTransformer(nn.Module):
    def __init__(self, n_features: int, hidden: int = 24, dropout: float = 0.1,
                 rul_scale: float = 20.0):
        super().__init__()
        self.rul_scale = rul_scale
        n_heads = 4 if hidden % 4 == 0 else 2
        self.project = nn.Linear(n_features, hidden)
        self.position = PositionalEncoding(hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)
        self.head = PositiveHead(hidden, max(8, hidden // 2), dropout)

    def forward(self, x: torch.Tensor, physics_rul: Optional[torch.Tensor] = None):
        del physics_rul
        h = self.encoder(self.position(self.project(x)))
        return NozzleModelOutput(rul=self.rul_scale * self.head(self.norm(h[:, -1])))


class PhysicsGuidedTCN(nn.Module):
    """TCN with an explicit, auditable current-rate physics branch.

    ``physics_rul`` is depth margin divided by the observed ablation rate.  It
    is transformed with ``log1p`` and fused with the causal state.  In ``full``
    mode an auxiliary positive rate head supports supervised rate and
    depth/rate/RUL consistency losses in the trainer.
    """

    def __init__(self, n_features: int, hidden: int = 24, dropout: float = 0.1,
                 full_physics: bool = False):
        super().__init__()
        self.full_physics = full_physics
        self.encoder = TinyTCNEncoder(n_features, hidden, dropout)
        self.physics_projection = nn.Sequential(
            nn.Linear(1, 8),
            nn.GELU(),
        )
        residual_hidden = max(8, hidden // 2)
        self.rul_residual = nn.Sequential(
            nn.Linear(hidden + 8, residual_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, 1),
        )
        # Start exactly from the deterministic physics baseline. Training may
        # learn at most a bounded multiplicative correction.
        nn.init.zeros_(self.rul_residual[-1].weight)
        nn.init.zeros_(self.rul_residual[-1].bias)
        self.rate_head = PositiveHead(hidden + 8, residual_hidden, dropout)

    def forward(self, x: torch.Tensor, physics_rul: Optional[torch.Tensor] = None):
        if physics_rul is None:
            raise ValueError("PhysicsGuidedTCN requires physics_rul")
        z = self.encoder(x)
        p = self.physics_projection(torch.log1p(physics_rul.clamp_min(0)).unsqueeze(-1))
        fused = torch.cat([z, p], dim=-1)
        log_factor = 0.5 * torch.tanh(self.rul_residual(fused).squeeze(-1))
        rul = physics_rul.clamp_min(0) * torch.exp(log_factor)
        rate = 0.02 * self.rate_head(fused) if self.full_physics else None
        return NozzleModelOutput(rul=rul, rate_mm_s=rate)


def build_nozzle_model(name: str, n_features: int, hidden: int = 24,
                       dropout: float = 0.1) -> nn.Module:
    if name == "lstm":
        return TinyLSTM(n_features, hidden, dropout)
    if name == "tcn":
        return TinyTCN(n_features, hidden, dropout)
    if name == "transformer":
        return TinyTransformer(n_features, hidden, dropout)
    if name == "physics_input":
        return PhysicsGuidedTCN(n_features, hidden, dropout, full_physics=False)
    if name == "physics_full":
        return PhysicsGuidedTCN(n_features, hidden, dropout, full_physics=True)
    raise ValueError(f"Unknown nozzle model: {name}")
