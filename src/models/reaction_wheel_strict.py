"""Small sequence models for the FEMTO reaction-wheel proxy benchmark."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ReactionWheelOutput:
    rul: torch.Tensor


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=(kernel - 1) * dilation)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        h = self.conv(x)[..., :length]
        h = self.dropout(F.gelu(self.mix(h)))
        return self.norm((x + h).transpose(1, 2)).transpose(1, 2)


class GRURUL(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 48), nn.GELU(), nn.Dropout(dropout), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> ReactionWheelOutput:
        h, _ = self.encoder(x)
        return ReactionWheelOutput(self.head(h[:, -1]).squeeze(-1))


class MultiScaleRUL(nn.Module):
    def __init__(self, n_features: int, hidden: int = 72, dropout: float = 0.1):
        super().__init__()
        branch = max(12, hidden // 3)
        self.branch = branch
        self.proj = nn.Conv1d(n_features, branch * 3, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalBlock(branch, kernel, 1, dropout),
                CausalBlock(branch, kernel, 2, dropout),
            )
            for kernel in (3, 5, 7)
        ])
        self.norm = nn.LayerNorm(branch * 3)
        self.head = nn.Sequential(nn.Linear(branch * 3, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> ReactionWheelOutput:
        h = self.proj(x.transpose(1, 2))
        outputs = []
        for branch_x, block in zip(torch.split(h, self.branch, dim=1), self.branches):
            outputs.append(block(branch_x))
        z = self.norm(torch.cat(outputs, dim=1).transpose(1, 2)).mean(dim=1)
        return ReactionWheelOutput(self.head(z).squeeze(-1))


class TinyTransformerRUL(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, 48), nn.GELU(), nn.Dropout(dropout), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> ReactionWheelOutput:
        z = self.norm(self.encoder(self.proj(x))[:, -1])
        return ReactionWheelOutput(self.head(z).squeeze(-1))


class DeepMultiScaleRUL(nn.Module):
    """Deeper multi-scale TCN with hidden=128 and three dilation levels."""
    def __init__(self, n_features: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        branch = max(32, hidden // 4)
        self.branch = branch
        self.proj = nn.Conv1d(n_features, branch * 4, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalBlock(branch, kernel, 1, dropout),
                CausalBlock(branch, kernel, 2, dropout),
                CausalBlock(branch, kernel, 4, dropout),
            )
            for kernel in (3, 5, 7, 9)
        ])
        self.norm = nn.LayerNorm(branch * 4)
        self.head = nn.Sequential(
            nn.Linear(branch * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> ReactionWheelOutput:
        h = self.proj(x.transpose(1, 2))
        outputs = []
        for branch_x, block in zip(torch.split(h, self.branch, dim=1), self.branches):
            outputs.append(block(branch_x))
        z = self.norm(torch.cat(outputs, dim=1).transpose(1, 2)).mean(dim=1)
        return ReactionWheelOutput(self.head(z).squeeze(-1))


class LargeGRURUL(nn.Module):
    """Two-layer GRU with hidden=128 and a larger MLP head."""
    def __init__(self, n_features: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.GRU(
            n_features,
            hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> ReactionWheelOutput:
        h, _ = self.encoder(x)
        return ReactionWheelOutput(self.head(h[:, -1]).squeeze(-1))


def build_reaction_wheel_model(name: str, n_features: int) -> nn.Module:
    key = name.lower()
    if key in {"gru", "lstm"}:
        return GRURUL(n_features)
    if key in {"ms", "tcn", "multiscale"}:
        return MultiScaleRUL(n_features)
    if key in {"transformer", "tiny_transformer"}:
        return TinyTransformerRUL(n_features)
    if key in {"deep_ms", "deep_multiscale", "dms"}:
        return DeepMultiScaleRUL(n_features)
    if key in {"large_gru", "gru_large"}:
        return LargeGRURUL(n_features)
    raise ValueError(f"Unknown reaction-wheel model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
