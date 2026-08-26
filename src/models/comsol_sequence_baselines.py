"""Target-only sequence baselines for the COMSOL outer-fold benchmark."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SequenceRULOutput:
    rul: torch.Tensor


class CausalTCNBlock(nn.Module):
    """Residual causal convolution block that never consumes future samples."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
        )
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        length = values.size(-1)
        hidden = self.convolution(values)[..., :length]
        hidden = self.dropout(F.gelu(self.mix(hidden)))
        return self.norm((values + hidden).transpose(1, 2)).transpose(1, 2)


class TargetGRUBaseline(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(n_features)
        self.encoder = nn.GRU(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(self, values: torch.Tensor) -> SequenceRULOutput:
        hidden, _ = self.encoder(self.input_norm(values))
        return SequenceRULOutput(self.head(hidden[:, -1]).squeeze(-1))


class TargetTCNBaseline(nn.Module):
    def __init__(self, n_features: int, hidden: int = 72, dropout: float = 0.1) -> None:
        super().__init__()
        branch = max(12, hidden // 3)
        self.branch = branch
        self.projection = nn.Conv1d(n_features, branch * 3, 1)
        self.branches = nn.ModuleList(
            nn.Sequential(
                CausalTCNBlock(branch, kernel_size=kernel_size, dilation=1, dropout=dropout),
                CausalTCNBlock(branch, kernel_size=kernel_size, dilation=2, dropout=dropout),
            )
            for kernel_size in (3, 5, 7)
        )
        self.norm = nn.LayerNorm(branch * 3)
        self.head = nn.Sequential(
            nn.Linear(branch * 3, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, values: torch.Tensor) -> SequenceRULOutput:
        hidden = self.projection(values.transpose(1, 2))
        branches = [
            block(branch_values)
            for branch_values, block in zip(torch.split(hidden, self.branch, dim=1), self.branches)
        ]
        encoded = self.norm(torch.cat(branches, dim=1).transpose(1, 2))[:, -1]
        return SequenceRULOutput(self.head(encoded).squeeze(-1))


class TargetTransformerBaseline(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.1, sequence_length: int = 20) -> None:
        super().__init__()
        self.projection = nn.Linear(n_features, hidden)
        self.position = nn.Parameter(torch.zeros(1, sequence_length, hidden))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward(self, values: torch.Tensor) -> SequenceRULOutput:
        if values.size(1) > self.position.size(1):
            raise ValueError("Sequence exceeds the configured positional embedding length")
        hidden = self.projection(values) + self.position[:, : values.size(1)]
        encoded = self.norm(self.encoder(hidden)[:, -1])
        return SequenceRULOutput(self.head(encoded).squeeze(-1))


def build_comsol_sequence_baseline(name: str, n_features: int, sequence_length: int = 20) -> nn.Module:
    key = name.lower()
    if key == "gru":
        return TargetGRUBaseline(n_features)
    if key == "tcn":
        return TargetTCNBaseline(n_features)
    if key == "transformer":
        return TargetTransformerBaseline(n_features, sequence_length=sequence_length)
    raise ValueError(f"Unknown COMSOL sequence baseline: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
