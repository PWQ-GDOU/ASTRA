"""Small domain-adapter model for source-bearing to COMSOL transfer."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CrossDomainOutput:
    rul: torch.Tensor


@dataclass
class TrendResidualOutput:
    rul: torch.Tensor
    residual: torch.Tensor


class CrossDomainRULModel(nn.Module):
    """Separate source/target projections with one shared degradation encoder."""

    def __init__(
        self,
        source_features: int,
        target_features: int,
        *,
        projection_dim: int = 48,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.source_projection = nn.Sequential(
            nn.LayerNorm(source_features),
            nn.Linear(source_features, projection_dim),
            nn.GELU(),
        )
        self.target_projection = nn.Sequential(
            nn.LayerNorm(target_features),
            nn.Linear(target_features, projection_dim),
            nn.GELU(),
        )
        self.encoder = nn.GRU(projection_dim, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def _forward(self, x: torch.Tensor, projection: nn.Module) -> CrossDomainOutput:
        projected = projection(x)
        encoded, _ = self.encoder(projected)
        return CrossDomainOutput(self.head(encoded[:, -1]).squeeze(-1))

    def forward_source(self, x: torch.Tensor) -> CrossDomainOutput:
        return self._forward(x, self.source_projection)

    def forward_target(self, x: torch.Tensor) -> CrossDomainOutput:
        return self._forward(x, self.target_projection)


def copy_shared_encoder(source: CrossDomainRULModel, target: CrossDomainRULModel) -> None:
    target.encoder.load_state_dict(source.encoder.state_dict())


class TrendResidualRULModel(nn.Module):
    """Target model that preserves a causal trend and transfers residual dynamics.

    The prior is supplied by the target-side fit only.  The network therefore
    learns a correction to a train-only trend rather than relearning the
    dominant absolute time-to-failure scale from a few target trajectories.
    """

    def __init__(
        self,
        target_features: int,
        *,
        context_features: int | None = None,
        projection_dim: int = 48,
        hidden: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.target_projection = nn.Sequential(
            nn.LayerNorm(target_features),
            nn.Linear(target_features, projection_dim),
            nn.GELU(),
        )
        self.encoder = nn.GRU(projection_dim, hidden, batch_first=True)
        self.context_features = int(context_features or 0)
        self.context_projection = (
            nn.Sequential(
                nn.LayerNorm(self.context_features),
                nn.Linear(self.context_features, 16),
                nn.GELU(),
            )
            if self.context_features
            else None
        )
        head_input = hidden + 1 + (16 if self.context_features else 0)
        self.residual_head = nn.Sequential(
            nn.Linear(head_input, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )

    def forward_target(
        self,
        x: torch.Tensor,
        prior: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> TrendResidualOutput:
        projected = self.target_projection(x)
        encoded, _ = self.encoder(projected)
        prior_column = prior.reshape(-1, 1)
        head_inputs = [encoded[:, -1], prior_column]
        if self.context_features:
            if context is None:
                context = torch.zeros(
                    (x.shape[0], self.context_features), dtype=x.dtype, device=x.device
                )
            head_inputs.append(self.context_projection(context))
        residual = self.residual_head(torch.cat(head_inputs, dim=1)).squeeze(-1)
        # A bounded correction prevents a low-shot target fold from discarding
        # the causal trend entirely while still allowing sizeable deviations.
        residual = 0.5 * torch.tanh(residual)
        return TrendResidualOutput(prior + residual, residual)


def copy_shared_encoder_to_trend_residual(
    source: CrossDomainRULModel,
    target: TrendResidualRULModel,
    *,
    copy_residual_head: bool = False,
) -> None:
    """Transfer the shared encoder and optionally the source residual head.

    The target head has one extra prior column (and optional context columns),
    so source hidden-state weights are copied into the matching prefix and
    target-only columns remain zero when the explicit head ablation is enabled.
    """
    target.encoder.load_state_dict(source.encoder.state_dict())
    if not copy_residual_head:
        return
    source_first = source.head[0]
    target_first = target.residual_head[0]
    source_last = source.head[3]
    target_last = target.residual_head[3]
    if source_first.out_features != target_first.out_features:
        raise ValueError("Source and target residual heads have incompatible hidden widths")
    if source_last.weight.shape != target_last.weight.shape:
        raise ValueError("Source and target residual heads have incompatible output widths")
    with torch.no_grad():
        target_first.weight.zero_()
        target_first.weight[:, : source_first.in_features].copy_(source_first.weight)
        target_first.bias.copy_(source_first.bias)
        target_last.weight.copy_(source_last.weight)
        target_last.bias.copy_(source_last.bias)

def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
