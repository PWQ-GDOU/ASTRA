"""Models for the multi-trajectory nozzle ablation RUL benchmark.

Key design principles:
- PhysicsResidualRateNet learns future ablation-rate evolution and integrates
  to threshold, rather than extrapolating the instantaneous current rate.
- WeibullRULLoss weights errors by proximity to failure so late-life
  predictions receive proportionally more gradient (adapted from
  tvhahn/weibull-knowledge-informed-ml, MIT license).
- All models predict non-negative time-RUL in seconds.
- Oracle simulation fields must NEVER reach this module.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NozzleOutput:
    rul_s: torch.Tensor
    rate_m_s: Optional[torch.Tensor] = None
    rate_growth: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Weibull-weighted RUL loss
# ---------------------------------------------------------------------------


class WeibullRULLoss(nn.Module):
    """Weibull-CDF-weighted smooth-L1 loss for time-to-failure RUL.

    Near-failure windows receive higher weight because CDF rises steeply
    near eta.  The weight is |CDF(t_age) - CDF(t_predicted_eol)| where
    t_age is the current age (elapsed time) of the trajectory.

    Reference: von Hahn & Mechefske (2022), arXiv:2201.01769.
    Adapted to smooth-L1 with trajectory-level normalisation.
    """

    def __init__(self, eta: float = 15.0, beta: float = 2.5, eps: float = 1.0e-8):
        super().__init__()
        self.eta = eta
        self.beta = beta
        self.eps = eps

    def _weibull_cdf(self, t: torch.Tensor) -> torch.Tensor:
        t_clamped = t.clamp(min=self.eps)
        return 1.0 - torch.exp(-((t_clamped / self.eta) ** self.beta))

    def forward(
        self,
        pred_rul: torch.Tensor,
        true_rul: torch.Tensor,
        age_s: torch.Tensor,
        rul_scale: float = 20.0,
    ) -> torch.Tensor:
        pred_eol = age_s + pred_rul
        true_eol = age_s + true_rul
        cdf_true = self._weibull_cdf(true_eol)
        cdf_pred = self._weibull_cdf(pred_eol)
        weibull_weight = (cdf_true - cdf_pred).abs().detach() + self.eps
        huber = F.smooth_l1_loss(
            pred_rul / rul_scale,
            true_rul / rul_scale,
            beta=0.05,
            reduction="none",
        )
        return (weibull_weight * huber).mean()


class CensoredRULLoss(nn.Module):
    """Combined loss for event-observed and right-censored trajectories."""

    def __init__(self, eta: float = 15.0, beta: float = 2.5):
        super().__init__()
        self.weibull = WeibullRULLoss(eta=eta, beta=beta)
        self.eps = 1.0e-8

    def forward(
        self,
        pred_rul: torch.Tensor,
        true_rul: torch.Tensor,
        age_s: torch.Tensor,
        target_mask: torch.Tensor,
        lower_bound_s: torch.Tensor,
        rul_scale: float = 20.0,
    ) -> torch.Tensor:
        total = torch.tensor(0.0, device=pred_rul.device)
        if target_mask.any():
            exact = target_mask
            total = total + self.weibull(pred_rul[exact], true_rul[exact], age_s[exact], rul_scale)
        censored = ~target_mask
        if censored.any():
            lb = lower_bound_s[censored].to(pred_rul.dtype)
            violation = F.relu(lb - pred_rul[censored]) / max(rul_scale, self.eps)
            total = total + violation.pow(2).mean()
        return total


# ---------------------------------------------------------------------------
# Encoder blocks (shared across models)
# ---------------------------------------------------------------------------


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int = 3, dilation: int = 1, dropout: float = 0.1):
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


class MultiScaleEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.1):
        super().__init__()
        branch = max(12, hidden // 3)
        self.branch = branch
        self.proj = nn.Conv1d(n_features, branch * 3, 1)
        self.branches = nn.ModuleList([
            nn.Sequential(CausalBlock(branch, k, 1, dropout), CausalBlock(branch, k, 2, dropout))
            for k in (3, 5, 7)
        ])
        self.norm = nn.LayerNorm(branch * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x.transpose(1, 2))
        outputs = [block(branch_x) for branch_x, block in zip(torch.split(h, self.branch, 1), self.branches)]
        return self.norm(torch.cat(outputs, 1).transpose(1, 2)).mean(dim=1)


class GRUEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return h[:, -1]


class TransformerEncoder(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(self.proj(x))[:, -1])


# ---------------------------------------------------------------------------
# Plain RUL heads (LSTM / TCN / Transformer)
# ---------------------------------------------------------------------------


def _rul_head(hidden: int, dropout: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1)
    )


class GRUModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.1, rul_scale: float = 20.0):
        super().__init__()
        self.rul_scale = rul_scale
        self.encoder = GRUEncoder(n_features, hidden)
        self.head = _rul_head(hidden, dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> NozzleOutput:
        return NozzleOutput(rul_s=self.rul_scale * F.softplus(self.head(self.encoder(x)).squeeze(-1)))


class MultiScaleModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.1, rul_scale: float = 20.0):
        super().__init__()
        self.rul_scale = rul_scale
        self.encoder = MultiScaleEncoder(n_features, hidden, dropout)
        branch = max(12, hidden // 3)
        self.head = _rul_head(branch * 3, dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> NozzleOutput:
        z = self.encoder(x)
        return NozzleOutput(rul_s=self.rul_scale * F.softplus(self.head(z).squeeze(-1)))


class TransformerModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.1, rul_scale: float = 20.0):
        super().__init__()
        self.rul_scale = rul_scale
        self.encoder = TransformerEncoder(n_features, hidden, dropout)
        self.head = _rul_head(hidden, dropout)

    def forward(self, x: torch.Tensor, **kwargs) -> NozzleOutput:
        return NozzleOutput(rul_s=self.rul_scale * F.softplus(self.head(self.encoder(x)).squeeze(-1)))


# ---------------------------------------------------------------------------
# PhysicsResidualRateNet — the primary new model
# ---------------------------------------------------------------------------


class PhysicsResidualRateNet(nn.Module):
    """Physics-residual model that predicts future rate growth and integrates to failure.

    Architecture:
    - TCN encoder processes observable history (temperature, heat flux, pressure)
    - Rate head predicts multiplicative rate-growth factor relative to current rate
    - Optional EOL head predicts integrated time-to-threshold
    - RUL = max(EOL_time - current_time, 0)

    When depth/rate are available (estimated tier) the model can also
    output a depth-consistent rate prediction. When only thermal observables
    are available (observable tier) depth and rate are excluded from inputs.

    The model initialises the rate-growth branch to identity (zero correction)
    so it starts from a neutral prior, not a random initialisation.
    """

    def __init__(
        self,
        n_features: int,
        hidden: int = 48,
        dropout: float = 0.1,
        rul_scale: float = 20.0,
        n_future_steps: int = 4,
    ):
        super().__init__()
        self.rul_scale = rul_scale
        self.n_future_steps = n_future_steps

        self.encoder = MultiScaleEncoder(n_features, hidden, dropout)
        branch = max(12, hidden // 3)
        encoded_dim = branch * 3

        # Rate-growth head: predicts log-scale multiplicative factor for future rate.
        # Initialised to zero so initial output is exp(0)=1 (identity correction).
        self.rate_growth_head = nn.Sequential(
            nn.Linear(encoded_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_future_steps),
        )
        nn.init.zeros_(self.rate_growth_head[-1].weight)
        nn.init.zeros_(self.rate_growth_head[-1].bias)

        # EOL head integrates rate trajectory to estimate crossing time.
        self.eol_head = nn.Sequential(
            nn.Linear(encoded_dim + n_future_steps, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        # Initialise EOL residual to zero (no correction from physics prior).
        nn.init.zeros_(self.eol_head[-1].weight)
        nn.init.zeros_(self.eol_head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        current_rate_m_s: Optional[torch.Tensor] = None,
        depth_margin_mm: Optional[torch.Tensor] = None,
        current_time_s: Optional[torch.Tensor] = None,
    ) -> NozzleOutput:
        z = self.encoder(x)

        # Rate-growth factor per future step (log-space for positivity).
        log_growth = self.rate_growth_head(z)
        growth_factors = torch.exp(log_growth.clamp(-2.0, 2.0))

        # Physics-informed EOL estimate.
        # If current rate and margin are available use them as prior; otherwise
        # rely on learned features alone.
        fused = torch.cat([z, log_growth.detach()], dim=-1)
        eol_residual = F.softplus(self.eol_head(fused).squeeze(-1))

        if current_rate_m_s is not None and depth_margin_mm is not None:
            # Step-integrated time estimate: Σ(Δdepth_per_step / rate_step)
            # where each step uses the previous step's rate * growth factor.
            # This is a physics-grounded prior that the EOL head can correct.
            rate = current_rate_m_s.clamp(min=1.0e-9)
            margin_m = depth_margin_mm.clamp(min=0.0) * 1.0e-3
            step_depth = margin_m / max(self.n_future_steps, 1)
            t_physics = torch.zeros_like(rate)
            for i in range(self.n_future_steps):
                rate = rate * growth_factors[:, i]
                t_physics = t_physics + step_depth / rate.clamp(min=1.0e-9)
            rul_s = (t_physics + eol_residual).clamp(min=0.0)
        else:
            # Observable tier: no depth/rate available; rely on encoded features.
            rul_s = (eol_residual * self.rul_scale).clamp(min=0.0)

        return NozzleOutput(
            rul_s=rul_s,
            rate_m_s=current_rate_m_s * growth_factors[:, 0] if current_rate_m_s is not None else None,
            rate_growth=growth_factors,
        )


# ---------------------------------------------------------------------------
# Factory and utilities
# ---------------------------------------------------------------------------


def build_nozzle_mt_model(
    name: str,
    n_features: int,
    hidden: int = 48,
    dropout: float = 0.1,
    rul_scale: float = 20.0,
) -> nn.Module:
    key = name.lower()
    if key in {"gru", "lstm"}:
        return GRUModel(n_features, hidden, dropout, rul_scale)
    if key in {"ms", "tcn", "multiscale"}:
        return MultiScaleModel(n_features, hidden, dropout, rul_scale)
    if key in {"transformer"}:
        return TransformerModel(n_features, hidden, dropout, rul_scale)
    if key in {"physics_residual", "rate_net"}:
        return PhysicsResidualRateNet(n_features, hidden, dropout, rul_scale)
    raise ValueError(f"Unknown multi-trajectory nozzle model: {name!r}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
