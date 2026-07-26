"""Small battery RUL models used by the strict nested evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# RUL is reported in cycles, but this scale keeps the positive head in the
# same numerical range as the normalized degradation features during training.
RUL_SCALE = 100.0


@dataclass
class BatteryModelOutput:
    rul: torch.Tensor
    capacity_delta_5: torch.Tensor
    capacity_delta_10: torch.Tensor
    life_cycle: torch.Tensor | None = None


class CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation,
        )
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(-1)
        h = self.conv(x)[..., :length]
        h = self.dropout(F.gelu(self.mix(h)))
        return self.norm((x + h).transpose(1, 2)).transpose(1, 2)


class _Heads(nn.Module):
    def __init__(self, hidden: int, predict_life: bool = False):
        super().__init__()
        self.predict_life = predict_life
        self.rul = nn.Sequential(nn.Linear(hidden, 48), nn.GELU(), nn.Linear(48, 1))
        self.delta5 = nn.Linear(hidden, 1)
        self.delta10 = nn.Linear(hidden, 1)
        self.life = nn.Sequential(nn.Linear(hidden, 48), nn.GELU(), nn.Linear(48, 1)) if predict_life else None

    def forward(self, z: torch.Tensor, age: torch.Tensor | None = None) -> BatteryModelOutput:
        # Predict normalized remaining life, then return physical cycle units.
        rul = RUL_SCALE * F.softplus(self.rul(z).squeeze(-1))
        life = None
        if self.predict_life:
            if age is None:
                age = torch.zeros_like(rul)
            life = age + RUL_SCALE * F.softplus(self.life(z).squeeze(-1))
            rul = F.relu(life - age)
        return BatteryModelOutput(
            rul=rul,
            capacity_delta_5=self.delta5(z).squeeze(-1),
            capacity_delta_10=self.delta10(z).squeeze(-1),
            life_cycle=life,
        )


class GRUModel(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.GRU(
            n_features,
            hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.heads = _Heads(hidden)

    def forward(self, x: torch.Tensor, age: torch.Tensor | None = None) -> BatteryModelOutput:
        h, _ = self.encoder(x)
        return self.heads(self.dropout(h[:, -1]), age)


class MultiScaleTCN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 72, dropout: float = 0.1):
        super().__init__()
        branch = max(12, hidden // 3)
        self.branch = branch
        self.proj = nn.Conv1d(n_features, branch * 3, 1)
        self.branches = nn.ModuleList(
            nn.Sequential(
                CausalBlock(branch, kernel_size=k, dilation=1, dropout=dropout),
                CausalBlock(branch, kernel_size=k, dilation=2, dropout=dropout),
            )
            for k in (3, 5, 7)
        )
        self.norm = nn.LayerNorm(branch * 3)
        self.shared = nn.Sequential(nn.Linear(branch * 3, 72), nn.GELU(), nn.Dropout(dropout))
        self.heads = _Heads(72)

    def forward(self, x: torch.Tensor, age: torch.Tensor | None = None) -> BatteryModelOutput:
        h = self.proj(x.transpose(1, 2))
        outputs = []
        for branch_x, branch_net in zip(torch.split(h, self.branch, dim=1), self.branches):
            outputs.append(branch_net(branch_x))
        z = self.norm(torch.cat(outputs, dim=1).transpose(1, 2)).mean(dim=1)
        return self.heads(self.shared(z), age)


class TinyTransformer(nn.Module):
    def __init__(self, n_features: int, hidden: int = 48, dropout: float = 0.1):
        super().__init__()
        heads = 4 if hidden % 4 == 0 else 3
        self.proj = nn.Linear(n_features, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.norm = nn.LayerNorm(hidden)
        self.heads = _Heads(hidden)

    def forward(self, x: torch.Tensor, age: torch.Tensor | None = None) -> BatteryModelOutput:
        z = self.encoder(self.proj(x))[:, -1]
        return self.heads(self.norm(z), age)


class BatteryLifeNet(nn.Module):
    """Capacity-aware model that predicts total EOL cycle and converts to RUL."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 72,
        dropout: float = 0.1,
        use_age: bool = True,
        use_capacity_aux: bool = True,
    ):
        super().__init__()
        self.use_age = use_age
        self.use_capacity_aux = use_capacity_aux
        branch = max(12, hidden // 3)
        self.branch = branch
        self.proj = nn.Conv1d(n_features, branch * 3, 1)
        self.branches = nn.ModuleList(
            nn.Sequential(
                CausalBlock(branch, kernel_size=k, dilation=1, dropout=dropout),
                CausalBlock(branch, kernel_size=k, dilation=2, dropout=dropout),
            )
            for k in (3, 5, 7)
        )
        self.norm = nn.LayerNorm(branch * 3)
        extra = 1 if use_age else 0
        self.shared = nn.Sequential(
            nn.Linear(branch * 3 + extra, 80),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.life_head = nn.Sequential(nn.Linear(80, 48), nn.GELU(), nn.Linear(48, 1))
        self.delta5 = nn.Linear(80, 1)
        self.delta10 = nn.Linear(80, 1)

    def forward(self, x: torch.Tensor, age: torch.Tensor | None = None) -> BatteryModelOutput:
        h = self.proj(x.transpose(1, 2))
        outputs = []
        for branch_x, branch_net in zip(torch.split(h, self.branch, dim=1), self.branches):
            outputs.append(branch_net(branch_x))
        z = self.norm(torch.cat(outputs, dim=1).transpose(1, 2)).mean(dim=1)
        if self.use_age:
            if age is None:
                age = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
            # Keep cycle units for life -> RUL conversion, but normalize the
            # auxiliary age input to the scale of the encoded features.
            z = torch.cat([z, (age / 100.0).unsqueeze(-1)], dim=-1)
        z = self.shared(z)
        if age is None:
            age = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        life = age + RUL_SCALE * F.softplus(self.life_head(z).squeeze(-1))
        rul = F.relu(life - age)
        return BatteryModelOutput(
            rul=rul,
            capacity_delta_5=self.delta5(z).squeeze(-1),
            capacity_delta_10=self.delta10(z).squeeze(-1),
            life_cycle=life,
        )


def build_battery_model(
    name: str,
    n_features: int,
    hidden: int | None = None,
    *,
    use_age: bool = True,
    use_capacity_aux: bool = True,
) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return GRUModel(n_features, hidden=hidden or 64)
    if name in {"ms", "multiscale", "tcn"}:
        return MultiScaleTCN(n_features, hidden=hidden or 72)
    if name in {"transformer", "tiny_transformer"}:
        return TinyTransformer(n_features, hidden=hidden or 48)
    if name in {"life", "batterylife", "battery_life"}:
        return BatteryLifeNet(
            n_features,
            hidden=hidden or 72,
            use_age=use_age,
            use_capacity_aux=use_capacity_aux,
        )
    raise ValueError(f"Unknown battery model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
