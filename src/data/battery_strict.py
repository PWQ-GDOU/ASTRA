"""Auditable, causal NASA battery data preparation for strict RUL evaluation.

This module intentionally keeps protocol decisions explicit.  In particular, a
cell that does not cross the requested EOL threshold is right-censored rather
than silently assigned an end-of-record RUL of zero.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.io as sio


BATTERY_NAMES = ("B0005", "B0006", "B0007", "B0018")
RATED_CAPACITY_AH = 2.0
STRICT_EOL_AH = 1.4
FEATURE_NAMES = (
    "soh",
    "soh_loss",
    "capacity_margin",
    "fade_1",
    "fade_5",
    "slope_5",
    "slope_10",
    "slope_20",
    "slope_40",
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_mid",
    "voltage_p10",
    "current_mean_abs",
    "current_std",
    "temperature_mean",
    "temperature_std",
    "temperature_max",
    "temperature_delta_early",
    "ttv_30",
    "ttv_27",
    "ttv_25",
    "duration_s",
    "duration_ratio",
    "ttv_27_ratio",
    "cycle_age",
)


@dataclass(frozen=True)
class RawBattery:
    """Discharge-cycle summaries before an EOL protocol is applied."""

    name: str
    cycle_index: np.ndarray
    capacity_raw: np.ndarray
    capacity_clean: np.ndarray
    rows: tuple[Mapping[str, float], ...]
    init_capacity: float

    def __len__(self) -> int:
        return int(self.capacity_clean.size)


@dataclass(frozen=True)
class BatterySeries:
    """A protocol-materialized battery trajectory.

    ``target_mask`` marks exact RUL labels.  For a censored series it is false
    for every endpoint and ``lower_bound_rul`` is the observed remaining-life
    lower bound that can be used by a one-sided survival loss.
    """

    name: str
    protocol: str
    eol_ah: float
    status: str
    life_cycle: float | None
    observed_end: int
    cycle_index: np.ndarray
    capacity_raw: np.ndarray
    capacity_clean: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    soh: np.ndarray
    rul: np.ndarray
    target_mask: np.ndarray
    lower_bound_rul: np.ndarray
    future_delta_5: np.ndarray
    future_delta_10: np.ndarray
    future_mask_5: np.ndarray
    future_mask_10: np.ndarray
    init_capacity: float
    metadata: Mapping[str, object]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    @property
    def is_censored(self) -> bool:
        return self.status == "right_censored"


@dataclass(frozen=True)
class FeatureScaler:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    train_cells: tuple[str, ...]
    train_row_indices: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return (values - self.mean) / self.scale

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "train_cells": list(self.train_cells),
            "train_row_indices": self.train_row_indices.tolist(),
        }


@dataclass(frozen=True)
class BatteryWindowSet:
    X: np.ndarray
    rul: np.ndarray
    target_mask: np.ndarray
    lower_bound_rul: np.ndarray
    capacity: np.ndarray
    future_delta_5: np.ndarray
    future_delta_10: np.ndarray
    future_mask_5: np.ndarray
    future_mask_10: np.ndarray
    cells: np.ndarray
    endpoints: np.ndarray
    raw_window_indices: np.ndarray
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def exact_count(self) -> int:
        return int(self.target_mask.sum())

    def subset(self, indices: np.ndarray) -> "BatteryWindowSet":
        idx = np.asarray(indices, dtype=np.int64)
        return BatteryWindowSet(
            X=self.X[idx],
            rul=self.rul[idx],
            target_mask=self.target_mask[idx],
            lower_bound_rul=self.lower_bound_rul[idx],
            capacity=self.capacity[idx],
            future_delta_5=self.future_delta_5[idx],
            future_delta_10=self.future_delta_10[idx],
            future_mask_5=self.future_mask_5[idx],
            future_mask_10=self.future_mask_10[idx],
            cells=self.cells[idx],
            endpoints=self.endpoints[idx],
            raw_window_indices=self.raw_window_indices[idx],
            feature_names=self.feature_names,
        )


def _array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _first_finite(value: object) -> float:
    values = _array(value)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite[0])


def _ttv(voltage: np.ndarray, time_s: np.ndarray, threshold: float) -> float:
    hit = np.flatnonzero(voltage <= threshold)
    return float(time_s[hit[0]]) if hit.size else float(time_s[-1])


def _causal_slope(values: np.ndarray, end: int, width: int) -> float:
    start = max(0, end - width + 1)
    segment = values[start : end + 1]
    if segment.size < 2 or np.allclose(segment, segment[0]):
        return 0.0
    x = np.arange(segment.size, dtype=np.float64)
    slope = float(np.polyfit(x, segment, 1)[0])
    return float(np.clip(slope, -10.0, 10.0))


def _clean_capacity(values: np.ndarray) -> np.ndarray:
    """Apply the historical upward-spike rule causally, using past values only."""
    cleaned = np.asarray(values, dtype=np.float64).copy()
    for i in range(1, len(cleaned)):
        if cleaned[i] > cleaned[i - 1] * 1.03:
            cleaned[i] = 0.6 * cleaned[i - 1] + 0.4 * cleaned[i]
    return cleaned


def load_battery_mat(path: str | Path) -> RawBattery:
    """Load one NASA battery MAT file and retain only valid discharge cycles."""
    mat_path = Path(path)
    if not mat_path.exists():
        raise FileNotFoundError(mat_path)
    data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if mat_path.stem not in data:
        raise KeyError(f"MAT file does not contain {mat_path.stem!r}: {mat_path}")
    battery = data[mat_path.stem]
    rows: list[dict[str, float]] = []
    cycle_indices: list[int] = []
    capacities: list[float] = []
    for original_index, cycle in enumerate(np.atleast_1d(battery.cycle)):
        if getattr(cycle, "type", None) != "discharge":
            continue
        record = cycle.data
        if not hasattr(record, "Capacity"):
            continue
        capacity = _first_finite(record.Capacity)
        voltage = _array(record.Voltage_measured)
        current = _array(record.Current_measured)
        temperature = _array(record.Temperature_measured)
        time_s = _array(record.Time)
        if not np.isfinite(capacity) or capacity <= 0:
            continue
        if voltage.size < 8 or time_s.size != voltage.size:
            continue
        if current.size != voltage.size or temperature.size != voltage.size:
            continue
        lo = voltage.size // 4
        hi = max(lo + 1, 3 * voltage.size // 4)
        duration = float(time_s[-1] - time_s[0])
        if not np.isfinite(duration) or duration < 0:
            continue
        rows.append(
            {
                "capacity": float(capacity),
                "voltage_mean": float(np.mean(voltage)),
                "voltage_std": float(np.std(voltage)),
                "voltage_min": float(np.min(voltage)),
                "voltage_mid": float(np.mean(voltage[lo:hi])),
                "voltage_p10": float(np.percentile(voltage, 10)),
                "current_mean_abs": float(np.mean(np.abs(current))),
                "current_std": float(np.std(current)),
                "temperature_mean": float(np.mean(temperature)),
                "temperature_std": float(np.std(temperature)),
                "temperature_max": float(np.max(temperature)),
                "ttv_30": _ttv(voltage, time_s, 3.0),
                "ttv_27": _ttv(voltage, time_s, 2.7),
                "ttv_25": _ttv(voltage, time_s, 2.5),
                "duration_s": duration,
            }
        )
        cycle_indices.append(original_index)
        capacities.append(float(capacity))
    if len(rows) < 25:
        raise ValueError(f"Not enough valid discharge cycles in {mat_path}: {len(rows)}")
    capacity_raw = np.asarray(capacities, dtype=np.float64)
    capacity_clean = _clean_capacity(capacity_raw)
    init = float(np.median(capacity_clean[: min(5, len(capacity_clean))]))
    if not np.isfinite(init) or init <= 0:
        raise ValueError(f"Invalid initial capacity in {mat_path}: {init}")
    for row, value in zip(rows, capacity_clean):
        row["capacity"] = float(value)
    return RawBattery(
        name=mat_path.stem,
        cycle_index=np.asarray(cycle_indices, dtype=np.int64),
        capacity_raw=capacity_raw,
        capacity_clean=capacity_clean,
        rows=tuple(rows),
        init_capacity=init,
    )


def _resolve_eol(raw: RawBattery, protocol: str) -> tuple[float, int | None, str]:
    protocol = protocol.lower()
    if protocol == "strict14":
        eol = STRICT_EOL_AH
    elif protocol == "rel80":
        eol = 0.8 * raw.init_capacity
    elif protocol == "rated75":
        eol = 0.75 * RATED_CAPACITY_AH
    elif protocol == "adaptive":
        strict_hit = np.flatnonzero(raw.capacity_clean <= STRICT_EOL_AH)
        if strict_hit.size:
            eol = STRICT_EOL_AH
        else:
            eol = 0.8 * raw.init_capacity
    else:
        raise ValueError(f"Unsupported battery protocol: {protocol}")
    hit = np.flatnonzero(raw.capacity_clean <= eol)
    return float(eol), (int(hit[0]) if hit.size else None), protocol


def materialize_battery(raw: RawBattery, protocol: str = "strict14") -> BatterySeries:
    """Apply an EOL protocol without inventing a failure for censored cells."""
    eol, hit, protocol = _resolve_eol(raw, protocol)
    observed_end = hit if hit is not None else len(raw) - 1
    row_slice = slice(0, observed_end + 1)
    capacities = raw.capacity_clean[row_slice].copy()
    raw_capacities = raw.capacity_raw[row_slice].copy()
    cycles = raw.cycle_index[row_slice].copy()
    rows = raw.rows[row_slice]
    n = len(capacities)
    indices = np.arange(n, dtype=np.float64)
    soh = capacities / max(raw.init_capacity, 1.0e-8)
    if hit is None:
        status = "right_censored"
        life_cycle = None
        rul = np.full(n, np.nan, dtype=np.float32)
        target_mask = np.zeros(n, dtype=bool)
        lower_bound = (float(n - 1) - indices).astype(np.float32)
    else:
        status = "event_observed"
        life_cycle = float(hit)
        rul = (life_cycle - indices).astype(np.float32)
        target_mask = np.ones(n, dtype=bool)
        lower_bound = rul.copy()

    features: list[list[float]] = []
    future_delta_5 = np.zeros(n, dtype=np.float32)
    future_delta_10 = np.zeros(n, dtype=np.float32)
    future_mask_5 = np.zeros(n, dtype=bool)
    future_mask_10 = np.zeros(n, dtype=bool)
    for i, row in enumerate(rows):
        prefix_rows = rows[: i + 1]
        early_count = min(5, len(prefix_rows))
        early_duration = float(np.median([r["duration_s"] for r in prefix_rows[:early_count]]))
        early_ttv = float(np.median([r["ttv_27"] for r in prefix_rows[:early_count]]))
        early_temp = float(np.median([r["temperature_mean"] for r in prefix_rows[:early_count]]))
        current_soh = float(soh[i])
        features.append(
            [
                current_soh,
                1.0 - current_soh,
                float(capacities[i] - eol),
                float(capacities[max(0, i - 1)] - capacities[i]) if i else 0.0,
                float(capacities[max(0, i - 4)] - capacities[i]) if i >= 4 else 0.0,
                _causal_slope(capacities, i, 5),
                _causal_slope(capacities, i, 10),
                _causal_slope(capacities, i, 20),
                _causal_slope(capacities, i, 40),
                row["voltage_mean"],
                row["voltage_std"],
                row["voltage_min"],
                row["voltage_mid"],
                row["voltage_p10"],
                row["current_mean_abs"],
                row["current_std"],
                row["temperature_mean"],
                row["temperature_std"],
                row["temperature_max"],
                row["temperature_mean"] - early_temp,
                row["ttv_30"],
                row["ttv_27"],
                row["ttv_25"],
                row["duration_s"],
                row["duration_s"] / max(early_duration, 1.0e-8),
                row["ttv_27"] / max(early_ttv, 1.0e-8),
                float(np.log1p(i)),
            ]
        )
        if i + 5 < n:
            future_delta_5[i] = float(capacities[i + 5] - capacities[i])
            future_mask_5[i] = True
        if i + 10 < n:
            future_delta_10[i] = float(capacities[i + 10] - capacities[i])
            future_mask_10[i] = True
    feature_array = np.asarray(features, dtype=np.float64)
    metadata = {
        "raw_cycle_count": len(raw),
        "observed_cycle_count": n,
        "strict14_hit": int(np.flatnonzero(raw.capacity_clean <= STRICT_EOL_AH)[0])
        if np.any(raw.capacity_clean <= STRICT_EOL_AH)
        else None,
        "protocol_hit": hit,
        "censoring": status == "right_censored",
        "eol_rule": protocol,
    }
    return BatterySeries(
        name=raw.name,
        protocol=protocol,
        eol_ah=eol,
        status=status,
        life_cycle=life_cycle,
        observed_end=observed_end,
        cycle_index=cycles,
        capacity_raw=raw_capacities,
        capacity_clean=capacities,
        features=feature_array,
        feature_names=FEATURE_NAMES,
        soh=soh.astype(np.float32),
        rul=rul,
        target_mask=target_mask,
        lower_bound_rul=lower_bound,
        future_delta_5=future_delta_5,
        future_delta_10=future_delta_10,
        future_mask_5=future_mask_5,
        future_mask_10=future_mask_10,
        init_capacity=raw.init_capacity,
        metadata=metadata,
    )


def materialize_all(raws: Mapping[str, RawBattery], protocol: str = "strict14") -> dict[str, BatterySeries]:
    return {name: materialize_battery(raws[name], protocol) for name in raws}


def _feature_positions(feature_names: Sequence[str]) -> np.ndarray:
    unknown = [name for name in feature_names if name not in FEATURE_NAMES]
    if unknown:
        raise KeyError(f"Unknown battery feature(s): {unknown}")
    return np.asarray([FEATURE_NAMES.index(name) for name in feature_names], dtype=np.int64)


def fit_scaler(series: Iterable[BatterySeries], feature_names: Sequence[str]) -> FeatureScaler:
    selected = tuple(feature_names)
    positions = _feature_positions(selected)
    chunks = []
    cells = []
    row_ids = []
    offset = 0
    for item in series:
        values = item.features[:, positions]
        chunks.append(values)
        cells.append(item.name)
        row_ids.extend((item.name, int(i)) for i in range(len(item)))
        offset += len(item)
    if not chunks:
        raise ValueError("Cannot fit battery scaler without training cells")
    matrix = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
    mean = matrix.mean(axis=0, dtype=np.float64)
    scale = matrix.std(axis=0, dtype=np.float64, ddof=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    return FeatureScaler(
        feature_names=selected,
        mean=mean,
        scale=scale,
        train_cells=tuple(cells),
        train_row_indices=np.asarray(row_ids, dtype=object),
    )


def _empty_windows(feature_names: tuple[str, ...]) -> BatteryWindowSet:
    return BatteryWindowSet(
        X=np.empty((0, 0, len(feature_names)), dtype=np.float32),
        rul=np.empty(0, dtype=np.float32),
        target_mask=np.empty(0, dtype=bool),
        lower_bound_rul=np.empty(0, dtype=np.float32),
        capacity=np.empty(0, dtype=np.float32),
        future_delta_5=np.empty(0, dtype=np.float32),
        future_delta_10=np.empty(0, dtype=np.float32),
        future_mask_5=np.empty(0, dtype=bool),
        future_mask_10=np.empty(0, dtype=bool),
        cells=np.empty(0, dtype=object),
        endpoints=np.empty(0, dtype=np.int64),
        raw_window_indices=np.empty((0, 0), dtype=np.int64),
        feature_names=feature_names,
    )


def make_windows(
    series: Sequence[BatterySeries],
    seq_len: int,
    scaler: FeatureScaler | None = None,
    feature_names: Sequence[str] | None = None,
    min_endpoint: int | None = None,
    include_censored: bool = True,
) -> tuple[BatteryWindowSet, FeatureScaler]:
    """Create cell-local windows; no window crosses a cell boundary."""
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    names = tuple(feature_names or FEATURE_NAMES)
    if scaler is None:
        scaler = fit_scaler(series, names)
    positions = _feature_positions(names)
    first_endpoint = max(seq_len - 1, int(min_endpoint or 0))
    xs: list[np.ndarray] = []
    ruls: list[float] = []
    masks: list[bool] = []
    bounds: list[float] = []
    caps: list[float] = []
    d5: list[float] = []
    d10: list[float] = []
    m5: list[bool] = []
    m10: list[bool] = []
    cells: list[str] = []
    endpoints: list[int] = []
    raw_indices: list[np.ndarray] = []
    for item in series:
        if item.name not in scaler.train_cells and scaler is not None:
            # Test cells are intentionally allowed here; the scaler itself is
            # fitted only on the training cells and carries that audit trail.
            pass
        values = scaler.transform(item.features[:, positions]).astype(np.float32)
        for end in range(first_endpoint, len(item)):
            if not include_censored and not bool(item.target_mask[end]):
                continue
            start = end - seq_len + 1
            xs.append(values[start : end + 1])
            ruls.append(float(item.rul[end]) if item.target_mask[end] else float("nan"))
            masks.append(bool(item.target_mask[end]))
            bounds.append(float(item.lower_bound_rul[end]))
            caps.append(float(item.capacity_clean[end]))
            d5.append(float(item.future_delta_5[end]))
            d10.append(float(item.future_delta_10[end]))
            m5.append(bool(item.future_mask_5[end]))
            m10.append(bool(item.future_mask_10[end]))
            cells.append(item.name)
            endpoints.append(end)
            raw_indices.append(np.arange(start, end + 1, dtype=np.int64))
    if not xs:
        return _empty_windows(names), scaler
    return (
        BatteryWindowSet(
            X=np.stack(xs).astype(np.float32),
            rul=np.asarray(ruls, dtype=np.float32),
            target_mask=np.asarray(masks, dtype=bool),
            lower_bound_rul=np.asarray(bounds, dtype=np.float32),
            capacity=np.asarray(caps, dtype=np.float32),
            future_delta_5=np.asarray(d5, dtype=np.float32),
            future_delta_10=np.asarray(d10, dtype=np.float32),
            future_mask_5=np.asarray(m5, dtype=bool),
            future_mask_10=np.asarray(m10, dtype=bool),
            cells=np.asarray(cells, dtype=object),
            endpoints=np.asarray(endpoints, dtype=np.int64),
            raw_window_indices=np.stack(raw_indices),
            feature_names=names,
        ),
        scaler,
    )


def describe_series(series: BatterySeries) -> dict[str, object]:
    return {
        "name": series.name,
        "protocol": series.protocol,
        "status": series.status,
        "eol_ah": series.eol_ah,
        "life_cycle": series.life_cycle,
        "observed_end": series.observed_end,
        "observed_cycles": len(series),
        "init_capacity": series.init_capacity,
        "last_capacity": float(series.capacity_clean[-1]),
        "min_capacity": float(series.capacity_clean.min()),
        "exact_label_count": series.exact_count if hasattr(series, "exact_count") else int(series.target_mask.sum()),
        "metadata": dict(series.metadata),
    }
