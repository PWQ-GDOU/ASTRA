"""Auditable FEMTO/PRONOSTIA bearing data for a reaction-wheel proxy.

The source data has no official RUL labels in this project.  Labels therefore
represent ordinal remaining measurements within each complete Learning_set run,
not physical time-to-failure or spacecraft reaction-wheel life.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tempfile
import zipfile
from typing import Iterable, Mapping, Sequence

import numpy as np


BEARING_NAMES = (
    "Bearing1_1",
    "Bearing1_2",
    "Bearing2_1",
    "Bearing2_2",
    "Bearing3_1",
    "Bearing3_2",
)

BASE_FEATURE_NAMES = (
    "acc_x_rms", "acc_x_peak", "acc_x_std", "acc_x_kurtosis", "acc_x_skew",
    "acc_y_rms", "acc_y_peak", "acc_y_std", "acc_y_kurtosis", "acc_y_skew",
)
TREND_FEATURE_NAMES = (
    "acc_x_rms_slope_5", "acc_y_rms_slope_5",
    "acc_x_rms_slope_20", "acc_y_rms_slope_20",
    "acc_x_kurtosis_slope_5", "acc_y_kurtosis_slope_5",
    "acc_x_kurtosis_slope_20", "acc_y_kurtosis_slope_20",
)
FREQUENCY_FEATURE_NAMES = (
    "acc_x_spectral_centroid", "acc_y_spectral_centroid",
    "acc_x_high_band_ratio", "acc_y_high_band_ratio",
)
FEATURE_GROUPS = {
    "base": BASE_FEATURE_NAMES,
    "base_trend": BASE_FEATURE_NAMES + TREND_FEATURE_NAMES,
    "full": BASE_FEATURE_NAMES + TREND_FEATURE_NAMES + FREQUENCY_FEATURE_NAMES,
}


@dataclass(frozen=True)
class FemtoSeries:
    name: str
    features: np.ndarray
    raw_rul: np.ndarray
    life_fraction: np.ndarray
    file_names: tuple[str, ...]
    timestamps_s: np.ndarray
    feature_names: tuple[str, ...]
    endpoint_rule: str = "end_of_learning_run"

    def __len__(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class FemtoScaler:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    train_bearings: tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "train_bearings": list(self.train_bearings),
        }


@dataclass(frozen=True)
class FemtoWindowSet:
    X: np.ndarray
    rul: np.ndarray
    life_fraction: np.ndarray
    bearings: np.ndarray
    endpoints: np.ndarray
    feature_names: tuple[str, ...]
    rul_scale: float

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def subset(self, indices: np.ndarray) -> "FemtoWindowSet":
        idx = np.asarray(indices, dtype=np.int64)
        return FemtoWindowSet(
            X=self.X[idx],
            rul=self.rul[idx],
            life_fraction=self.life_fraction[idx],
            bearings=self.bearings[idx],
            endpoints=self.endpoints[idx],
            feature_names=self.feature_names,
            rul_scale=self.rul_scale,
        )


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def _finite_rows(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw[np.all(np.isfinite(raw), axis=1)]


def _acc_summary(acc: np.ndarray) -> np.ndarray:
    values = []
    for channel in range(acc.shape[1]):
        x = acc[:, channel].astype(np.float64)
        mean = float(np.mean(x))
        std = float(np.std(x))
        scale = std + 1.0e-12
        values.extend([
            float(np.sqrt(np.mean(x * x))),
            float(np.max(np.abs(x))),
            std,
            float(np.mean(((x - mean) / scale) ** 4)),
            float(np.mean(((x - mean) / scale) ** 3)),
        ])
    return np.asarray(values, dtype=np.float32)


def _spectrum_summary(acc: np.ndarray) -> np.ndarray:
    values = []
    for channel in range(acc.shape[1]):
        x = acc[:, channel].astype(np.float64)
        if x.size < 4:
            values.extend([0.0, 0.0])
            continue
        centered = x - np.mean(x)
        power = np.abs(np.fft.rfft(centered)) ** 2
        frequencies = np.fft.rfftfreq(centered.size, d=1.0)
        if power.size > 1:
            power = power[1:]
            frequencies = frequencies[1:]
        total = float(np.sum(power))
        if total <= 1.0e-12:
            values.extend([0.0, 0.0])
            continue
        centroid = float(np.sum(frequencies * power) / total)
        high = float(np.sum(power[frequencies >= 0.25]) / total)
        values.extend([centroid, high])
    return np.asarray(values, dtype=np.float32)


def _slope(history: np.ndarray, end: int, width: int, column: int) -> float:
    start = max(0, end - width + 1)
    values = history[start : end + 1, column]
    if values.size < 2 or np.allclose(values, values[0]):
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    return float(np.polyfit(x, values.astype(np.float64), 1)[0])


def _read_csv(raw_bytes: bytes) -> tuple[np.ndarray, float]:
    raw = _finite_rows(np.genfromtxt(raw_bytes.splitlines(), delimiter=","))
    if raw.size == 0:
        raise ValueError("Empty FEMTO acceleration file")
    if raw.shape[1] >= 6:
        acc = raw[:, 4:6]
        timestamp = raw[0, 0] * 3600.0 + raw[0, 1] * 60.0 + raw[0, 2] + raw[0, 3] * 1.0e-6
    elif raw.shape[1] >= 2:
        acc = raw[:, -2:]
        timestamp = float(raw[0, 0])
    else:
        acc = raw[:, -1:]
        timestamp = float(raw[0, 0]) if raw.shape[1] else 0.0
    if acc.shape[1] == 1:
        acc = np.concatenate([acc, np.zeros_like(acc)], axis=1)
    return acc.astype(np.float32), float(timestamp)


def _materialize(name: str, rows: Sequence[np.ndarray], file_names: Sequence[str], timestamps: Sequence[float], group: str) -> FemtoSeries:
    base = np.stack([_acc_summary(acc) for acc in rows]).astype(np.float32)
    spectrum = np.stack([_spectrum_summary(acc) for acc in rows]).astype(np.float32)
    trend = np.zeros((len(rows), len(TREND_FEATURE_NAMES)), dtype=np.float32)
    for index in range(len(rows)):
        trend[index] = np.asarray([
            _slope(base, index, 5, 0), _slope(base, index, 5, 5),
            _slope(base, index, 20, 0), _slope(base, index, 20, 5),
            _slope(base, index, 5, 3), _slope(base, index, 5, 8),
            _slope(base, index, 20, 3), _slope(base, index, 20, 8),
        ], dtype=np.float32)
    blocks = [base]
    if group in ("base_trend", "full"):
        blocks.append(trend)
    if group == "full":
        blocks.append(spectrum)
    features = np.concatenate(blocks, axis=1).astype(np.float32)
    feature_names = tuple(FEATURE_GROUPS[group])
    n = len(features)
    raw_rul = np.arange(n - 1, -1, -1, dtype=np.float32)
    scale = max(float(n - 1), 1.0)
    return FemtoSeries(
        name=name,
        features=features,
        raw_rul=raw_rul,
        life_fraction=raw_rul / scale,
        file_names=tuple(file_names),
        timestamps_s=np.asarray(timestamps, dtype=np.float64),
        feature_names=feature_names,
    )


def load_femto_zip(path: str | Path, *, feature_group: str = "full", max_files_per_bearing: int | None = None) -> dict[str, FemtoSeries]:
    if feature_group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group: {feature_group}")
    zip_path = Path(path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    with tempfile.TemporaryDirectory(prefix="femto_strict_") as temp:
        work = Path(temp)
        with zipfile.ZipFile(zip_path) as outer:
            nested_names = [name for name in outer.namelist() if name.endswith("FEMTOBearingDataSet.zip")]
            if not nested_names:
                raise KeyError("FEMTOBearingDataSet.zip not found in outer archive")
            outer.extract(nested_names[0], work)
            nested_path = work / nested_names[0]
        with zipfile.ZipFile(nested_path) as middle:
            train_names = [name for name in middle.namelist() if name.endswith("Training_set.zip")]
            if not train_names:
                raise KeyError("Training_set.zip not found in FEMTO archive")
            middle.extract(train_names[0], work)
            train_path = work / train_names[0]
        with zipfile.ZipFile(train_path) as train:
            names = [
                name for name in train.namelist()
                if name.startswith("Learning_set/") and "/acc_" in name and name.endswith(".csv")
            ]
            by_bearing: dict[str, list[str]] = {}
            for item in names:
                parts = item.split("/")
                if len(parts) >= 3:
                    by_bearing.setdefault(parts[1], []).append(item)
            output: dict[str, FemtoSeries] = {}
            for bearing in sorted(by_bearing, key=_natural_key):
                file_names = sorted(by_bearing[bearing], key=_natural_key)
                if max_files_per_bearing is not None and len(file_names) > max_files_per_bearing:
                    indices = np.linspace(0, len(file_names) - 1, max_files_per_bearing).astype(int)
                    file_names = [file_names[int(i)] for i in indices]
                rows, timestamps = [], []
                for file_name in file_names:
                    with train.open(file_name) as handle:
                        acc, timestamp = _read_csv(handle.read())
                    rows.append(acc)
                    timestamps.append(timestamp)
                output[bearing] = _materialize(bearing, rows, file_names, timestamps, feature_group)
            missing = [name for name in BEARING_NAMES if name not in output]
            if missing:
                raise ValueError(f"Missing expected FEMTO bearings: {missing}")
            return output


def fit_scaler(series: Sequence[FemtoSeries], feature_names: Sequence[str]) -> FemtoScaler:
    names = tuple(feature_names)
    indices = [FEATURE_GROUPS["full"].index(name) for name in names]
    rows = np.concatenate([item.features[:, indices] for item in series], axis=0).astype(np.float64)
    mean = rows.mean(axis=0)
    scale = rows.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    return FemtoScaler(names, mean, scale, tuple(item.name for item in series))


def make_windows(
    series: Sequence[FemtoSeries],
    seq_len: int,
    scaler: FemtoScaler,
    *,
    min_endpoint: int | None = None,
    rul_scale: float | None = None,
) -> FemtoWindowSet:
    if not series:
        raise ValueError("At least one bearing is required")
    indices = [FEATURE_GROUPS["full"].index(name) for name in scaler.feature_names]
    xs, ys, fractions, bearings, endpoints = [], [], [], [], []
    scale = float(rul_scale or max(max(float(item.raw_rul[0]) for item in series), 1.0))
    for item in series:
        x = scaler.transform(item.features[:, indices])
        first = max(seq_len - 1, int(min_endpoint or 0))
        for endpoint in range(first, len(item)):
            start = endpoint - seq_len + 1
            if start < 0:
                continue
            xs.append(x[start : endpoint + 1])
            ys.append(item.raw_rul[endpoint])
            fractions.append(item.life_fraction[endpoint])
            bearings.append(item.name)
            endpoints.append(endpoint)
    if not xs:
        raise ValueError("No FEMTO windows generated")
    return FemtoWindowSet(
        X=np.stack(xs).astype(np.float32),
        rul=np.asarray(ys, dtype=np.float32),
        life_fraction=np.asarray(fractions, dtype=np.float32),
        bearings=np.asarray(bearings),
        endpoints=np.asarray(endpoints, dtype=np.int64),
        feature_names=scaler.feature_names,
        rul_scale=scale,
    )


def describe_series(series: FemtoSeries) -> dict[str, object]:
    return {
        "name": series.name,
        "n_records": len(series),
        "feature_names": list(series.feature_names),
        "life_index": int(max(len(series) - 1, 0)),
        "endpoint_rule": series.endpoint_rule,
        "first_file": series.file_names[0] if series.file_names else None,
        "last_file": series.file_names[-1] if series.file_names else None,
        "timestamp_start_s": float(series.timestamps_s[0]) if len(series.timestamps_s) else None,
        "timestamp_end_s": float(series.timestamps_s[-1]) if len(series.timestamps_s) else None,
    }
