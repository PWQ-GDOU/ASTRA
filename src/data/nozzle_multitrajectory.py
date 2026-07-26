"""Strict multi-trajectory nozzle-ablation RUL data protocol.

Expected CSV schema (one row per time observation):

``trajectory_id,condition_id,time_s,cumulative_ablation_depth_mm,
ablation_rate_m_s,failure_depth_mm,solid_temperature_K,heat_flux_W_m2,
pressure_Pa``

Optional rows may contain ``geometry_id``, ``material_id``, and additional
online-observable columns.  The loader interpolates the first threshold
crossing.  A trajectory that does not cross its own declared threshold is
right-censored; it never receives a fabricated RUL=0 label.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


REQUIRED_COLUMNS = (
    "trajectory_id",
    "condition_id",
    "time_s",
    "cumulative_ablation_depth_mm",
    "ablation_rate_m_s",
    "failure_depth_mm",
    "solid_temperature_K",
    "heat_flux_W_m2",
    "pressure_Pa",
)

OBSERVABLE_FEATURES = (
    "solid_temperature_K",
    "heat_flux_W_m2",
    "pressure_Pa",
)
ESTIMATED_FEATURES = OBSERVABLE_FEATURES + (
    "cumulative_ablation_depth_mm",
    "ablation_rate_m_s",
)


@dataclass(frozen=True)
class TrajectoryManifest:
    trajectory_id: str
    condition_id: str
    geometry_id: str | None
    material_id: str | None
    status: str
    source_rows: int
    retained_rows: int
    failure_depth_mm: float
    failure_time_s: float | None
    crossing_interpolated: bool
    source_sha256: str


@dataclass(frozen=True)
class NozzleTrajectory:
    manifest: TrajectoryManifest
    time_s: np.ndarray
    raw_features: np.ndarray
    feature_names: tuple[str, ...]
    depth_mm: np.ndarray
    rate_m_s: np.ndarray
    rul_s: np.ndarray
    target_mask: np.ndarray
    lower_bound_s: np.ndarray

    def __len__(self) -> int:
        return int(self.time_s.size)


@dataclass(frozen=True)
class NozzleTrajectoryScaler:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    train_trajectory_ids: tuple[str, ...]

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "train_trajectory_ids": list(self.train_trajectory_ids),
        }


@dataclass(frozen=True)
class NozzleTrajectoryWindows:
    X: np.ndarray
    rul_s: np.ndarray
    target_mask: np.ndarray
    lower_bound_s: np.ndarray
    depth_margin_mm: np.ndarray
    current_rate_m_s: np.ndarray
    endpoint_time_s: np.ndarray
    trajectory_ids: np.ndarray
    condition_ids: np.ndarray
    endpoint_indices: np.ndarray
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def subset(self, indices: np.ndarray) -> "NozzleTrajectoryWindows":
        index = np.asarray(indices, dtype=np.int64)
        return NozzleTrajectoryWindows(
            X=self.X[index],
            rul_s=self.rul_s[index],
            target_mask=self.target_mask[index],
            lower_bound_s=self.lower_bound_s[index],
            depth_margin_mm=self.depth_margin_mm[index],
            current_rate_m_s=self.current_rate_m_s[index],
            endpoint_time_s=self.endpoint_time_s[index],
            trajectory_ids=self.trajectory_ids[index],
            condition_ids=self.condition_ids[index],
            endpoint_indices=self.endpoint_indices[index],
            feature_names=self.feature_names,
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_csv(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "utf-16"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text[:2000] and text.count(",") >= len(REQUIRED_COLUMNS) - 1:
            return text
    raise ValueError(f"Unable to decode nozzle CSV: {path}")


def _number(row: Mapping[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid required value {name!r}: {row.get(name)!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"Non-finite required value {name!r}")
    return value


def _constant(rows: Sequence[Mapping[str, str]], name: str, trajectory_id: str) -> str:
    values = {str(row.get(name, "")).strip() for row in rows}
    if len(values) != 1:
        raise ValueError(f"Trajectory {trajectory_id!r} has non-constant {name!r}: {sorted(values)}")
    return values.pop()


def _interpolate_crossing(
    time_s: np.ndarray,
    features: np.ndarray,
    depth_mm: np.ndarray,
    rate_m_s: np.ndarray,
    threshold: float,
    hit: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
    if hit == 0:
        return time_s[:1], features[:1], depth_mm[:1], rate_m_s[:1], float(time_s[0]), False
    previous = hit - 1
    d0, d1 = float(depth_mm[previous]), float(depth_mm[hit])
    if d1 <= d0:
        raise ValueError("Failure depth crossing is not increasing and cannot be interpolated")
    fraction = float(np.clip((threshold - d0) / (d1 - d0), 0.0, 1.0))
    crossing_time = float(time_s[previous] + fraction * (time_s[hit] - time_s[previous]))
    crossing_features = features[previous] + fraction * (features[hit] - features[previous])
    crossing_rate = float(rate_m_s[previous] + fraction * (rate_m_s[hit] - rate_m_s[previous]))
    retained_time = np.concatenate([time_s[:hit], np.asarray([crossing_time])])
    retained_features = np.vstack([features[:hit], crossing_features])
    retained_depth = np.concatenate([depth_mm[:hit], np.asarray([threshold])])
    retained_rate = np.concatenate([rate_m_s[:hit], np.asarray([crossing_rate])])
    return retained_time, retained_features, retained_depth, retained_rate, crossing_time, True


def load_nozzle_trajectories(
    path: str | Path,
    *,
    feature_tier: str = "estimated",
) -> dict[str, NozzleTrajectory]:
    """Load, validate, and label multi-trajectory nozzle data.

    ``observable`` excludes current depth/rate; ``estimated`` includes them as
    online state estimates.  ``oracle`` is intentionally unsupported here so
    target-derived simulation fields cannot silently enter a formal benchmark.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    text = _decode_csv(csv_path)
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError("No rows in nozzle multi-trajectory CSV")
    missing = [name for name in REQUIRED_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"Missing required nozzle columns: {missing}")
    if feature_tier == "observable":
        feature_names = OBSERVABLE_FEATURES
    elif feature_tier == "estimated":
        feature_names = ESTIMATED_FEATURES
    else:
        raise ValueError("feature_tier must be 'observable' or 'estimated'; oracle columns are forbidden")
    source_hash = sha256_file(csv_path)
    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        trajectory_id = str(row["trajectory_id"]).strip()
        if not trajectory_id:
            raise ValueError("Blank trajectory_id")
        grouped.setdefault(trajectory_id, []).append(row)
    result: dict[str, NozzleTrajectory] = {}
    for trajectory_id, group_rows in grouped.items():
        condition_id = _constant(group_rows, "condition_id", trajectory_id)
        geometry_id = _constant(group_rows, "geometry_id", trajectory_id) if "geometry_id" in group_rows[0] else None
        material_id = _constant(group_rows, "material_id", trajectory_id) if "material_id" in group_rows[0] else None
        threshold_values = np.asarray([_number(row, "failure_depth_mm") for row in group_rows], dtype=np.float64)
        if not np.allclose(threshold_values, threshold_values[0], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Trajectory {trajectory_id!r} has non-constant failure_depth_mm")
        threshold = float(threshold_values[0])
        order = np.argsort(np.asarray([_number(row, "time_s") for row in group_rows], dtype=np.float64))
        ordered = [group_rows[int(index)] for index in order]
        time_s = np.asarray([_number(row, "time_s") for row in ordered], dtype=np.float64)
        if len(time_s) < 3 or not np.all(np.diff(time_s) > 0.0):
            raise ValueError(f"Trajectory {trajectory_id!r} requires at least 3 strictly increasing time observations")
        depth_mm = np.asarray([_number(row, "cumulative_ablation_depth_mm") for row in ordered], dtype=np.float64)
        rate_m_s = np.asarray([_number(row, "ablation_rate_m_s") for row in ordered], dtype=np.float64)
        if np.any(rate_m_s <= 0.0):
            raise ValueError(f"Trajectory {trajectory_id!r} has non-positive ablation_rate_m_s")
        features = np.column_stack([[_number(row, name) for row in ordered] for name in feature_names]).astype(np.float64)
        hits = np.flatnonzero(depth_mm >= threshold)
        if hits.size:
            hit = int(hits[0])
            time_s, features, depth_mm, rate_m_s, failure_time, interpolated = _interpolate_crossing(
                time_s, features, depth_mm, rate_m_s, threshold, hit
            )
            rul_s = np.maximum(failure_time - time_s, 0.0)
            target_mask = np.ones(len(time_s), dtype=bool)
            lower_bound = rul_s.copy()
            status = "event_observed"
        else:
            failure_time = None
            rul_s = np.full(len(time_s), np.nan, dtype=np.float64)
            target_mask = np.zeros(len(time_s), dtype=bool)
            lower_bound = float(time_s[-1]) - time_s
            interpolated = False
            status = "right_censored"
        manifest = TrajectoryManifest(
            trajectory_id=trajectory_id,
            condition_id=condition_id,
            geometry_id=geometry_id,
            material_id=material_id,
            status=status,
            source_rows=len(group_rows),
            retained_rows=len(time_s),
            failure_depth_mm=threshold,
            failure_time_s=failure_time,
            crossing_interpolated=interpolated,
            source_sha256=source_hash,
        )
        result[trajectory_id] = NozzleTrajectory(
            manifest=manifest,
            time_s=time_s.astype(np.float64),
            raw_features=features.astype(np.float64),
            feature_names=tuple(feature_names),
            depth_mm=depth_mm.astype(np.float64),
            rate_m_s=rate_m_s.astype(np.float64),
            rul_s=rul_s.astype(np.float64),
            target_mask=target_mask,
            lower_bound_s=lower_bound.astype(np.float64),
        )
    return dict(sorted(result.items()))


def fit_scaler(trajectories: Sequence[NozzleTrajectory]) -> NozzleTrajectoryScaler:
    if not trajectories:
        raise ValueError("At least one training trajectory is required")
    feature_names = trajectories[0].feature_names
    if any(item.feature_names != feature_names for item in trajectories):
        raise ValueError("Feature tiers differ across trajectories")
    values = np.concatenate([item.raw_features for item in trajectories], axis=0)
    mean = values.mean(axis=0, dtype=np.float64)
    scale = values.std(axis=0, dtype=np.float64)
    scale[scale < 1.0e-12] = 1.0
    return NozzleTrajectoryScaler(feature_names, mean, scale, tuple(item.manifest.trajectory_id for item in trajectories))


def make_windows(
    trajectories: Sequence[NozzleTrajectory],
    scaler: NozzleTrajectoryScaler,
    *,
    window_size: int,
) -> NozzleTrajectoryWindows:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    xs, rul, masks, lower, margins, rates, times, trajectory_ids, condition_ids, endpoint_indices = [], [], [], [], [], [], [], [], [], []
    for item in trajectories:
        values = scaler.transform(item.raw_features)
        for endpoint in range(window_size - 1, len(item)):
            start = endpoint - window_size + 1
            xs.append(values[start : endpoint + 1])
            rul.append(item.rul_s[endpoint])
            masks.append(item.target_mask[endpoint])
            lower.append(item.lower_bound_s[endpoint])
            margins.append(max(item.manifest.failure_depth_mm - item.depth_mm[endpoint], 0.0))
            rates.append(item.rate_m_s[endpoint])
            times.append(item.time_s[endpoint])
            trajectory_ids.append(item.manifest.trajectory_id)
            condition_ids.append(item.manifest.condition_id)
            endpoint_indices.append(endpoint)
    if not xs:
        raise ValueError("No windows generated")
    return NozzleTrajectoryWindows(
        X=np.stack(xs).astype(np.float32),
        rul_s=np.asarray(rul, dtype=np.float32),
        target_mask=np.asarray(masks, dtype=bool),
        lower_bound_s=np.asarray(lower, dtype=np.float32),
        depth_margin_mm=np.asarray(margins, dtype=np.float32),
        current_rate_m_s=np.asarray(rates, dtype=np.float32),
        endpoint_time_s=np.asarray(times, dtype=np.float32),
        trajectory_ids=np.asarray(trajectory_ids),
        condition_ids=np.asarray(condition_ids),
        endpoint_indices=np.asarray(endpoint_indices, dtype=np.int64),
        feature_names=scaler.feature_names,
    )


def describe_trajectory(item: NozzleTrajectory) -> dict[str, object]:
    return {
        "trajectory_id": item.manifest.trajectory_id,
        "condition_id": item.manifest.condition_id,
        "geometry_id": item.manifest.geometry_id,
        "material_id": item.manifest.material_id,
        "status": item.manifest.status,
        "source_rows": item.manifest.source_rows,
        "retained_rows": item.manifest.retained_rows,
        "failure_depth_mm": item.manifest.failure_depth_mm,
        "failure_time_s": item.manifest.failure_time_s,
        "crossing_interpolated": item.manifest.crossing_interpolated,
        "feature_names": list(item.feature_names),
    }
