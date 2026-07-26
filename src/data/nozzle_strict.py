"""Strict data protocol for the 56-row COMSOL nozzle benchmark.

The public entry points are :func:`load_nozzle_csv`,
:func:`get_split_specs`, and :func:`prepare_fold`.  The implementation reads
only the named columns in the canonical GBK CSV, verifies its SHA256 digest,
and never falls back to a processed array or historical result.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CANONICAL_RELATIVE_PATH = "data/raw/nozzle_ablation_full.csv"
CANONICAL_ENCODING = "gbk"
CANONICAL_ROW_COUNT = 56
CANONICAL_SHA256 = "2d1b4d906b8832e84f802eea449eb1ec0d744911604bd37cbfa70241e807a7c2"
SHA256_MANIFEST: Mapping[str, str] = {
    CANONICAL_RELATIVE_PATH: CANONICAL_SHA256,
}
DEFAULT_FAILURE_DEPTH_MM = 0.2585
FAILURE_DEPTH_ATOL_MM = 2.0e-6
DEFAULT_WINDOW_SIZE = 3

CANONICAL_COLUMNS: tuple[str, ...] = (
    "阶段",
    "累计时间_s",
    "本段推进_s",
    "状态",
    "最小网格质量",
    "喉部流体温度_K",
    "喉部固体温度_K",
    "喉部压力_Pa",
    "内壁热流_W_m2",
    "最大烧蚀速度_m_s",
    "累计烧蚀深度_mm",
    "目标烧蚀深度_mm",
    "相对误差_pct",
)

COLUMN_UNITS: Mapping[str, str] = {
    "阶段": "label",
    "累计时间_s": "s",
    "本段推进_s": "s",
    "状态": "label",
    "最小网格质量": "1",
    "喉部流体温度_K": "K",
    "喉部固体温度_K": "K",
    "喉部压力_Pa": "Pa",
    "内壁热流_W_m2": "W/m^2",
    "最大烧蚀速度_m_s": "m/s",
    "累计烧蚀深度_mm": "mm",
    "目标烧蚀深度_mm": "mm",
    "相对误差_pct": "%",
}

SOLID_TEMPERATURE = "solid_temperature_K"
HEAT_FLUX = "heat_flux_W_m2"
DEPTH = "depth_mm"
ABLATION_RATE = "ablation_rate_m_s"
DELTA_T = "delta_t_s"
ABSOLUTE_TIME = "absolute_time_s"
PRESSURE = "pressure_Pa"

FEATURE_UNITS: Mapping[str, str] = {
    SOLID_TEMPERATURE: "K",
    HEAT_FLUX: "W/m^2",
    DEPTH: "mm",
    ABLATION_RATE: "m/s",
    DELTA_T: "s",
    ABSOLUTE_TIME: "s",
    PRESSURE: "Pa",
}

FEATURE_TIERS: Mapping[str, tuple[str, ...]] = {
    "thermal_only": (SOLID_TEMPERATURE, HEAT_FLUX, DELTA_T),
    "state_only": (DEPTH, ABLATION_RATE, DELTA_T),
    "state_aware": (
        SOLID_TEMPERATURE,
        HEAT_FLUX,
        DEPTH,
        ABLATION_RATE,
        DELTA_T,
    ),
    "age_aware": (
        SOLID_TEMPERATURE,
        HEAT_FLUX,
        DEPTH,
        ABLATION_RATE,
        DELTA_T,
        ABSOLUTE_TIME,
    ),
}
MAIN_FEATURE_TIER = "state_aware"

LABEL_UNITS: Mapping[str, str] = {
    "rul_s": "s",
    "y_rul": "s",
    "y_rul_s": "s",
    "physics_rul": "s",
    "physics_rul_s": "s",
    "depth_margin": "mm",
    "depth_margin_mm": "mm",
    "endpoint_times": "s",
    "endpoint_times_s": "s",
    "current_rate": "m/s",
    "current_rate_m_s": "m/s",
    "current_depth": "mm",
    "current_depth_mm": "mm",
    "dt": "s",
    "dt_s": "s",
    "quadrature_weights": "s",
    "quadrature_weights_s": "s",
    "time_weights": "1",
    "weights": "1",
}


def _readonly(values: Any, dtype: Any) -> NDArray[Any]:
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class DatasetManifest:
    """Identity and label provenance for one strict CSV load."""

    path: str
    canonical_relative_path: str
    encoding: str
    sha256: str
    expected_sha256: str
    sha256_verified: bool
    row_count: int
    columns: tuple[str, ...]
    failure_depth_mm: float
    failure_depth_atol_mm: float
    failure_index: int
    failure_time_s: float

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self.__dict__)


@dataclass(frozen=True)
class NozzleData:
    """Typed canonical observations and strict regression labels."""

    manifest: DatasetManifest
    raw_indices: IntArray
    stages: tuple[str, ...]
    statuses: tuple[str, ...]
    time_s: FloatArray
    delta_t_s: FloatArray
    min_mesh_quality: FloatArray
    fluid_temperature_K: FloatArray
    solid_temperature_K: FloatArray
    pressure_Pa: FloatArray
    heat_flux_W_m2: FloatArray
    ablation_rate_m_s: FloatArray
    depth_mm: FloatArray
    target_depth_mm: FloatArray
    relative_error_pct: FloatArray
    rul_s: FloatArray
    depth_margin_mm: FloatArray

    def __len__(self) -> int:
        return int(self.time_s.size)

    @property
    def failure_depth_mm(self) -> float:
        return self.manifest.failure_depth_mm

    @property
    def failure_time_s(self) -> float:
        return self.manifest.failure_time_s

    @property
    def column_units(self) -> Mapping[str, str]:
        return COLUMN_UNITS

    @property
    def label_units(self) -> Mapping[str, str]:
        return LABEL_UNITS

    def feature(self, name: str) -> FloatArray:
        features: Mapping[str, FloatArray] = {
            SOLID_TEMPERATURE: self.solid_temperature_K,
            HEAT_FLUX: self.heat_flux_W_m2,
            DEPTH: self.depth_mm,
            ABLATION_RATE: self.ablation_rate_m_s,
            DELTA_T: self.delta_t_s,
            ABSOLUTE_TIME: self.time_s,
            PRESSURE: self.pressure_Pa,
        }
        try:
            return features[name]
        except KeyError as exc:
            raise KeyError(f"Unknown nozzle feature {name!r}") from exc

    def feature_matrix(self, feature_names: Sequence[str]) -> FloatArray:
        if not feature_names:
            raise ValueError("At least one feature is required")
        return np.column_stack([self.feature(name) for name in feature_names]).astype(
            np.float64, copy=False
        )


@dataclass(frozen=True)
class SplitSpec:
    """Half-open raw-row ranges for a chronological fold."""

    name: str
    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    locked: bool = False

    @property
    def validation(self) -> tuple[int, int]:
        return self.val

    def range_for(self, partition: str) -> tuple[int, int]:
        if partition == "train":
            return self.train
        if partition in ("val", "validation"):
            return self.val
        if partition == "test":
            return self.test
        raise KeyError(f"Unknown partition {partition!r}")

    def indices_for(self, partition: str) -> IntArray:
        start, stop = self.range_for(partition)
        return np.arange(start, stop, dtype=np.int64)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "train": list(self.train),
            "val": list(self.val),
            "test": list(self.test),
            "locked": self.locked,
        }


_SPLIT_SPECS: Mapping[str, SplitSpec] = {
    "fold1": SplitSpec("fold1", (0, 24), (24, 32), (32, 40)),
    "fold2": SplitSpec("fold2", (0, 32), (32, 40), (40, 48)),
    "fold3": SplitSpec("fold3", (0, 40), (40, 48), (48, 56), locked=True),
}


def get_split_specs() -> dict[str, SplitSpec]:
    """Return fresh mapping of the three raw-disjoint strict split specs."""

    return dict(_SPLIT_SPECS)


@dataclass(frozen=True)
class FeatureScaler:
    """Float64 standardization fitted on unique training rows only."""

    requested_feature_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    dropped_feature_names: tuple[str, ...]
    mean: FloatArray
    scale: FloatArray
    train_indices: IntArray

    @property
    def mean_(self) -> FloatArray:
        return self.mean

    @property
    def scale_(self) -> FloatArray:
        return self.scale

    def transform(self, values: Any) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        expected = len(self.requested_feature_names)
        if array.shape[-1] != expected:
            raise ValueError(
                f"Expected {expected} requested features, got shape {array.shape}"
            )
        keep = [
            self.requested_feature_names.index(name) for name in self.feature_names
        ]
        selected = array[..., keep]
        return (selected - self.mean) / self.scale

    def as_dict(self) -> dict[str, Any]:
        return {
            "dtype": "float64",
            "fit_scope": "unique_train_rows",
            "requested_feature_names": list(self.requested_feature_names),
            "feature_names": list(self.feature_names),
            "dropped_feature_names": list(self.dropped_feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "train_indices": self.train_indices.tolist(),
        }


@dataclass(frozen=True)
class WindowSet:
    """One independently-windowed partition.

    All unit-bearing fields use explicit suffixes.  Short aliases matching the
    benchmark vocabulary are also exposed as read-only properties.
    """

    partition: str
    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    raw_partition_indices: IntArray
    raw_window_indices: IntArray
    X: FloatArray
    y_rul_s: FloatArray
    physics_rul_s: FloatArray
    depth_margin_mm: FloatArray
    endpoint_indices: IntArray
    endpoint_times_s: FloatArray
    current_rate_m_s: FloatArray
    current_depth_mm: FloatArray
    dt_s: FloatArray
    quadrature_weights_s: FloatArray
    weights: FloatArray

    def __len__(self) -> int:
        return int(self.endpoint_indices.size)

    @property
    def y_rul(self) -> FloatArray:
        return self.y_rul_s

    @property
    def physics_rul(self) -> FloatArray:
        return self.physics_rul_s

    @property
    def depth_margin(self) -> FloatArray:
        return self.depth_margin_mm

    @property
    def endpoint_times(self) -> FloatArray:
        return self.endpoint_times_s

    @property
    def current_rate(self) -> FloatArray:
        return self.current_rate_m_s

    @property
    def current_depth(self) -> FloatArray:
        return self.current_depth_mm

    @property
    def dt(self) -> FloatArray:
        return self.dt_s

    @property
    def quadrature_weights(self) -> FloatArray:
        return self.quadrature_weights_s

    @property
    def time_weights(self) -> FloatArray:
        return self.weights

    @property
    def raw_indices(self) -> IntArray:
        return self.raw_partition_indices

    @property
    def units(self) -> Mapping[str, str]:
        return LABEL_UNITS

    def as_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "partition": self.partition,
            "n_windows": len(self),
            "window_size": int(self.X.shape[1]),
            "feature_names": list(self.feature_names),
            "feature_units": list(self.feature_units),
            "units": dict(LABEL_UNITS),
            "raw_partition_indices": self.raw_partition_indices.tolist(),
            "endpoint_indices": self.endpoint_indices.tolist(),
        }
        if include_arrays:
            result.update(
                {
                    "raw_window_indices": self.raw_window_indices.tolist(),
                    "X": self.X.tolist(),
                    "y_rul_s": self.y_rul_s.tolist(),
                    "physics_rul_s": self.physics_rul_s.tolist(),
                    "depth_margin_mm": self.depth_margin_mm.tolist(),
                    "endpoint_times_s": self.endpoint_times_s.tolist(),
                    "current_rate_m_s": self.current_rate_m_s.tolist(),
                    "current_depth_mm": self.current_depth_mm.tolist(),
                    "dt_s": self.dt_s.tolist(),
                    "quadrature_weights_s": self.quadrature_weights_s.tolist(),
                    "weights": self.weights.tolist(),
                }
            )
        return result


@dataclass(frozen=True)
class PreparedFold:
    """Train/validation/test windows and train-only preprocessing state."""

    manifest: DatasetManifest
    split: SplitSpec
    feature_tier: str
    window_size: int
    requested_feature_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    scaler: FeatureScaler
    train: WindowSet
    val: WindowSet
    test: WindowSet

    @property
    def validation(self) -> WindowSet:
        return self.val

    @property
    def partitions(self) -> Mapping[str, WindowSet]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def as_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        return {
            "manifest": self.manifest.as_dict(),
            "split": self.split.as_dict(),
            "feature_tier": self.feature_tier,
            "window_size": self.window_size,
            "requested_feature_names": list(self.requested_feature_names),
            "feature_names": list(self.feature_names),
            "scaler": self.scaler.as_dict(),
            "train": self.train.as_dict(include_arrays=include_arrays),
            "val": self.val.as_dict(include_arrays=include_arrays),
            "test": self.test.as_dict(include_arrays=include_arrays),
        }


def _parse_float(row: Mapping[str, str], column: str, row_number: int) -> float:
    value = row[column].strip()
    if not value:
        raise ValueError(f"Missing numeric value in {column!r} at CSV row {row_number}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value {value!r} in {column!r} at CSV row {row_number}"
        ) from exc
    if not np.isfinite(parsed):
        raise ValueError(f"Non-finite value in {column!r} at CSV row {row_number}")
    return parsed


def _parse_optional_float(
    row: Mapping[str, str], column: str, row_number: int
) -> float:
    value = row[column].strip()
    if not value:
        return float("nan")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value {value!r} in {column!r} at CSV row {row_number}"
        ) from exc
    if not np.isfinite(parsed):
        raise ValueError(f"Non-finite value in {column!r} at CSV row {row_number}")
    return parsed


def load_nozzle_csv(
    path: str | Path,
    failure_depth_mm: float = DEFAULT_FAILURE_DEPTH_MM,
) -> NozzleData:
    """Load and verify the canonical strict nozzle CSV.

    The SHA256, exact 13-column header, 56-row count, units, time ordering, and
    recorded ``本段推进_s`` deltas are validated.  Failure time is the first
    observed threshold crossing; when no exact crossing exists, the final row
    is accepted only if it is within ``FAILURE_DEPTH_ATOL_MM`` of the threshold.
    RUL is exactly ``failure_time_s - current_time_s``.
    """

    csv_path = Path(path)
    payload = csv_path.read_bytes()  # Deliberately raises FileNotFoundError.
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CANONICAL_SHA256:
        raise ValueError(
            "SHA256 mismatch for strict nozzle CSV: "
            f"expected {CANONICAL_SHA256}, got {digest}"
        )
    if not np.isfinite(failure_depth_mm) or failure_depth_mm <= 0.0:
        raise ValueError("failure_depth_mm must be a positive finite value")

    with csv_path.open("r", encoding=CANONICAL_ENCODING, newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != CANONICAL_COLUMNS:
            raise ValueError(
                "Strict nozzle CSV columns do not match the canonical named schema: "
                f"expected {CANONICAL_COLUMNS!r}, got {actual_columns!r}"
            )
        rows = list(reader)

    if len(rows) != CANONICAL_ROW_COUNT:
        raise ValueError(
            f"Strict nozzle CSV must contain {CANONICAL_ROW_COUNT} rows; "
            f"found {len(rows)}"
        )

    stages: list[str] = []
    statuses: list[str] = []
    numeric: dict[str, list[float]] = {
        column: []
        for column in CANONICAL_COLUMNS
        if column not in ("阶段", "状态", "相对误差_pct")
    }
    relative_error: list[float] = []
    for data_index, row in enumerate(rows):
        csv_row = data_index + 2
        stage = row["阶段"].strip()
        status = row["状态"].strip()
        if not stage or not status:
            raise ValueError(f"Missing stage/status label at CSV row {csv_row}")
        stages.append(stage)
        statuses.append(status)
        for column in numeric:
            numeric[column].append(_parse_float(row, column, csv_row))
        relative_error.append(_parse_optional_float(row, "相对误差_pct", csv_row))

    time_s = _readonly(numeric["累计时间_s"], np.float64)
    delta_t_s = _readonly(numeric["本段推进_s"], np.float64)
    depth_mm = _readonly(numeric["累计烧蚀深度_mm"], np.float64)
    target_depth_mm = _readonly(numeric["目标烧蚀深度_mm"], np.float64)

    observed_deltas = np.diff(time_s)
    if np.any(observed_deltas <= 0.0):
        raise ValueError("累计时间_s must be strictly increasing")
    if not np.isclose(delta_t_s[0], 0.0, rtol=0.0, atol=1.0e-15):
        raise ValueError("The first 本段推进_s value must be zero")
    if not np.allclose(delta_t_s[1:], observed_deltas, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("本段推进_s does not match consecutive 累计时间_s deltas")
    if np.any(np.diff(depth_mm) < 0.0):
        raise ValueError("累计烧蚀深度_mm must be nondecreasing")
    if not np.allclose(target_depth_mm, target_depth_mm[0], rtol=0.0, atol=0.0):
        raise ValueError("目标烧蚀深度_mm must be constant in the strict benchmark")

    exact_crossings = np.flatnonzero(depth_mm >= float(failure_depth_mm))
    if exact_crossings.size:
        failure_index = int(exact_crossings[0])
    elif np.isclose(
        depth_mm[-1],
        float(failure_depth_mm),
        rtol=0.0,
        atol=FAILURE_DEPTH_ATOL_MM,
    ):
        failure_index = len(depth_mm) - 1
    else:
        raise ValueError(
            "Failure depth is neither observed nor reached by the final row within "
            f"{FAILURE_DEPTH_ATOL_MM:g} mm tolerance"
        )

    failure_time_s = float(time_s[failure_index])
    rul_s = _readonly(failure_time_s - time_s, np.float64)
    depth_margin_mm = _readonly(float(failure_depth_mm) - depth_mm, np.float64)
    manifest = DatasetManifest(
        path=str(csv_path.resolve()),
        canonical_relative_path=CANONICAL_RELATIVE_PATH,
        encoding=CANONICAL_ENCODING,
        sha256=digest,
        expected_sha256=CANONICAL_SHA256,
        sha256_verified=True,
        row_count=len(rows),
        columns=actual_columns,
        failure_depth_mm=float(failure_depth_mm),
        failure_depth_atol_mm=FAILURE_DEPTH_ATOL_MM,
        failure_index=failure_index,
        failure_time_s=failure_time_s,
    )

    return NozzleData(
        manifest=manifest,
        raw_indices=_readonly(np.arange(len(rows)), np.int64),
        stages=tuple(stages),
        statuses=tuple(statuses),
        time_s=time_s,
        delta_t_s=delta_t_s,
        min_mesh_quality=_readonly(numeric["最小网格质量"], np.float64),
        fluid_temperature_K=_readonly(numeric["喉部流体温度_K"], np.float64),
        solid_temperature_K=_readonly(numeric["喉部固体温度_K"], np.float64),
        pressure_Pa=_readonly(numeric["喉部压力_Pa"], np.float64),
        heat_flux_W_m2=_readonly(numeric["内壁热流_W_m2"], np.float64),
        ablation_rate_m_s=_readonly(numeric["最大烧蚀速度_m_s"], np.float64),
        depth_mm=depth_mm,
        target_depth_mm=target_depth_mm,
        relative_error_pct=_readonly(relative_error, np.float64),
        rul_s=rul_s,
        depth_margin_mm=depth_margin_mm,
    )


def _resolve_split(split: str | SplitSpec) -> SplitSpec:
    if isinstance(split, SplitSpec):
        return split
    aliases = {"locked": "fold3", "fold3_locked": "fold3"}
    key = aliases.get(split, split)
    try:
        return _SPLIT_SPECS[key]
    except KeyError as exc:
        raise KeyError(
            f"Unknown split {split!r}; expected one of {tuple(_SPLIT_SPECS)}"
        ) from exc


def _validate_split(split: SplitSpec, row_count: int, window_size: int) -> None:
    all_sets: list[set[int]] = []
    for partition in ("train", "val", "test"):
        start, stop = split.range_for(partition)
        if not (0 <= start < stop <= row_count):
            raise ValueError(
                f"Invalid {partition} range [{start}, {stop}) for {row_count} rows"
            )
        if stop - start < window_size:
            raise ValueError(
                f"{partition} range has fewer than window_size={window_size} rows"
            )
        all_sets.append(set(range(start, stop)))
    if any(all_sets[i] & all_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Train/validation/test raw ranges must be disjoint")


def _fit_scaler(
    feature_matrix: FloatArray,
    requested_feature_names: tuple[str, ...],
    train_indices: IntArray,
) -> FeatureScaler:
    # This is intentionally indexed by unique raw rows, never by overlapping
    # windows, so rows receive equal influence regardless of window frequency.
    unique_train_indices = np.unique(np.asarray(train_indices, dtype=np.int64))
    train_rows = np.asarray(feature_matrix[unique_train_indices], dtype=np.float64)
    ranges = np.ptp(train_rows, axis=0)
    magnitudes = np.maximum(1.0, np.max(np.abs(train_rows), axis=0))
    # COMSOL exports a nominally constant 1.43 MPa pressure with a 0.007 Pa
    # first-row round-off.  Treat only machine-scale relative excursions as
    # zero variance while computing the decision exclusively from train rows.
    zero_variance = ranges <= 1.0e-8 * magnitudes
    keep_mask = ~zero_variance
    if not np.any(keep_mask):
        raise ValueError("All requested features have zero variance on training rows")

    kept_names = tuple(
        name for name, keep in zip(requested_feature_names, keep_mask) if keep
    )
    dropped_names = tuple(
        name for name, keep in zip(requested_feature_names, keep_mask) if not keep
    )
    kept_rows = train_rows[:, keep_mask]
    mean = kept_rows.mean(axis=0, dtype=np.float64)
    scale = kept_rows.std(axis=0, dtype=np.float64, ddof=0)
    if np.any(scale == 0.0):
        raise RuntimeError("Internal zero-variance feature filtering failure")

    return FeatureScaler(
        requested_feature_names=requested_feature_names,
        feature_names=kept_names,
        dropped_feature_names=dropped_names,
        mean=_readonly(mean, np.float64),
        scale=_readonly(scale, np.float64),
        train_indices=_readonly(unique_train_indices, np.int64),
    )


def _trapezoidal_weights_s(times_s: FloatArray) -> FloatArray:
    count = len(times_s)
    if count == 1:
        return _readonly(np.ones(1, dtype=np.float64), np.float64)
    gaps = np.diff(times_s)
    if np.any(gaps <= 0.0):
        raise ValueError("Window endpoint times must be strictly increasing")
    result = np.empty(count, dtype=np.float64)
    result[0] = 0.5 * gaps[0]
    result[-1] = 0.5 * gaps[-1]
    if count > 2:
        result[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    return _readonly(result, np.float64)


def _make_window_set(
    data: NozzleData,
    partition: str,
    bounds: tuple[int, int],
    feature_matrix: FloatArray,
    scaler: FeatureScaler,
    window_size: int,
) -> WindowSet:
    start, stop = bounds
    raw_partition_indices = np.arange(start, stop, dtype=np.int64)
    local_window_starts = np.arange(0, len(raw_partition_indices) - window_size + 1)
    raw_window_indices = np.stack(
        [raw_partition_indices[offset : offset + window_size] for offset in local_window_starts]
    ).astype(np.int64, copy=False)
    endpoint_indices = raw_window_indices[:, -1]

    raw_windows = feature_matrix[raw_window_indices]
    X = scaler.transform(raw_windows).astype(np.float64, copy=False)
    endpoint_times_s = np.asarray(data.time_s[endpoint_indices], dtype=np.float64)
    quadrature_weights_s = _trapezoidal_weights_s(endpoint_times_s)
    quadrature_total = float(quadrature_weights_s.sum())
    if quadrature_total > 0.0:
        weights = np.asarray(quadrature_weights_s / quadrature_total, dtype=np.float64)
    else:
        weights = np.full(len(endpoint_indices), 1.0 / len(endpoint_indices), dtype=np.float64)

    depth_margin_mm = np.asarray(data.depth_margin_mm[endpoint_indices], dtype=np.float64)
    current_rate_m_s = np.asarray(data.ablation_rate_m_s[endpoint_indices], dtype=np.float64)
    if np.any(current_rate_m_s <= 0.0):
        raise ValueError("Ablation rate must be positive for physics RUL")
    physics_rul_s = np.maximum(depth_margin_mm, 0.0) * 1.0e-3 / current_rate_m_s

    return WindowSet(
        partition=partition,
        feature_names=scaler.feature_names,
        feature_units=tuple(FEATURE_UNITS[name] for name in scaler.feature_names),
        raw_partition_indices=_readonly(raw_partition_indices, np.int64),
        raw_window_indices=_readonly(raw_window_indices, np.int64),
        X=_readonly(X, np.float64),
        y_rul_s=_readonly(data.rul_s[endpoint_indices], np.float64),
        physics_rul_s=_readonly(physics_rul_s, np.float64),
        depth_margin_mm=_readonly(depth_margin_mm, np.float64),
        endpoint_indices=_readonly(endpoint_indices, np.int64),
        endpoint_times_s=_readonly(endpoint_times_s, np.float64),
        current_rate_m_s=_readonly(current_rate_m_s, np.float64),
        current_depth_mm=_readonly(data.depth_mm[endpoint_indices], np.float64),
        dt_s=_readonly(data.delta_t_s[endpoint_indices], np.float64),
        quadrature_weights_s=quadrature_weights_s,
        weights=_readonly(weights, np.float64),
    )


def prepare_fold(
    data: NozzleData,
    split: str | SplitSpec,
    feature_tier: str = MAIN_FEATURE_TIER,
    window_size: int = DEFAULT_WINDOW_SIZE,
    *,
    include_pressure: bool = False,
) -> PreparedFold:
    """Prepare one strict fold without crossing raw partition boundaries.

    Pressure can be appended explicitly.  Every requested column that is
    numerically zero-variance on the split's unique training rows is removed
    before the float64 scaler is fitted; validation and test rows never affect
    this step.
    """

    resolved_split = _resolve_split(split)
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
        raise ValueError("window_size must be a positive integer")
    _validate_split(resolved_split, len(data), window_size)
    try:
        tier_features = FEATURE_TIERS[feature_tier]
    except KeyError as exc:
        raise KeyError(
            f"Unknown feature tier {feature_tier!r}; expected one of {tuple(FEATURE_TIERS)}"
        ) from exc

    requested_feature_names = tier_features + ((PRESSURE,) if include_pressure else ())
    feature_matrix = data.feature_matrix(requested_feature_names)
    train_indices = resolved_split.indices_for("train")
    scaler = _fit_scaler(feature_matrix, requested_feature_names, train_indices)

    train = _make_window_set(
        data,
        "train",
        resolved_split.train,
        feature_matrix,
        scaler,
        window_size,
    )
    val = _make_window_set(
        data,
        "val",
        resolved_split.val,
        feature_matrix,
        scaler,
        window_size,
    )
    test = _make_window_set(
        data,
        "test",
        resolved_split.test,
        feature_matrix,
        scaler,
        window_size,
    )
    return PreparedFold(
        manifest=data.manifest,
        split=resolved_split,
        feature_tier=feature_tier,
        window_size=window_size,
        requested_feature_names=requested_feature_names,
        feature_names=scaler.feature_names,
        scaler=scaler,
        train=train,
        val=val,
        test=test,
    )


__all__ = [
    "ABLATION_RATE",
    "ABSOLUTE_TIME",
    "CANONICAL_COLUMNS",
    "CANONICAL_ENCODING",
    "CANONICAL_RELATIVE_PATH",
    "CANONICAL_ROW_COUNT",
    "CANONICAL_SHA256",
    "COLUMN_UNITS",
    "DEFAULT_FAILURE_DEPTH_MM",
    "DEFAULT_WINDOW_SIZE",
    "DELTA_T",
    "DEPTH",
    "DatasetManifest",
    "FAILURE_DEPTH_ATOL_MM",
    "FEATURE_TIERS",
    "FEATURE_UNITS",
    "FeatureScaler",
    "HEAT_FLUX",
    "LABEL_UNITS",
    "MAIN_FEATURE_TIER",
    "NozzleData",
    "PRESSURE",
    "PreparedFold",
    "SHA256_MANIFEST",
    "SOLID_TEMPERATURE",
    "SplitSpec",
    "WindowSet",
    "get_split_specs",
    "load_nozzle_csv",
    "prepare_fold",
]
