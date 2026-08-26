"""Leakage-safe feature extraction for the public IMS bearing archive.

The IMS archive is used only as a source-domain bearing degradation corpus.
It has no spacecraft or COMSOL labels in this project, so each measurement
file receives an ordinal remaining-measurement label within its test unit.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re
import tempfile
import zipfile
from typing import Mapping

import numpy as np


IMS_DEFAULT_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
IMS_COMMON_CHANNELS = 4


@dataclass(frozen=True)
class IMSUnit:
    name: str
    features: np.ndarray
    raw_rul: np.ndarray
    file_names: tuple[str, ...]
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(self.features.shape[0])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(value: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value))


def _read_numeric(path: Path) -> np.ndarray | None:
    try:
        values = np.genfromtxt(path, delimiter=None, dtype=np.float64)
    except (OSError, ValueError, UnicodeError):
        return None
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] < 32 or values.shape[1] < 2:
        return None
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.shape[0] < 32:
        return None
    return values.astype(np.float32)


def _channel_features(values: np.ndarray, *, max_channels: int = IMS_COMMON_CHANNELS) -> tuple[np.ndarray, tuple[str, ...]]:
    blocks: list[float] = []
    names: list[str] = []
    # The public IMS archive has 8 channels in 1st_test but 4 channels in
    # 2nd_test/3rd_test.  The first four channels are the common measurement
    # schema, so the source protocol fixes the feature space to those channels.
    n_channels = min(int(values.shape[1]), int(max_channels))
    for channel in range(n_channels):
        x = values[:, channel].astype(np.float64)
        mean = float(np.mean(x))
        std = float(np.std(x))
        scale = std + 1.0e-12
        centered = x - mean
        power = np.abs(np.fft.rfft(centered)) ** 2
        frequencies = np.fft.rfftfreq(centered.size, d=1.0)
        if power.size > 1:
            power = power[1:]
            frequencies = frequencies[1:]
        total = float(np.sum(power))
        centroid = float(np.sum(frequencies * power) / total) if total > 1.0e-12 else 0.0
        high_band = float(np.sum(power[frequencies >= 0.25]) / total) if total > 1.0e-12 else 0.0
        prefix = f"ch{channel + 1}"
        current = (
            float(np.sqrt(np.mean(x * x))),
            float(np.max(np.abs(x))),
            std,
            float(np.mean((centered / scale) ** 4)),
            float(np.mean((centered / scale) ** 3)),
            centroid,
            high_band,
        )
        blocks.extend(current)
        names.extend(
            [
                f"{prefix}_rms",
                f"{prefix}_peak",
                f"{prefix}_std",
                f"{prefix}_kurtosis",
                f"{prefix}_skew",
                f"{prefix}_spectral_centroid",
                f"{prefix}_high_band_ratio",
            ]
        )
    return np.asarray(blocks, dtype=np.float32), tuple(names)


def _unit_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).parts
    for part in parts:
        if re.match(r"^[123](st|nd|rd)_test$", part, re.IGNORECASE):
            return part
    return parts[0] if parts else path.parent.name


def _materialize(path: Path):
    if path.is_dir():
        return path, None
    if path.suffix.lower() != ".zip":
        raise ValueError(f"IMS input must be a directory or zip archive: {path}")
    temp = tempfile.TemporaryDirectory(prefix="ims_strict_")
    root = Path(temp.name)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(root)
    return root, temp


def load_ims_archive(path: str | Path, *, max_files_per_unit: int | None = None) -> dict[str, IMSUnit]:
    """Load numeric IMS measurement files into deterministic source units."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    root, temp = _materialize(source)
    try:
        candidates: list[tuple[str, Path, np.ndarray]] = []
        for file_path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: _natural_key(str(item))):
            values = _read_numeric(file_path)
            if values is None:
                continue
            candidates.append((_unit_name(file_path, root), file_path, values))
        if not candidates:
            raise ValueError(f"No numeric IMS measurement files found under {source}")

        grouped: dict[str, list[tuple[Path, np.ndarray]]] = {}
        for unit, file_path, values in candidates:
            grouped.setdefault(unit, []).append((file_path, values))
        output: dict[str, IMSUnit] = {}
        expected_names: tuple[str, ...] | None = None
        for unit in sorted(grouped, key=_natural_key):
            files = sorted(grouped[unit], key=lambda item: _natural_key(str(item[0])))
            if max_files_per_unit is not None and len(files) > max_files_per_unit:
                indices = np.linspace(0, len(files) - 1, int(max_files_per_unit)).astype(int)
                files = [files[int(index)] for index in indices]
            rows: list[np.ndarray] = []
            names: list[str] = []
            for file_path, values in files:
                current, current_names = _channel_features(values)
                if expected_names is None:
                    expected_names = current_names
                if current_names != expected_names:
                    raise ValueError("IMS files have inconsistent channel counts")
                rows.append(current)
                names.append(str(file_path.relative_to(root)))
            features = np.stack(rows).astype(np.float32)
            raw_rul = np.arange(len(features) - 1, -1, -1, dtype=np.float32)
            output[unit] = IMSUnit(
                name=unit,
                features=features,
                raw_rul=raw_rul,
                file_names=tuple(names),
                feature_names=tuple(expected_names or ()),
            )
        if len(output) < 2:
            raise ValueError(f"Expected at least two IMS source units, found {sorted(output)}")
        return output
    finally:
        if temp is not None:
            temp.cleanup()


def load_ims_processed_dir(path: str | Path) -> dict[str, IMSUnit]:
    """Load the supplied processed IMS source artifact without raw-data leakage.

    The accompanying Weibull repository stores the two source runs separately
    as ``x_train_2/y_train_2`` and ``x_train_3/y_train_3``.  The third label
    column is the remaining-life value in days.  The preprocessed 1st run is
    intentionally ignored: it was shuffled and contains the validation/test
    channel variants from the source repository.
    """
    source = Path(path)
    if not source.is_dir():
        raise FileNotFoundError(source)
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Processed IMS loading requires h5py") from exc

    feature_names = tuple(f"spectral_bin_{index:02d}" for index in range(1, 21))
    output: dict[str, IMSUnit] = {}
    for unit, x_name, y_name in (
        ("2nd_test", "x_train_2.hdf5", "y_train_2.hdf5"),
        ("3rd_test", "x_train_3.hdf5", "y_train_3.hdf5"),
    ):
        x_path = source / x_name
        y_path = source / y_name
        if not x_path.is_file() or not y_path.is_file():
            raise FileNotFoundError(f"Missing processed IMS pair: {x_path}, {y_path}")
        with h5py.File(x_path, "r") as x_file, h5py.File(y_path, "r") as y_file:
            x_key = next(iter(x_file.keys()))
            y_key = next(iter(y_file.keys()))
            features = np.asarray(x_file[x_key][:], dtype=np.float32)
            labels = np.asarray(y_file[y_key][:], dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != len(feature_names):
            raise ValueError(f"Unexpected processed IMS feature shape for {unit}: {features.shape}")
        if labels.ndim != 2 or labels.shape[0] != features.shape[0] or labels.shape[1] < 3:
            raise ValueError(f"Unexpected processed IMS label shape for {unit}: {labels.shape}")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(labels[:, 2])):
            raise ValueError(f"Non-finite processed IMS values for {unit}")
        output[unit] = IMSUnit(
            name=unit,
            features=features,
            raw_rul=labels[:, 2].astype(np.float32),
            file_names=tuple(f"{x_name}:{index}" for index in range(len(features))),
            feature_names=feature_names,
        )
    return output


def describe_unit(unit: IMSUnit) -> dict[str, object]:
    return {
        "name": unit.name,
        "n_records": len(unit),
        "feature_count": int(unit.features.shape[1]),
        "feature_names": list(unit.feature_names),
        "rul_min": float(np.min(unit.raw_rul)),
        "rul_max": float(np.max(unit.raw_rul)),
        "first_file": unit.file_names[0] if unit.file_names else None,
        "last_file": unit.file_names[-1] if unit.file_names else None,
        "channel_policy": f"first_{IMS_COMMON_CHANNELS}_common_channels",
    }
