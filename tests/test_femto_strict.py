from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from src.data.femto_strict import (
    BASE_FEATURE_NAMES,
    BEARING_NAMES,
    FEATURE_GROUPS,
    FemtoSeries,
    fit_scaler,
    load_femto_zip,
    make_windows,
)


def _csv_bytes(index: int) -> bytes:
    rows = []
    for sample in range(32):
        t = float(sample)
        x = 0.1 * index + np.sin(0.1 * t + index)
        y = 0.2 * index + np.cos(0.07 * t + index)
        rows.append(f"0,0,{sample},0,{x:.7f},{y:.7f}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def _nested_femto_zip(path: Path, n_files: int = 24) -> None:
    train_buffer = io.BytesIO()
    with zipfile.ZipFile(train_buffer, "w", zipfile.ZIP_DEFLATED) as train:
        for bearing_index, bearing in enumerate(BEARING_NAMES):
            for index in range(n_files):
                train.writestr(
                    f"Learning_set/{bearing}/acc_{index + 1:05d}.csv",
                    _csv_bytes(bearing_index + index),
                )
    middle_buffer = io.BytesIO()
    with zipfile.ZipFile(middle_buffer, "w", zipfile.ZIP_DEFLATED) as middle:
        middle.writestr("Training_set.zip", train_buffer.getvalue())
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("FEMTOBearingDataSet.zip", middle_buffer.getvalue())


class TestFemtoStrictData(unittest.TestCase):
    def test_nested_loader_preserves_bearings_and_feature_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "femto_bearing.zip"
            _nested_femto_zip(path)
            series = load_femto_zip(path, feature_group="full")
        self.assertEqual(tuple(series), BEARING_NAMES)
        item = series["Bearing1_1"]
        self.assertEqual(len(item), 24)
        self.assertEqual(item.features.shape[1], len(FEATURE_GROUPS["full"]))
        self.assertEqual(item.feature_names, FEATURE_GROUPS["full"])
        self.assertEqual(float(item.raw_rul[0]), 23.0)
        self.assertEqual(float(item.raw_rul[-1]), 0.0)
        self.assertAlmostEqual(float(item.life_fraction[0]), 1.0)
        self.assertAlmostEqual(float(item.life_fraction[-1]), 0.0)
        self.assertEqual(item.file_names[0].split("/")[-1], "acc_00001.csv")

    def test_scaler_is_training_bearing_only_and_windows_are_local(self) -> None:
        def make_series(name: str, offset: float) -> FemtoSeries:
            features = np.arange(24 * len(BASE_FEATURE_NAMES), dtype=np.float32).reshape(24, -1) + offset
            rul = np.arange(23, -1, -1, dtype=np.float32)
            return FemtoSeries(
                name=name,
                features=features,
                raw_rul=rul,
                life_fraction=rul / 23.0,
                file_names=tuple(f"acc_{i:05d}.csv" for i in range(24)),
                timestamps_s=np.arange(24, dtype=np.float64),
                feature_names=BASE_FEATURE_NAMES,
            )

        train = [make_series("Bearing1_1", 0.0), make_series("Bearing1_2", 10.0)]
        test = [make_series("Bearing2_1", 1000.0)]
        scaler = fit_scaler(train, BASE_FEATURE_NAMES)
        windows = make_windows(train, 8, scaler, min_endpoint=7, rul_scale=23.0)
        test_windows = make_windows(test, 8, scaler, min_endpoint=7, rul_scale=23.0)
        self.assertEqual(tuple(scaler.train_bearings), ("Bearing1_1", "Bearing1_2"))
        self.assertEqual(set(windows.bearings.tolist()), {"Bearing1_1", "Bearing1_2"})
        self.assertEqual(set(test_windows.bearings.tolist()), {"Bearing2_1"})
        self.assertEqual(int(windows.endpoints[0]), 7)
        self.assertEqual(tuple(windows.X.shape[1:]), (8, len(BASE_FEATURE_NAMES)))

    def test_feature_group_selection_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "femto_bearing.zip"
            _nested_femto_zip(path, n_files=12)
            base = load_femto_zip(path, feature_group="base")
            trend = load_femto_zip(path, feature_group="base_trend")
        self.assertEqual(base["Bearing1_1"].features.shape[1], len(FEATURE_GROUPS["base"]))
        self.assertEqual(trend["Bearing1_1"].features.shape[1], len(FEATURE_GROUPS["base_trend"]))


if __name__ == "__main__":
    unittest.main()
