from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.nozzle_strict import (
    ABSOLUTE_TIME,
    CANONICAL_COLUMNS,
    CANONICAL_ENCODING,
    CANONICAL_ROW_COUNT,
    CANONICAL_SHA256,
    DEFAULT_FAILURE_DEPTH_MM,
    FEATURE_TIERS,
    LABEL_UNITS,
    MAIN_FEATURE_TIER,
    PRESSURE,
    get_split_specs,
    load_nozzle_csv,
    prepare_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "nozzle_ablation_full.csv"


class TestStrictNozzleBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_nozzle_csv(CSV_PATH)

    def test_manifest_exact_schema_hash_and_failure_label(self) -> None:
        manifest = self.data.manifest
        self.assertEqual(len(self.data), CANONICAL_ROW_COUNT)
        self.assertEqual(manifest.row_count, CANONICAL_ROW_COUNT)
        self.assertEqual(manifest.encoding, CANONICAL_ENCODING)
        self.assertEqual(manifest.sha256, CANONICAL_SHA256)
        self.assertEqual(manifest.expected_sha256, CANONICAL_SHA256)
        self.assertTrue(manifest.sha256_verified)
        self.assertEqual(manifest.columns, CANONICAL_COLUMNS)
        self.assertEqual(manifest.failure_depth_mm, DEFAULT_FAILURE_DEPTH_MM)
        self.assertEqual(manifest.failure_index, 55)
        self.assertEqual(manifest.failure_time_s, 20.0)
        self.assertEqual(self.data.rul_s[-1], 0.0)
        np.testing.assert_allclose(
            self.data.rul_s,
            self.data.failure_time_s - self.data.time_s,
            rtol=0.0,
            atol=0.0,
        )

        # The final COMSOL value is 0.258498939 mm.  It is accepted only as
        # the final observed failure point under the documented tolerance.
        self.assertGreater(self.data.depth_margin_mm[-1], 0.0)
        self.assertLess(self.data.depth_margin_mm[-1], manifest.failure_depth_atol_mm)
        self.assertIsInstance(manifest.as_dict()["columns"], list)

    def test_split_specs_are_exact_and_raw_disjoint(self) -> None:
        specs = get_split_specs()
        expected = {
            "fold1": ((0, 24), (24, 32), (32, 40), False),
            "fold2": ((0, 32), (32, 40), (40, 48), False),
            "fold3": ((0, 40), (40, 48), (48, 56), True),
        }
        self.assertEqual(set(specs), set(expected))
        for name, (train, val, test, locked) in expected.items():
            spec = specs[name]
            self.assertEqual((spec.train, spec.val, spec.test), (train, val, test))
            self.assertEqual(spec.locked, locked)
            raw_sets = [
                set(spec.indices_for(partition).tolist())
                for partition in ("train", "val", "test")
            ]
            self.assertTrue(raw_sets[0].isdisjoint(raw_sets[1]))
            self.assertTrue(raw_sets[0].isdisjoint(raw_sets[2]))
            self.assertTrue(raw_sets[1].isdisjoint(raw_sets[2]))

    def test_partition_local_windows_have_expected_counts(self) -> None:
        expected_counts = {
            "fold1": (22, 6, 6),
            "fold2": (30, 6, 6),
            "fold3": (38, 6, 6),
        }
        for name, counts in expected_counts.items():
            prepared = prepare_fold(self.data, name, MAIN_FEATURE_TIER, window_size=3)
            self.assertEqual(
                (len(prepared.train), len(prepared.val), len(prepared.test)),
                counts,
            )
            partition_sets = []
            for window_set, bounds in (
                (prepared.train, prepared.split.train),
                (prepared.val, prepared.split.val),
                (prepared.test, prepared.split.test),
            ):
                allowed = set(range(*bounds))
                used = set(window_set.raw_window_indices.ravel().tolist())
                self.assertTrue(used.issubset(allowed))
                self.assertTrue(set(window_set.endpoint_indices.tolist()).issubset(allowed))
                self.assertEqual(window_set.raw_window_indices.shape[1], 3)
                partition_sets.append(used)
            self.assertTrue(partition_sets[0].isdisjoint(partition_sets[1]))
            self.assertTrue(partition_sets[0].isdisjoint(partition_sets[2]))
            self.assertTrue(partition_sets[1].isdisjoint(partition_sets[2]))

    def test_scaler_is_float64_and_fit_on_unique_train_rows_only(self) -> None:
        prepared = prepare_fold(self.data, "fold1", MAIN_FEATURE_TIER, window_size=3)
        scaler = prepared.scaler
        requested = self.data.feature_matrix(prepared.requested_feature_names)
        train_indices = np.arange(0, 24, dtype=np.int64)
        kept_positions = [
            prepared.requested_feature_names.index(name)
            for name in prepared.feature_names
        ]
        expected_train_rows = requested[train_indices][:, kept_positions]

        np.testing.assert_array_equal(scaler.train_indices, train_indices)
        np.testing.assert_allclose(
            scaler.mean,
            expected_train_rows.mean(axis=0, dtype=np.float64),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            scaler.scale,
            expected_train_rows.std(axis=0, dtype=np.float64, ddof=0),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(scaler.mean.dtype, np.float64)
        self.assertEqual(scaler.scale.dtype, np.float64)
        self.assertEqual(prepared.train.X.dtype, np.float64)
        self.assertEqual(prepared.val.X.dtype, np.float64)
        self.assertEqual(prepared.test.X.dtype, np.float64)

        first_window_indices = prepared.train.raw_window_indices[0]
        expected_first_window = scaler.transform(requested[first_window_indices])
        np.testing.assert_allclose(prepared.train.X[0], expected_first_window)

        # A held-out-row fit would be drastically different on this
        # chronological data, proving validation/test observations were not used.
        all_rows_mean = requested[:, kept_positions].mean(axis=0)
        self.assertFalse(np.allclose(scaler.mean, all_rows_mean, rtol=1.0e-8, atol=0.0))

    def test_pressure_is_optional_and_constant_train_column_is_dropped(self) -> None:
        no_pressure = prepare_fold(self.data, "fold1", MAIN_FEATURE_TIER)
        with_pressure = prepare_fold(
            self.data,
            "fold1",
            MAIN_FEATURE_TIER,
            include_pressure=True,
        )
        self.assertNotIn(PRESSURE, no_pressure.requested_feature_names)
        self.assertIn(PRESSURE, with_pressure.requested_feature_names)
        self.assertIn(PRESSURE, with_pressure.scaler.dropped_feature_names)
        self.assertNotIn(PRESSURE, with_pressure.feature_names)
        np.testing.assert_array_equal(
            with_pressure.scaler.train_indices,
            np.arange(*with_pressure.split.train, dtype=np.int64),
        )

    def test_feature_tiers_labels_units_and_delta_t(self) -> None:
        self.assertEqual(
            set(FEATURE_TIERS),
            {"thermal_only", "state_only", "state_aware", "age_aware"},
        )
        self.assertEqual(MAIN_FEATURE_TIER, "state_aware")
        self.assertNotIn(ABSOLUTE_TIME, FEATURE_TIERS[MAIN_FEATURE_TIER])
        self.assertEqual(
            FEATURE_TIERS["age_aware"],
            FEATURE_TIERS["state_aware"] + (ABSOLUTE_TIME,),
        )
        self.assertEqual(
            FEATURE_TIERS["thermal_only"],
            ("solid_temperature_K", "heat_flux_W_m2", "delta_t_s"),
        )
        self.assertEqual(
            FEATURE_TIERS["state_only"],
            ("depth_mm", "ablation_rate_m_s", "delta_t_s"),
        )

        prepared = prepare_fold(self.data, "fold3", MAIN_FEATURE_TIER, window_size=3)
        final = prepared.test
        self.assertEqual(final.endpoint_indices[-1], 55)
        self.assertEqual(final.endpoint_times_s[-1], 20.0)
        self.assertEqual(final.y_rul[-1], 0.0)
        self.assertEqual(final.current_depth[-1], self.data.depth_mm[-1])
        self.assertEqual(final.current_rate[-1], self.data.ablation_rate_m_s[-1])
        self.assertEqual(final.dt[-1], self.data.delta_t_s[-1])
        self.assertEqual(final.dt[-1], 1.0)
        self.assertEqual(LABEL_UNITS["y_rul"], "s")
        self.assertEqual(LABEL_UNITS["physics_rul"], "s")
        self.assertEqual(LABEL_UNITS["depth_margin"], "mm")
        self.assertEqual(LABEL_UNITS["current_rate"], "m/s")
        self.assertEqual(LABEL_UNITS["current_depth"], "mm")
        self.assertEqual(LABEL_UNITS["dt"], "s")
        self.assertEqual(final.feature_units, ("K", "W/m^2", "mm", "m/s", "s"))

    def test_physics_rul_and_time_weights_have_correct_units(self) -> None:
        prepared = prepare_fold(self.data, "fold3", MAIN_FEATURE_TIER, window_size=3)
        for window_set in (prepared.train, prepared.val, prepared.test):
            expected_physics_rul = (
                np.maximum(window_set.depth_margin_mm, 0.0)
                * 1.0e-3
                / window_set.current_rate_m_s
            )
            np.testing.assert_allclose(window_set.physics_rul_s, expected_physics_rul)
            self.assertTrue(np.all(window_set.quadrature_weights_s > 0.0))
            self.assertAlmostEqual(float(window_set.weights.sum()), 1.0, places=15)
            np.testing.assert_allclose(
                window_set.weights,
                window_set.quadrature_weights_s / window_set.quadrature_weights_s.sum(),
            )
            np.testing.assert_array_equal(window_set.time_weights, window_set.weights)

    def test_custom_failure_depth_uses_first_observed_threshold(self) -> None:
        data = load_nozzle_csv(CSV_PATH, failure_depth_mm=0.01)
        self.assertEqual(data.manifest.failure_index, 32)
        self.assertEqual(data.failure_time_s, 0.6)
        self.assertEqual(data.rul_s[32], 0.0)

    def test_missing_or_modified_csv_never_falls_back(self) -> None:
        missing = PROJECT_ROOT / "data" / "raw" / "does_not_exist.csv"
        with self.assertRaises(FileNotFoundError):
            load_nozzle_csv(missing)

        with tempfile.TemporaryDirectory() as directory:
            altered = Path(directory) / "nozzle_ablation_full.csv"
            payload = CSV_PATH.read_bytes()
            altered.write_bytes(payload.replace(b"0.2585", b"0.2586", 1))
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_nozzle_csv(altered)


if __name__ == "__main__":
    unittest.main()
