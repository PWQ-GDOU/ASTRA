"""Regression tests for the N-CMAPSS strict benchmark framework."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from src.data.ncmapss_strict import (
    FEATURE_SETS,
    RUL_CAP,
    describe_split,
    fit_scaler,
    make_synthetic_units,
    make_windows,
)
from src.models.ncmapss_strict import (
    RUL_SCALE,
    build_ncmapss_model,
    count_parameters,
)


class TestNCMAPSSData(unittest.TestCase):
    def setUp(self) -> None:
        self.splits = make_synthetic_units(n_dev=6, n_test=2, rng_seed=42)
        self.dev = self.splits["dev"]
        self.test = self.splits["test"]

    def test_unit_ids_are_disjoint(self) -> None:
        dev_ids = {u.unit_id for u in self.dev}
        test_ids = {u.unit_id for u in self.test}
        self.assertTrue(dev_ids.isdisjoint(test_ids))

    def test_rul_is_nonnegative_and_monotone(self) -> None:
        for unit in self.dev:
            rul = unit.Y
            self.assertTrue(np.all(rul >= 0))
            # RUL should be non-increasing along the trajectory
            self.assertTrue(np.all(np.diff(rul) <= 1e-8))

    def test_scaler_fit_on_dev_only(self) -> None:
        scaler = fit_scaler(self.dev, "physical_with_conditions", dataset="DS01", split="dev")
        self.assertEqual(scaler.fit_split, "dev")
        # Scaler values should differ from full-set stats
        all_units = self.dev + self.test
        all_features = np.concatenate([
            np.concatenate([u.W, u.X_s], axis=1) for u in all_units
        ], axis=0)
        dev_features = np.concatenate([
            np.concatenate([u.W, u.X_s], axis=1) for u in self.dev
        ], axis=0)
        np.testing.assert_allclose(scaler.mean, dev_features.mean(0), rtol=1e-10)
        self.assertFalse(np.allclose(scaler.mean, all_features.mean(0), rtol=1e-5))

    def test_windows_are_unit_local(self) -> None:
        scaler = fit_scaler(self.dev, "physical_with_conditions")
        windows = make_windows(self.dev[:2], scaler, seq_len=10, rul_cap=RUL_CAP)
        # Each window should contain only one unit
        for i in range(len(windows)):
            uid = windows.unit_ids[i]
            endpoint = windows.cycle_ends[i]
            self.assertGreaterEqual(endpoint, 9)  # seq_len - 1

    def test_rul_capping(self) -> None:
        scaler = fit_scaler(self.dev, "physical")
        windows = make_windows(self.dev, scaler, seq_len=10, rul_cap=50)
        self.assertTrue(np.all(windows.Y <= 50))

    def test_feature_sets_have_correct_dimensions(self) -> None:
        from src.data.ncmapss_strict import N_OP_CONDITIONS, N_PHYSICAL_SENSORS, N_VIRTUAL_SENSORS
        expected = {
            "physical": N_PHYSICAL_SENSORS,
            "virtual": N_VIRTUAL_SENSORS,
            "full": N_PHYSICAL_SENSORS + N_VIRTUAL_SENSORS,
            "physical_with_conditions": N_OP_CONDITIONS + N_PHYSICAL_SENSORS,
            "full_with_conditions": N_OP_CONDITIONS + N_PHYSICAL_SENSORS + N_VIRTUAL_SENSORS,
        }
        for fs, expected_dim in expected.items():
            scaler = fit_scaler(self.dev, fs)
            self.assertEqual(len(scaler.feature_names), expected_dim, f"feature_set={fs}")
            windows = make_windows(self.dev[:2], scaler, seq_len=10)
            self.assertEqual(windows.X.shape[2], expected_dim, f"feature_set={fs}")


class TestNCMAPSSModels(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.n_features = 18  # 4 conditions + 14 physical
        self.n_conditions = 4
        self.x = torch.randn(4, 30, self.n_features)

    def test_all_models_output_valid_rul(self) -> None:
        for name in ("ms_tcn", "bigru_attn", "transformer", "physics_gru"):
            with self.subTest(model=name):
                model = build_ncmapss_model(
                    name, self.n_features, self.n_conditions, hidden=32
                )
                out = model(self.x)
                self.assertEqual(tuple(out.rul.shape), (4,))
                self.assertTrue(torch.isfinite(out.rul).all())
                self.assertTrue((out.rul >= 0).all())

    def test_parameter_count_reasonable(self) -> None:
        for name in ("ms_tcn", "bigru_attn", "transformer", "physics_gru"):
            with self.subTest(model=name):
                model = build_ncmapss_model(
                    name, self.n_features, self.n_conditions, hidden=96
                )
                n = count_parameters(model)
                self.assertLess(n, 2_000_000, f"{name} has too many params: {n}")

    def test_condition_norm_changes_output(self) -> None:
        model = build_ncmapss_model("ms_tcn", self.n_features, self.n_conditions, hidden=32)
        model.train()
        out1 = model(self.x).rul.detach()
        x_no_cond = self.x.clone()
        x_no_cond[..., :self.n_conditions] = 0.0
        out2 = model(x_no_cond).rul.detach()
        # Outputs should differ when conditions differ
        self.assertFalse(torch.allclose(out1, out2))


if __name__ == "__main__":
    unittest.main()
