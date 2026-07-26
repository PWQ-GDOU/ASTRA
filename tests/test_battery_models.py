from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.exp_battery_strict import (
    Candidate,
    _choose_selection_rows,
    _fit_blend_alpha,
    select_all,
    seed_everything,
    train_model,
)
from src.data.battery_strict import FEATURE_NAMES, fit_scaler, make_windows, materialize_battery
from src.models.battery_strict import build_battery_model, count_parameters
try:
    from test_battery_strict import synthetic_raw
except ImportError:
    from tests.test_battery_strict import synthetic_raw


class TestStrictBatteryModels(unittest.TestCase):
    def setUp(self) -> None:
        self.series = [
            materialize_battery(synthetic_raw("B0005"), "strict14"),
            materialize_battery(synthetic_raw("B0006"), "strict14"),
        ]
        scaler = fit_scaler(self.series, FEATURE_NAMES)
        self.windows, _ = make_windows(self.series, 8, scaler=scaler, min_endpoint=23)

    def test_models_have_finite_nonnegative_rul(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(4, 8, len(FEATURE_NAMES))
        age = torch.arange(4, dtype=torch.float32) + 23.0
        for name in ("gru", "ms", "transformer", "life"):
            with self.subTest(model=name):
                model = build_battery_model(name, len(FEATURE_NAMES))
                output = model(x, age)
                self.assertEqual(tuple(output.rul.shape), (4,))
                self.assertTrue(torch.isfinite(output.rul).all())
                self.assertTrue((output.rul >= 0).all())
                self.assertTrue(torch.isfinite(output.capacity_delta_5).all())
                self.assertTrue(torch.isfinite(output.capacity_delta_10).all())

    def test_parameter_budget_is_small(self) -> None:
        for name in ("gru", "ms", "transformer", "life"):
            with self.subTest(model=name):
                model = build_battery_model(name, len(FEATURE_NAMES))
                self.assertLess(count_parameters(model), 100_000)

    def test_seed_controls_initialization_and_training(self) -> None:
        candidate = Candidate("life_l8", "life", 8)
        train = self.windows.subset(np.arange(0, min(20, len(self.windows))))
        val = self.windows.subset(np.arange(min(20, len(self.windows)), min(30, len(self.windows))))
        model_a, _, _ = train_model(candidate, train, val, "cpu", 42, epochs=2, patience=2)
        state_a = {key: value.detach().clone() for key, value in model_a.state_dict().items()}
        model_b, _, _ = train_model(candidate, train, val, "cpu", 42, epochs=2, patience=2)
        for key, value in model_b.state_dict().items():
            torch.testing.assert_close(value, state_a[key])

    def test_final_fixed_epoch_training_changes_parameters(self) -> None:
        candidate = Candidate("ms_l16", "ms", 16)
        train = self.windows.subset(np.arange(0, min(20, len(self.windows))))
        model_zero = build_battery_model("ms", len(FEATURE_NAMES))
        initial = {key: value.detach().clone() for key, value in model_zero.state_dict().items()}
        trained, _, _ = train_model(candidate, train, None, "cpu", 42, epochs=2)
        self.assertTrue(any(not torch.equal(initial[key], value.detach().cpu()) for key, value in trained.state_dict().items()))

    def test_oof_blend_alpha_is_bounded_and_uses_validation_labels(self) -> None:
        class Validation:
            rul = np.asarray([10.0, 10.0], dtype=np.float32)
            target_mask = np.asarray([True, True])

        alpha = _fit_blend_alpha(
            {"B0005": [np.asarray([8.0, 8.0]), np.asarray([12.0, 12.0])]},
            {"B0005": [np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0])]},
            {"B0005": [Validation()]},
        )
        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)
        self.assertAlmostEqual(alpha, 1.0)

    def test_blend_alpha_can_select_ridge_fallback(self) -> None:
        class Validation:
            rul = np.asarray([10.0, 8.0], dtype=np.float32)
            target_mask = np.asarray([True, True])

        alpha = _fit_blend_alpha(
            {"B0005": [np.asarray([25.0, 20.0])]},
            {"B0005": [np.asarray([10.0, 8.0])]},
            {"B0005": [Validation()]},
        )
        self.assertAlmostEqual(alpha, 0.0)

    def test_neural_and_method_selection_objectives_are_independent(self) -> None:
        rows = [
            {"candidate": {"name": "neural_best"}, "mean_rmse": 8.0, "blend_mean_rmse": 7.5},
            {"candidate": {"name": "method_best"}, "mean_rmse": 9.0, "blend_mean_rmse": 6.0},
        ]
        neural, method = _choose_selection_rows(rows)
        self.assertEqual(neural["candidate"]["name"], "neural_best")
        self.assertEqual(method["candidate"]["name"], "method_best")

    def test_final_training_preserves_frozen_scheduler_budget(self) -> None:
        candidate = Candidate("ms_l16", "ms", 16)
        train = self.windows.subset(np.arange(0, min(20, len(self.windows))))
        trained, best_epoch, _ = train_model(
            candidate,
            train,
            None,
            "cpu",
            42,
            epochs=2,
            scheduler_epoch_budget=8,
        )
        self.assertEqual(best_epoch, 2)
        self.assertEqual(trained.training_metadata["trained_epoch_limit"], 2)
        self.assertEqual(trained.training_metadata["scheduler_epoch_budget"], 8)

    def test_selection_and_final_seeds_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "Selection and final seeds must match"):
            select_all(
                self.series,
                None,
                "cpu",
                (),
                (42,),
                (42, 123),
                epochs=1,
                common_endpoint=23,
            )


if __name__ == "__main__":
    unittest.main()
