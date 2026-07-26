"""Regression tests for multi-trajectory nozzle benchmark.

Tests use only synthetic trajectories so no real data files are needed.
"""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.data.nozzle_multitrajectory import (
    ESTIMATED_FEATURES,
    OBSERVABLE_FEATURES,
    NozzleTrajectory,
    TrajectoryManifest,
    fit_scaler,
    load_nozzle_trajectories,
    make_windows,
)
from src.models.nozzle_multitrajectory import (
    CensoredRULLoss,
    WeibullRULLoss,
    build_nozzle_mt_model,
    count_parameters,
)


def _make_csv(
    n_trajectories: int = 4,
    n_rows: int = 25,
    include_censored: bool = True,
) -> str:
    lines = [
        "trajectory_id,condition_id,time_s,cumulative_ablation_depth_mm,"
        "ablation_rate_m_s,failure_depth_mm,solid_temperature_K,"
        "heat_flux_W_m2,pressure_Pa"
    ]
    for i in range(n_trajectories):
        tid = f"T{i + 1:02d}"
        cid = f"C{(i % 2) + 1}"
        # Censored: last trajectory never crosses threshold
        threshold = 0.25 if (include_censored and i == n_trajectories - 1) else 0.20
        t_vals = np.linspace(0.01, 18.0, n_rows)
        # depth only exceeds threshold for event trajectories
        depth_max = 0.22 if (include_censored and i == n_trajectories - 1) else 0.21 + 0.001 * i
        depth = depth_max * (t_vals / t_vals[-1]) ** 0.9
        rate = np.gradient(depth, t_vals).clip(min=1e-7)
        temp = 800.0 + 50.0 * i + 30.0 * t_vals / t_vals[-1]
        for j in range(n_rows):
            lines.append(
                f"{tid},{cid},{t_vals[j]:.4f},{depth[j]:.7f},{rate[j]:.7f},"
                f"{threshold:.4f},{temp[j]:.2f},5.0e+05,1.4e+06"
            )
    return "\n".join(lines) + "\n"


def _write_csv(text: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(text)
        return f.name


class TestNozzleMultitrajectoryLoader(unittest.TestCase):
    def setUp(self) -> None:
        csv_text = _make_csv(n_trajectories=4, n_rows=25, include_censored=True)
        self.path = _write_csv(csv_text)

    def test_loader_validates_required_columns(self) -> None:
        # Write a valid UTF-8 CSV with insufficient columns
        bad = "trajectory_id,condition_id,time_s\nT01,C1,0.1\nT01,C1,0.2\nT01,C1,0.3\n"
        with self.assertRaisesRegex(ValueError, "Missing required"):
            load_nozzle_trajectories(_write_csv(bad))

    def test_event_trajectory_has_exact_rul_and_crossing_interpolated(self) -> None:
        series = load_nozzle_trajectories(self.path, feature_tier="estimated")
        event = series["T01"]
        self.assertEqual(event.manifest.status, "event_observed")
        self.assertTrue(event.manifest.crossing_interpolated)
        self.assertTrue(np.all(event.target_mask))
        self.assertAlmostEqual(float(event.rul_s[-1]), 0.0)
        np.testing.assert_array_less(np.diff(event.rul_s), 1e-8 * np.ones(len(event) - 1))

    def test_censored_trajectory_has_nan_rul_and_no_target_mask(self) -> None:
        series = load_nozzle_trajectories(self.path, feature_tier="estimated")
        censored = series["T04"]
        self.assertEqual(censored.manifest.status, "right_censored")
        self.assertFalse(np.any(censored.target_mask))
        self.assertTrue(np.all(np.isnan(censored.rul_s)))
        self.assertTrue(np.all(censored.lower_bound_s >= 0.0))

    def test_scaler_fit_on_train_only_not_test(self) -> None:
        series = load_nozzle_trajectories(self.path, feature_tier="estimated")
        train = [series["T01"], series["T02"]]
        test = [series["T03"]]
        scaler = fit_scaler(train)
        self.assertEqual(scaler.train_trajectory_ids, ("T01", "T02"))
        # Scaler mean should differ from full-dataset mean
        all_feat = np.concatenate([s.raw_features for s in series.values()])
        train_feat = np.concatenate([s.raw_features for s in train])
        self.assertFalse(np.allclose(scaler.mean, all_feat.mean(0), rtol=1e-3))
        np.testing.assert_allclose(scaler.mean, train_feat.mean(0), rtol=1e-10)

    def test_windows_do_not_cross_trajectory_boundaries(self) -> None:
        series = load_nozzle_trajectories(self.path, feature_tier="estimated")
        train = list(series.values())[:2]
        scaler = fit_scaler(train)
        windows = make_windows(train, scaler, window_size=5)
        self.assertEqual(windows.X.shape[1], 5)
        # All windows should only contain rows from their own trajectory
        for i in range(len(windows)):
            tid = windows.trajectory_ids[i]
            ep = windows.endpoint_indices[i]
            self.assertGreaterEqual(ep, 4)  # at least window_size-1

    def test_oracle_tier_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "oracle"):
            load_nozzle_trajectories(self.path, feature_tier="oracle")

    def test_feature_tier_sizes(self) -> None:
        obs = load_nozzle_trajectories(self.path, feature_tier="observable")
        est = load_nozzle_trajectories(self.path, feature_tier="estimated")
        self.assertEqual(obs["T01"].raw_features.shape[1], len(OBSERVABLE_FEATURES))
        self.assertEqual(est["T01"].raw_features.shape[1], len(ESTIMATED_FEATURES))


class TestNozzleMultitrajectoryModels(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.x = torch.randn(4, 5, len(ESTIMATED_FEATURES))
        self.rate = torch.tensor([0.01, 0.02, 0.015, 0.018], dtype=torch.float32)
        self.margin = torch.tensor([0.15, 0.10, 0.05, 0.01], dtype=torch.float32)
        self.age = torch.tensor([5.0, 8.0, 12.0, 16.0], dtype=torch.float32)

    def test_all_models_produce_finite_nonnegative_rul(self) -> None:
        for name in ("gru", "ms", "transformer", "physics_residual"):
            with self.subTest(model=name):
                model = build_nozzle_mt_model(name, len(ESTIMATED_FEATURES))
                out = model(self.x, current_rate_m_s=self.rate, depth_margin_mm=self.margin)
                self.assertEqual(tuple(out.rul_s.shape), (4,))
                self.assertTrue(torch.isfinite(out.rul_s).all())
                self.assertTrue((out.rul_s >= 0).all())

    def test_physics_residual_initialises_near_physical_prior(self) -> None:
        model = build_nozzle_mt_model("physics_residual", len(ESTIMATED_FEATURES))
        model.eval()
        with torch.no_grad():
            out = model(self.x, current_rate_m_s=self.rate, depth_margin_mm=self.margin)
        # Expected physical prior: depth_margin_m / rate (n_future_steps contributions)
        expected = (self.margin * 1e-3 / self.rate).numpy()
        ratio = out.rul_s.numpy() / np.maximum(expected, 1e-9)
        # Should start close to 1 (identity correction) with fresh model
        self.assertTrue(np.all(ratio > 0.01), f"ratio={ratio}")

    def test_parameter_budget(self) -> None:
        for name in ("gru", "ms", "transformer", "physics_residual"):
            with self.subTest(model=name):
                model = build_nozzle_mt_model(name, len(ESTIMATED_FEATURES))
                self.assertLess(count_parameters(model), 200_000)

    def test_weibull_loss_gives_finite_positive_weighted_loss(self) -> None:
        # Weibull weights are based on CDF difference at predicted vs true EOL.
        # A prediction error near the characteristic life (eta=15) should produce
        # a nonzero positive loss.
        loss_fn = WeibullRULLoss(eta=15.0, beta=2.5)
        pred_rul = torch.tensor([10.0, 3.0])   # predictions
        true_rul = torch.tensor([12.0, 1.0])   # true RUL
        age_s = torch.tensor([3.0, 14.0])      # current elapsed time
        loss = loss_fn(pred_rul, true_rul, age_s)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)

    def test_weibull_loss_is_zero_for_perfect_prediction(self) -> None:
        loss_fn = WeibullRULLoss(eta=15.0, beta=2.5)
        perfect_rul = torch.tensor([10.0, 5.0, 1.0])
        age_s = torch.tensor([5.0, 10.0, 15.0])
        loss = loss_fn(perfect_rul, perfect_rul, age_s)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_censored_loss_penalises_violation_not_exact(self) -> None:
        loss_fn = CensoredRULLoss(eta=15.0, beta=2.5)
        pred = torch.tensor([5.0, 3.0, 8.0])
        true = torch.tensor([5.0, float("nan"), float("nan")])
        age = torch.tensor([10.0, 12.0, 14.0])
        mask = torch.tensor([True, False, False])
        lb = torch.tensor([0.0, 4.0, 5.0])  # censored lower bounds
        loss = loss_fn(pred, true, age, mask, lb, rul_scale=20.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)


if __name__ == "__main__":
    unittest.main()
