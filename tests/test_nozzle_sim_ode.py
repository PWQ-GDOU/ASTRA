"""Tests for the reduced-order nozzle ablation ODE simulator."""
import unittest
import numpy as np

from src.data.nozzle_sim_ode import (
    NozzleConditionParams,
    _K_AB,
    _CALIB_DEPTH_20S,
    generate_multi_trajectory,
    make_split_assignment,
    simulate_nozzle,
)


class TestCalibration(unittest.TestCase):
    """Reference run must reproduce the COMSOL 20-second benchmark."""

    def setUp(self):
        self.ref = NozzleConditionParams("REF", "C_ref")
        self.rec = simulate_nozzle(self.ref, output_dt_s=1.0)

    def test_k_ab_positive(self):
        self.assertGreater(_K_AB, 0.0)

    def test_reference_depth_at_20s(self):
        """Depth at 20 s must be within 5% of the COMSOL 0.2585 mm target."""
        t = self.rec["time_s"]
        d = self.rec["cumulative_ablation_depth_mm"]
        idx = np.searchsorted(t, 20.0, side="right") - 1
        self.assertGreaterEqual(idx, 0)
        depth_20 = float(d[idx])
        target_mm = _CALIB_DEPTH_20S * 1000.0  # m → mm
        self.assertAlmostEqual(depth_20, target_mm, delta=target_mm * 0.05,
                               msg=f"depth@20s={depth_20:.4f}mm vs target {target_mm:.4f}mm")

    def test_reference_reaches_failure(self):
        self.assertTrue(self.rec["reached_failure"],
                        "Reference run should reach 0.5mm failure depth")

    def test_ablation_rate_positive(self):
        """All recorded rates must be strictly positive."""
        rates = self.rec["ablation_rate_m_s"]
        self.assertTrue(np.all(rates > 0), f"Non-positive rates found: {rates[rates <= 0]}")

    def test_depth_monotone(self):
        d = self.rec["cumulative_ablation_depth_mm"]
        diffs = np.diff(d)
        self.assertTrue(np.all(diffs >= -1e-12),
                        "Cumulative ablation depth must be non-decreasing")

    def test_solid_temp_between_initial_and_inlet(self):
        T_s = self.rec["solid_temperature_K"]
        T_in = self.ref.T_in_K
        T_s0 = self.ref.T_s0_K
        self.assertTrue(np.all(T_s >= T_s0 - 1e-6))
        self.assertTrue(np.all(T_s <= T_in + 1e-6))

    def test_output_schema(self):
        required = {
            "time_s", "cumulative_ablation_depth_mm", "ablation_rate_m_s",
            "solid_temperature_K", "heat_flux_W_m2", "pressure_Pa",
            "failure_depth_mm", "reached_failure", "n_steps",
        }
        self.assertTrue(required.issubset(self.rec.keys()))


class TestConditionScaling(unittest.TestCase):
    """Higher inlet temperature / pressure should shorten component life."""

    def _life(self, **kwargs):
        p = NozzleConditionParams("T", "C", **kwargs)
        r = simulate_nozzle(p, output_dt_s=1.0)
        return float(r["time_s"][-1])

    def test_higher_T_in_shorter_life(self):
        life_lo = self._life(T_in_K=950.0)
        life_hi = self._life(T_in_K=1250.0)
        self.assertLess(life_hi, life_lo,
                        "Higher inlet temperature must reduce component life")

    def test_higher_pressure_shorter_life(self):
        life_lo = self._life(p_ch0_Pa=0.9e6)
        life_hi = self._life(p_ch0_Pa=1.8e6)
        self.assertLess(life_hi, life_lo,
                        "Higher chamber pressure must reduce component life")

    def test_higher_mdot_shorter_life(self):
        life_lo = self._life(mdot_factor=0.7)
        life_hi = self._life(mdot_factor=1.3)
        self.assertLess(life_hi, life_lo,
                        "Higher mass flow rate must reduce component life")


class TestMultiTrajectory(unittest.TestCase):
    """Smoke tests for LHC sampling and multi-trajectory generation."""

    def setUp(self):
        self.trajs = generate_multi_trajectory(n_trajectories=10, seed=42)

    def test_count(self):
        self.assertEqual(len(self.trajs), 10)

    def test_all_have_unique_ids(self):
        ids = [p.trajectory_id for p, _ in self.trajs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_condition_diversity(self):
        """LHC should produce at least 3 distinct condition IDs in 10 runs."""
        cids = {p.condition_id for p, _ in self.trajs}
        self.assertGreaterEqual(len(cids), 3,
                                "LHC sampling should produce diverse conditions")

    def test_life_diversity(self):
        lives = [r["time_s"][-1] for _, r in self.trajs]
        self.assertGreater(max(lives) - min(lives), 10.0,
                           "Condition diversity must yield spread in component life")

    def test_split_assignment(self):
        split = make_split_assignment(self.trajs, val_fraction=0.2, test_fraction=0.2)
        self.assertEqual(len(split), 10)
        self.assertIn("test", split.values())
        self.assertIn("train", split.values())


if __name__ == "__main__":
    unittest.main()
