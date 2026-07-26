from __future__ import annotations

import unittest

import numpy as np

from src.data.battery_strict import (
    FEATURE_NAMES,
    RawBattery,
    fit_scaler,
    make_windows,
    materialize_battery,
)


def synthetic_raw(name: str, *, n: int = 48, eol: float = 1.4, censor: bool = False) -> RawBattery:
    cycle = np.arange(n, dtype=np.int64)
    capacity = 2.0 - 0.014 * cycle
    if censor:
        capacity = np.maximum(capacity, 1.55)
    rows = []
    for i, value in enumerate(capacity):
        rows.append(
            {
                "capacity": float(value),
                "voltage_mean": 4.2 - 0.004 * i,
                "voltage_std": 0.03,
                "voltage_min": 2.5 - 0.002 * i,
                "voltage_mid": 3.8 - 0.003 * i,
                "voltage_p10": 3.2 - 0.003 * i,
                "current_mean_abs": 2.0,
                "current_std": 0.1,
                "temperature_mean": 25.0 + 0.01 * i,
                "temperature_std": 0.2,
                "temperature_max": 26.0 + 0.01 * i,
                "ttv_30": 100.0 - i,
                "ttv_27": 120.0 - i,
                "ttv_25": 140.0 - i,
                "duration_s": 1000.0 - i,
            }
        )
    return RawBattery(
        name=name,
        cycle_index=cycle,
        capacity_raw=capacity.copy(),
        capacity_clean=capacity.copy(),
        rows=tuple(rows),
        init_capacity=2.0,
    )


class TestStrictBatteryData(unittest.TestCase):
    def test_strict_hit_is_exact_and_censoring_never_invents_zero_rul(self) -> None:
        observed = materialize_battery(synthetic_raw("B0005"), "strict14")
        self.assertEqual(observed.status, "event_observed")
        self.assertEqual(observed.eol_ah, 1.4)
        self.assertTrue(np.all(observed.target_mask))
        self.assertEqual(float(observed.rul[-1]), 0.0)

        censored = materialize_battery(synthetic_raw("B0007", censor=True), "strict14")
        self.assertEqual(censored.status, "right_censored")
        self.assertIsNone(censored.life_cycle)
        self.assertFalse(np.any(censored.target_mask))
        self.assertTrue(np.all(np.isnan(censored.rul)))
        self.assertGreater(float(censored.lower_bound_rul[0]), 0.0)

    def test_adaptive_b0007_uses_fixed_rel80_rule(self) -> None:
        censored = materialize_battery(synthetic_raw("B0007", censor=True), "adaptive")
        self.assertEqual(censored.status, "event_observed")
        self.assertAlmostEqual(censored.eol_ah, 1.6)
        self.assertEqual(censored.metadata["eol_rule"], "adaptive")

    def test_features_are_causal(self) -> None:
        first = synthetic_raw("B0005", n=48)
        changed_capacity = first.capacity_clean.copy()
        changed_capacity[30:] -= 0.25
        changed_rows = list(first.rows)
        for i in range(30, len(changed_rows)):
            changed_rows[i] = dict(changed_rows[i], capacity=float(changed_capacity[i]))
        second = RawBattery(
            name="B0005",
            cycle_index=first.cycle_index,
            capacity_raw=changed_capacity.copy(),
            capacity_clean=changed_capacity.copy(),
            rows=tuple(changed_rows),
            init_capacity=first.init_capacity,
        )
        a = materialize_battery(first, "strict14")
        b = materialize_battery(second, "strict14")
        prefix = min(30, len(a), len(b))
        np.testing.assert_allclose(a.features[:prefix], b.features[:prefix])

    def test_scaler_and_windows_are_cell_local(self) -> None:
        train = [materialize_battery(synthetic_raw("B0005"), "strict14"), materialize_battery(synthetic_raw("B0006"), "strict14")]
        test = materialize_battery(synthetic_raw("B0007", censor=True), "strict14")
        scaler = fit_scaler(train, FEATURE_NAMES)
        windows, _ = make_windows(train, seq_len=8, scaler=scaler, min_endpoint=23)
        test_windows, _ = make_windows([test], seq_len=8, scaler=scaler, min_endpoint=23)
        self.assertEqual(set(windows.cells.tolist()), {"B0005", "B0006"})
        self.assertEqual(set(test_windows.cells.tolist()), {"B0007"})
        self.assertEqual(tuple(scaler.train_cells), ("B0005", "B0006"))
        self.assertEqual(windows.raw_window_indices.shape[1], 8)
        self.assertTrue(np.all(test_windows.raw_window_indices[:, -1] == test_windows.endpoints))
        self.assertEqual(windows.X.dtype, np.float32)

    def test_common_endpoint_is_preserved_across_window_lengths(self) -> None:
        item = materialize_battery(synthetic_raw("B0005"), "strict14")
        scaler = fit_scaler([item], FEATURE_NAMES)
        short, _ = make_windows([item], 8, scaler=scaler, min_endpoint=23)
        long, _ = make_windows([item], 24, scaler=scaler, min_endpoint=23)
        self.assertEqual(int(short.endpoints[0]), 23)
        self.assertEqual(int(long.endpoints[0]), 23)
        self.assertEqual(int(short.endpoints[-1]), int(long.endpoints[-1]))


if __name__ == "__main__":
    unittest.main()
