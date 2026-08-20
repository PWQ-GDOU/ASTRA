from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.exp_cross_transfer_v3 import (
    PreparedData,
    TargetRULModel,
    assert_holdout_excluded,
    build_target_bundle,
    direction_correlation,
    fit_health_transform,
    select_trial_units,
    train_model,
)


def _record(name: str, *, n: int = 42, offset: float = 0.0):
    from scripts.exp_cross_transfer_v3 import TargetRecord

    raw = offset + np.linspace(0.2, 1.1, n, dtype=np.float32)
    rul = np.arange(n - 1, -1, -1, dtype=np.float32)
    return TargetRecord(
        name=name,
        raw_signal=raw,
        rul=rul,
        exact=np.ones(n, dtype=bool),
        lower=rul.copy(),
        status="event_observed",
        metadata={},
    )


class TestCrossTransferV3(unittest.TestCase):
    def test_health_direction_aligns_with_rul(self) -> None:
        damage = np.linspace(0.0, 1.0, 25)
        health = 1.0 - damage
        rul = 24.0 - 24.0 * damage
        self.assertGreater(direction_correlation(health, rul), 0.99)

    def test_battery_soh_is_converted_to_rising_damage_before_health_transform(self) -> None:
        soh = np.linspace(1.0, 0.75, 25)
        damage = 1.0 - soh
        transform = fit_health_transform({"B0005": damage})
        health = transform.transform(damage)
        rul = np.linspace(24.0, 0.0, 25)
        self.assertGreater(direction_correlation(health, rul), 0.99)

    def test_health_transform_is_train_only(self) -> None:
        transform = fit_health_transform({"train_a": np.array([10.0, 12.0]), "train_b": np.array([14.0])})
        self.assertEqual(transform.train_units, ("train_a", "train_b"))
        self.assertEqual(transform.low, 10.0)
        self.assertEqual(transform.scale, 4.0)
        np.testing.assert_allclose(transform.transform(np.array([10.0, 14.0, 18.0])), [1.0, 0.0, 0.0])

    def test_holdout_does_not_enter_fit_or_validation(self) -> None:
        records = {"A": _record("A"), "B": _record("B", offset=1.0), "H": _record("H", offset=100.0)}
        bundle = build_target_bundle(records, selected=("A", "B"), holdout="H", seq_len=8)
        assert_holdout_excluded("H", bundle.transform, bundle.train, bundle.val)
        self.assertNotIn("H", bundle.transform.train_units)
        self.assertTrue(np.all(bundle.test.x <= 1.0))
        self.assertLess(float(bundle.test.x.mean()), 0.1)

    def test_short_early_prefix_keeps_chronological_validation(self) -> None:
        records = {
            "A": _record("A", n=110),
            "H": _record("H", n=110, offset=1.0),
        }
        bundle = build_target_bundle(records, selected=("A",), holdout="H", seq_len=16)
        self.assertIsNotNone(bundle.val)
        assert bundle.val is not None
        self.assertGreater(len(bundle.val), 0)
        self.assertLess(int(bundle.train.endpoints.max()), int(bundle.val.endpoints.min()))

    def test_trial_selection_is_deterministic(self) -> None:
        available = ("U1", "U2", "U3", "U4")
        first = select_trial_units(available, 2, 1, domain="rw_proxy", holdout="Bearing1_1")
        second = select_trial_units(tuple(reversed(available)), 2, 1, domain="rw_proxy", holdout="Bearing1_1")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)

    def test_source_target_adapter_shapes_and_finite_outputs(self) -> None:
        torch.manual_seed(42)
        model = TargetRULModel()
        x = torch.randn(5, 8, 1)
        output = model(x)
        self.assertEqual(tuple(output.shape), (5,))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(model.projection(torch.zeros(2, 8, 1)).shape[-1], 24)

    def test_frozen_and_unfrozen_transfer_paths_train(self) -> None:
        rng = np.random.default_rng(42)
        x = rng.random((20, 8, 1), dtype=np.float32)
        y = np.linspace(1.0, 0.0, 20, dtype=np.float32)
        train = PreparedData(
            x=x,
            y=y,
            exact=np.ones(20, dtype=bool),
            lower=y.copy(),
            endpoints=np.arange(20, dtype=np.int64),
            units=np.asarray(["U"] * 20, dtype=object),
        )
        val = PreparedData(
            x=x[-4:],
            y=y[-4:],
            exact=np.ones(4, dtype=bool),
            lower=y[-4:].copy(),
            endpoints=np.arange(16, 20, dtype=np.int64),
            units=np.asarray(["U"] * 4, dtype=object),
        )
        for frozen in (True, False):
            with self.subTest(frozen=frozen):
                model, diag = train_model(
                    TargetRULModel(), train, val, device="cpu", seed=42, epochs=3,
                    patience=2, batch_size=8, freeze_encoder=frozen,
                )
                with torch.no_grad():
                    prediction = model(torch.as_tensor(x[:2]))
                self.assertTrue(torch.isfinite(prediction).all())
                self.assertEqual(diag["freeze_encoder"], frozen)


if __name__ == "__main__":
    unittest.main()
