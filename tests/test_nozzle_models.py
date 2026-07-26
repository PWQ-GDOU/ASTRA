from __future__ import annotations

import unittest

import torch

from src.models.nozzle_strict import build_nozzle_model


class TestStrictNozzleModels(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.x = torch.randn(4, 3, 5)
        self.physics_rul = torch.tensor([20.0, 10.0, 2.0, 0.0])

    def test_all_models_return_finite_nonnegative_rul(self) -> None:
        for name in ("lstm", "tcn", "transformer", "physics_input", "physics_full"):
            with self.subTest(model=name):
                model = build_nozzle_model(name, n_features=5, hidden=24)
                physics = self.physics_rul if name.startswith("physics") else None
                output = model(self.x, physics)
                self.assertEqual(tuple(output.rul.shape), (4,))
                self.assertTrue(torch.isfinite(output.rul).all())
                self.assertTrue((output.rul >= 0).all())

    def test_physics_model_starts_at_deterministic_baseline(self) -> None:
        for name in ("physics_input", "physics_full"):
            with self.subTest(model=name):
                model = build_nozzle_model(name, n_features=5, hidden=24)
                model.eval()
                with torch.no_grad():
                    output = model(self.x, self.physics_rul)
                torch.testing.assert_close(output.rul, self.physics_rul)

    def test_full_physics_exposes_positive_rate(self) -> None:
        model = build_nozzle_model("physics_full", n_features=5, hidden=24)
        output = model(self.x, self.physics_rul)
        self.assertIsNotNone(output.rate_mm_s)
        self.assertTrue((output.rate_mm_s > 0).all())

    def test_physics_models_require_auditable_input(self) -> None:
        model = build_nozzle_model("physics_input", n_features=5, hidden=24)
        with self.assertRaisesRegex(ValueError, "requires physics_rul"):
            model(self.x)

    def test_small_parameter_budget(self) -> None:
        for name in ("lstm", "tcn", "transformer", "physics_input", "physics_full"):
            model = build_nozzle_model(name, n_features=5, hidden=24)
            params = sum(parameter.numel() for parameter in model.parameters())
            self.assertLess(params, 10_000)


if __name__ == "__main__":
    unittest.main()
