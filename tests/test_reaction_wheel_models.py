from __future__ import annotations

import unittest

import torch

from src.models.reaction_wheel_strict import (
    build_reaction_wheel_model,
    count_parameters,
)


class TestReactionWheelModels(unittest.TestCase):
    def test_models_have_finite_scalar_outputs(self) -> None:
        torch.manual_seed(42)
        x = torch.randn(4, 20, 10)
        for name in ("gru", "ms", "transformer"):
            with self.subTest(model=name):
                model = build_reaction_wheel_model(name, 10)
                output = model(x)
                self.assertEqual(tuple(output.rul.shape), (4,))
                self.assertTrue(torch.isfinite(output.rul).all())

    def test_parameter_budget_is_small(self) -> None:
        for name in ("gru", "ms", "transformer"):
            with self.subTest(model=name):
                self.assertLess(count_parameters(build_reaction_wheel_model(name, 10)), 100_000)


if __name__ == "__main__":
    unittest.main()
