from __future__ import annotations

import unittest

import numpy as np

from src.data.reaction_wheel_sim import (
    ALIGNED_OPERATIONAL_FEATURES,
    LOW_ERROR_OBSERVABLE_EXTRA,
    SimulationTable,
    aligned_feature_names,
    low_error_feature_names,
    split_summary,
)


class TestReactionWheelSimulationData(unittest.TestCase):
    def test_feature_tiers_exclude_oracle_fields_from_operational(self) -> None:
        self.assertIn("恒速电流均值_安培", aligned_feature_names("operational"))
        self.assertNotIn("退化程度", aligned_feature_names("operational"))
        self.assertIn("退化程度", aligned_feature_names("oracle"))

    def test_low_error_observable_extra_is_explicit(self) -> None:
        rows = [{
            "甲相带噪电流均值_安培": "1.0",
            "振动健康指标": "0.2",
            "三相电流有效值均值_安培": "1.1",
            "剩余寿命_等级": "2",
            "样本编号": "a",
        }]
        names = low_error_feature_names(rows, "observable")
        self.assertIn("甲相带噪电流均值_安培", names)
        self.assertIn("振动健康指标", names)
        self.assertIn(LOW_ERROR_OBSERVABLE_EXTRA[0], names)
        self.assertNotIn("剩余寿命_等级", names)

    def test_split_summary_reports_group_and_target_range(self) -> None:
        table = SimulationTable(
            name="aligned_operational",
            rows=({}, {}),
            features=np.zeros((2, len(ALIGNED_OPERATIONAL_FEATURES)), dtype=np.float32),
            target=np.asarray([3.0, 0.0], dtype=np.float32),
            feature_names=ALIGNED_OPERATIONAL_FEATURES,
            group_ids=np.asarray(["1", "1"]),
            split=np.asarray(["训练", "训练"]),
        )
        summary = split_summary(table)
        self.assertEqual(summary["groups"], ["1"])
        self.assertEqual(summary["target_min"], 0.0)
        self.assertEqual(summary["target_max"], 3.0)


if __name__ == "__main__":
    unittest.main()
