from __future__ import annotations

import unittest

import numpy as np

from src.data.reaction_wheel_sim import ALIGNED_OPERATIONAL_FEATURES, SimulationTable
from scripts.exp_reaction_wheel_sim import (
    Candidate,
    SEQ_LEN,
    _make_windows,
    metric_dict,
    rmse,
)


class TestReactionWheelSimulationModel(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        features = []
        targets = []
        groups = []
        splits = []
        for group in ("1", "2", "3", "4", "5"):
            for cycle in range(32):
                row = {"周期编号": str(cycle)}
                rows.append(row)
                features.append(np.full(len(ALIGNED_OPERATIONAL_FEATURES), float(cycle), dtype=np.float32))
                targets.append(float(31 - cycle))
                groups.append(group)
                splits.append("训练" if group in {"1", "2", "3"} else ("验证" if group == "4" else "测试"))
        self.table = SimulationTable(
            name="aligned_operational",
            rows=tuple(rows),
            features=np.asarray(features, dtype=np.float32),
            target=np.asarray(targets, dtype=np.float32),
            feature_names=ALIGNED_OPERATIONAL_FEATURES,
            group_ids=np.asarray(groups),
            split=np.asarray(splits),
        )

    def test_windows_are_group_local_and_cover_common_endpoint(self) -> None:
        x, y, group, scale, endpoints, mean, std = _make_windows(self.table, ["1", "2"], target_scale=31.0)
        self.assertEqual(x.shape[1], SEQ_LEN)
        # +1 for normalised cycle-position column appended by _make_windows
        self.assertEqual(x.shape[2], len(ALIGNED_OPERATIONAL_FEATURES) + 1)
        self.assertEqual(int(endpoints.min()), SEQ_LEN - 1)
        self.assertEqual(set(group.tolist()), {"1", "2"})
        self.assertAlmostEqual(scale, 31.0)
        # mean / std are fitted on original features only (no cycle-pos)
        self.assertEqual(mean.shape[0], len(ALIGNED_OPERATIONAL_FEATURES))
        self.assertEqual(std.shape[0], len(ALIGNED_OPERATIONAL_FEATURES))

    def test_pre_failure_metric_excludes_zero_rul_tail(self) -> None:
        truth = np.asarray([3.0, 2.0, 1.0, 0.0, 0.0], dtype=np.float32)
        prediction = np.asarray([3.0, 2.0, 2.0, 10.0, 10.0], dtype=np.float32)
        all_metric = metric_dict(truth, prediction, "model")
        pre_metric = metric_dict(truth, prediction, "model", pre_failure=True)
        self.assertEqual(all_metric["n"], 5)
        self.assertEqual(pre_metric["n"], 3)
        self.assertAlmostEqual(pre_metric["rmse"], rmse(truth[:3], prediction[:3]))


if __name__ == "__main__":
    unittest.main()
