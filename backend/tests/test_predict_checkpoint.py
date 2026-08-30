import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict_checkpoint.py"


def load_module():
    spec = importlib.util.spec_from_file_location("predict_checkpoint", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PredictCheckpointTests(unittest.TestCase):
    def test_parser_accepts_arbitrary_dataset_mapping(self):
        module = load_module()
        args = module.create_parser().parse_args(["--checkpoint", "m.pt", "--dataset-id", "custom", "--test-file", "test.any", "--rul-file", "labels.any", "--output", "out.json"])
        self.assertEqual(args.dataset_id, "custom")
    def test_build_test_windows_reuses_checkpoint_stats_and_left_pads(self):
        module = load_module()
        raw = np.array(
            [
                [1, 1, 10, 20, 30],
                [1, 2, 12, 22, 32],
            ],
            dtype=np.float32,
        )

        windows, units = module.build_test_windows(
            raw=raw,
            feature_indices=np.array([0, 1, 2]),
            mean=np.array([[10, 20, 30]], dtype=np.float32),
            std=np.array([[2, 2, 2]], dtype=np.float32),
            seq_len=3,
        )

        self.assertEqual(units.tolist(), [1])
        np.testing.assert_allclose(
            windows[0],
            np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
        )

    def test_run_inference_loads_checkpoint_config_and_writes_metrics(self):
        module = load_module()

        class MeanModel(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.pt"
            output = root / "result.json"
            torch.save(
                {
                    "model": MeanModel().state_dict(),
                    "mean": np.zeros((1, 3), dtype=np.float32),
                    "std": np.ones((1, 3), dtype=np.float32),
                    "condition_centers": None,
                    "condition_mean": None,
                    "condition_std": None,
                    "config": {
                        "fd": "FD001",
                        "model": "v2",
                        "sensor_mode": "all24",
                        "seq_len": 2,
                        "rul_cap": 125.0,
                    },
                },
                checkpoint,
            )

            module.run_inference(
                checkpoint_path=checkpoint,
                data_root=root,
                fd="FD001",
                device="cpu",
                output_path=output,
                read_fd=lambda _root, _fd: (
                    np.empty((0, 5), dtype=np.float32),
                    np.array([[1, 1, 1, 2, 3], [1, 2, 2, 3, 4]], dtype=np.float32),
                    np.array([2.0], dtype=np.float32),
                ),
                make_model=lambda _name, _features: MeanModel(),
                feature_indices=lambda _mode: np.array([0, 1, 2]),
            )

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["fd"], "FD001")
            self.assertEqual(payload["config"]["seq_len"], 2)
            self.assertEqual(payload["units"], [1])
            self.assertEqual(len(payload["predictions"]), 1)
            self.assertEqual(payload["targets"], [2.0])
            self.assertEqual(set(payload["metrics"]), {"rmse", "mae", "score"})
            self.assertGreaterEqual(payload["seconds"], 0)


if __name__ == "__main__":
    unittest.main()
