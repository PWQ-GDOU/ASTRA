"""Run inference for a checkpoint produced by clean_benchmark.py."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


ASTRA_ROOT = Path(__file__).resolve().parents[1]
if str(ASTRA_ROOT) not in sys.path:
    sys.path.insert(0, str(ASTRA_ROOT))

from scripts.clean_benchmark import (  # noqa: E402
    _exp_smooth,
    _feature_indices,
    _read_fd,
    _read_dataset_files,
    make_model,
    score_phm08,
)


class StoredConditionNormalizer:
    """Nearest-centre condition assignment using checkpoint-only state."""

    def __init__(self, centers: np.ndarray):
        self.centers = np.asarray(centers, dtype=np.float32)

    def predict(self, settings: np.ndarray) -> np.ndarray:
        distances = ((settings[:, None, :] - self.centers[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)


def build_test_windows(
    raw: np.ndarray,
    feature_indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    seq_len: int,
    condition_centers: np.ndarray | None = None,
    condition_mean: np.ndarray | None = None,
    condition_std: np.ndarray | None = None,
    smooth: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    units: list[int] = []
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    condition = None
    if condition_centers is not None:
        condition = StoredConditionNormalizer(condition_centers)

    for unit in np.unique(raw[:, 0]).astype(np.int64):
        trajectory = raw[raw[:, 0] == unit, 2:][:, feature_indices].astype(np.float32)
        if smooth:
            trajectory = _exp_smooth(trajectory)
        normalized = (trajectory - mean) / std
        if condition is not None and condition_mean is not None and condition_std is not None:
            labels = condition.predict(trajectory[:, :3])
            cluster_mean = np.asarray(condition_mean, dtype=np.float32)
            cluster_std = np.asarray(condition_std, dtype=np.float32)
            for cluster_id in range(cluster_mean.shape[0]):
                mask = labels == cluster_id
                if np.any(mask):
                    normalized[mask, 3:] = (
                        trajectory[mask, 3:] - cluster_mean[cluster_id, 3:]
                    ) / cluster_std[cluster_id, 3:]
        if len(normalized) < seq_len:
            pad = np.repeat(normalized[:1], seq_len - len(normalized), axis=0)
            normalized = np.concatenate([pad, normalized], axis=0)
        windows.append(normalized[-seq_len:])
        units.append(int(unit))

    if not windows:
        raise RuntimeError("No test trajectories found")
    return np.stack(windows).astype(np.float32), np.asarray(units, dtype=np.int64)


def _predict(model: torch.nn.Module, windows: np.ndarray, device: torch.device, batch_size: int):
    tensor = torch.as_tensor(windows, device=device)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            values.append(model(tensor[start : start + batch_size]).squeeze(-1).cpu().numpy())
    return np.concatenate(values).astype(np.float64)


def run_inference(
    checkpoint_path: Path,
    data_root: Path,
    fd: str,
    device: str,
    output_path: Path,
    batch_size: int = 512,
    rul_cap: float | None = None,
    read_fd: Callable = _read_fd,
    make_model: Callable = make_model,
    feature_indices: Callable = _feature_indices,
    dataset_id: str | None = None,
    dataset_files: tuple[str, str, str] | None = None,
) -> dict:
    started = time.perf_counter()
    torch_device = torch.device(device if not device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    config = dict(checkpoint.get("config") or {})
    requested_dataset = dataset_id or fd
    checkpoint_dataset = config.get("dataset_id")
    if checkpoint_dataset and checkpoint_dataset != requested_dataset:
        raise ValueError(f"Checkpoint dataset is {checkpoint_dataset}, requested {requested_dataset}")
    if not dataset_files and not checkpoint_dataset and config.get("fd") and config["fd"] != fd:
        raise ValueError(f"Checkpoint dataset is {config['fd']}, requested {fd}")

    model_name = config.get("model", "v2")
    sensor_mode = config.get("sensor_mode", "all24")
    seq_len = int(config.get("seq_len", 30))
    indices = feature_indices(sensor_mode)
    model = make_model(model_name, len(indices)).to(torch_device)
    model.load_state_dict(checkpoint["model"])

    _, test_raw, targets = (_read_dataset_files(*dataset_files)
                            if dataset_files else read_fd(str(data_root), fd))
    windows, units = build_test_windows(
        raw=test_raw,
        feature_indices=indices,
        mean=checkpoint["mean"],
        std=checkpoint["std"],
        seq_len=seq_len,
        condition_centers=checkpoint.get("condition_centers"),
        condition_mean=checkpoint.get("condition_mean"),
        condition_std=checkpoint.get("condition_std"),
        smooth=sensor_mode == "tts14",
    )
    predictions = _predict(model, windows, torch_device, batch_size)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    effective_cap = rul_cap
    if effective_cap is not None:
        predictions = np.minimum(predictions, effective_cap)
        targets = np.minimum(targets, effective_cap)
    diff = predictions - targets
    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset_id": requested_dataset,
        "fd": fd,
        "device": str(torch_device),
        "config": config,
        "units": units.tolist(),
        "predictions": predictions.tolist(),
        "targets": targets.tolist(),
        "metrics": {
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "mae": float(np.mean(np.abs(diff))),
            "score": score_phm08(targets, predictions),
        },
        "seconds": time.perf_counter() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ASTRA benchmark checkpoint inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--fd", default="FD002")
    parser.add_argument("--dataset-id")
    parser.add_argument("--train-file")
    parser.add_argument("--test-file")
    parser.add_argument("--rul-file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--rul-cap", type=float)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    run_inference(
        checkpoint_path=Path(args.checkpoint),
        data_root=Path(args.data_root),
        fd=args.fd,
        device=args.device,
        output_path=Path(args.output),
        batch_size=args.batch_size,
        rul_cap=args.rul_cap,
        dataset_id=args.dataset_id,
        dataset_files=(args.train_file, args.test_file, args.rul_file)
        if args.train_file and args.test_file and args.rul_file else None,
    )


if __name__ == "__main__":
    main()
