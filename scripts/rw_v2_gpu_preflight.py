"""Small CUDA execution check for the reaction-wheel v2 model family."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.reaction_wheel_strict import build_reaction_wheel_model, count_parameters
from scripts.rw_v2_common import atomic_write_json


def resolve_cuda_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError("GPU preflight requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to fall back to CPU")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(f"Unavailable CUDA device: {value}")
    return device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="large_gru")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--features", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = resolve_cuda_device(args.device)
    torch.cuda.set_device(device)
    device_index = torch.cuda.current_device()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.cuda.reset_peak_memory_stats(device_index)

    model = build_reaction_wheel_model(args.model, args.features).to(device)
    x = torch.randn(args.batch_size, args.seq_len, args.features, device=device)
    target = torch.randn(args.batch_size, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4)

    for _ in range(args.warmup_steps):
        prediction = model(x).rul
        loss = F.smooth_l1_loss(prediction, target, beta=0.05)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize(device_index)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.steps):
        prediction = model(x).rul
        loss = F.smooth_l1_loss(prediction, target, beta=0.05)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    end.record()
    torch.cuda.synchronize(device_index)

    elapsed_ms = start.elapsed_time(end)
    payload = {
        "schema": "reaction_wheel_v2_gpu_preflight_v1",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device_index),
        "model": args.model,
        "parameters": count_parameters(model),
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "features": args.features,
        "steps": args.steps,
        "elapsed_ms": round(elapsed_ms, 3),
        "ms_per_step": round(elapsed_ms / args.steps, 3),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "input_is_cuda": bool(x.is_cuda),
        "parameter_is_cuda": bool(next(model.parameters()).is_cuda),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
