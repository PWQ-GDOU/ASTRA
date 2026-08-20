"""Shared checkpoint and selection helpers for reaction-wheel v2 runs."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_SCHEMA = "reaction_wheel_v2_checkpoint_v1"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def write_heartbeat(path: Path | None, **fields: Any) -> None:
    """Write a small atomic status record for an external supervisor."""
    if path is None:
        return
    payload = {
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **fields,
    }
    atomic_write_json(path, payload)


def parse_csv_values(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def select_names(
    available: Iterable[str],
    requested: tuple[str, ...] | None,
    *,
    label: str,
) -> tuple[str, ...]:
    available_values = tuple(available)
    if requested is None:
        return available_values
    unknown = [value for value in requested if value not in available_values]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; available: {available_values}")
    return requested


def load_checkpoint(
    path: Path,
    *,
    experiment_schema: str,
    seq_len: int,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: tuple[int, ...],
    candidate_names: tuple[str, ...],
    holdout_names: tuple[str, ...],
) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "experiment_schema": experiment_schema,
        "seq_len": seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "seeds": list(seeds),
        "candidate_names": list(candidate_names),
        "holdout_names": list(holdout_names),
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            f"Checkpoint {path} does not match the requested run: {mismatches}"
        )
    results = payload.get("per_fold", {})
    if not isinstance(results, dict):
        raise ValueError(f"Checkpoint {path} has invalid per_fold data")
    return results


def save_checkpoint(
    path: Path,
    *,
    experiment_schema: str,
    seq_len: int,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: tuple[int, ...],
    candidate_names: tuple[str, ...],
    holdout_names: tuple[str, ...],
    rul_scale: float,
    per_fold: dict[str, dict[str, dict[str, Any]]],
) -> None:
    atomic_write_json(
        path,
        {
            "schema": CHECKPOINT_SCHEMA,
            "experiment_schema": experiment_schema,
            "status": "running",
            "seq_len": seq_len,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "seeds": list(seeds),
            "candidate_names": list(candidate_names),
            "holdout_names": list(holdout_names),
            "rul_scale": rul_scale,
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "per_fold": per_fold,
        },
    )
