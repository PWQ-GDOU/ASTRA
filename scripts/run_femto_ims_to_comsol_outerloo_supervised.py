"""Supervise the COMSOL target outer-LOO experiment."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_FILES = (
    "PREFLIGHT.json", "PROTOCOL.json", "REPORT.json", "MACRO.csv", "PREDICTIONS.csv",
    "DATA_MANIFEST.json", "OUTER_FOLD_SUMMARY.csv", "ACCEPTANCE.json",
)


def _status(path: Path, **values: object) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps({"updated_at": time.time(), **values}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _validate(path: Path, sources: set[str], expected_rows: int | None) -> dict[str, object]:
    missing = [name for name in EXPECTED_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")
    report = json.loads((path / "REPORT.json").read_text(encoding="utf-8"))
    actual = set(report.get("protocol", {}).get("source_arms", []))
    if actual != sources:
        raise RuntimeError(f"Unexpected source arms: {sorted(actual)}")
    rows = report.get("macro_rows", [])
    if expected_rows is not None and len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} macro rows, found {len(rows)}")
    required = ("raw_rmse", "mae", "bias", "normalized_rmse", "outer_holdout", "fit_groups", "validation_groups", "test_groups")
    if not rows or any(not all(key in row for key in required) for row in rows):
        raise RuntimeError("Macro rows are incomplete")
    for row in rows:
        if set(row["fit_groups"]) & set(row["validation_groups"]) or set(row["fit_groups"]) & set(row["test_groups"]) or set(row["validation_groups"]) & set(row["test_groups"]):
            raise RuntimeError(f"Split leakage in row: {row}")
        if row["outer_holdout"] not in row["test_groups"]:
            raise RuntimeError("Outer holdout is not the sole test group")
    return {"macro_rows": len(rows), "sources": sorted(actual), "report_elapsed_sec": report.get("elapsed_sec")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/femto_ims_to_comsol_outerloo"))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--heartbeat-sec", type=float, default=30.0)
    args, child_args = parser.parse_known_args()
    if "--" in child_args:
        child_args = child_args[child_args.index("--") + 1:]
    if not child_args:
        child_args = ["--source", "both"]
    sources = {"femto", "ims"} if "both" in child_args else set()
    if not sources:
        for i, value in enumerate(child_args):
            if value == "--source" and i + 1 < len(child_args):
                sources = {child_args[i + 1]}
    full = "--quick" not in child_args
    holdout_arg = next((child_args[i + 1] for i, value in enumerate(child_args[:-1]) if value == "--holdouts"), "")
    fold_count = len([item for item in holdout_arg.split(",") if item]) if holdout_arg else 5
    include_extra_baselines = "--include-extra-baselines" in child_args
    rows_per_source_fold_shot = 44 if include_extra_baselines else 20
    expected_rows = (
        (2 if sources == {"femto", "ims"} else 1) * fold_count * 3 * rows_per_source_fold_shot
        if full
        else None
    )
    root = Path(__file__).resolve().parents[1]
    final = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    status_path = final.with_name(final.name + ".SUPERVISOR_STATUS.json")
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        summary = _validate(final, sources, expected_rows)
        _status(status_path, schema="femto_ims_to_comsol_outerloo_supervisor_v1", state="completed", published_output=str(final), **summary)
        (final / "SUPERVISOR_STATUS.json").write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(json.dumps({"state": "completed", "output": str(final), **summary}, ensure_ascii=False, indent=2)); return
    attempts = []
    for attempt in range(1, max(1, args.max_retries) + 2):
        staging = final.parent / f".{final.name}.attempt_{attempt}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        command = [sys.executable, str(root / "scripts" / "exp_femto_ims_to_comsol_outerloo.py"), *child_args, "--output", str(staging)]
        started = time.monotonic()
        _status(status_path, schema="femto_ims_to_comsol_outerloo_supervisor_v1", state="running", attempt=attempt, command=command, staging=str(staging))
        with (staging / "experiment.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True)
            timed_out = False
            while proc.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > args.timeout_hours * 3600:
                    timed_out = True; proc.kill(); break
                _status(status_path, schema="femto_ims_to_comsol_outerloo_supervisor_v1", state="running", attempt=attempt, pid=proc.pid, elapsed_sec=elapsed, staging=str(staging))
                time.sleep(max(1.0, args.heartbeat_sec))
            return_code = proc.wait()
        try:
            if timed_out: raise TimeoutError(f"Attempt {attempt} exceeded timeout")
            if return_code != 0: raise RuntimeError(f"Attempt {attempt} exited with code {return_code}")
            summary = _validate(staging, sources, expected_rows)
            staging.rename(final)
            _status(status_path, schema="femto_ims_to_comsol_outerloo_supervisor_v1", state="completed", published_output=str(final), attempt=attempt, **summary)
            (final / "SUPERVISOR_STATUS.json").write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(json.dumps({"state": "completed", "output": str(final), **summary}, ensure_ascii=False, indent=2)); return
        except Exception as exc:
            attempts.append({"attempt": attempt, "error": str(exc), "return_code": return_code})
            _status(status_path, schema="femto_ims_to_comsol_outerloo_supervisor_v1", state="retrying" if attempt <= args.max_retries else "failed", attempts=attempts, error=str(exc))
            if attempt > args.max_retries: raise


if __name__ == "__main__":
    main()
