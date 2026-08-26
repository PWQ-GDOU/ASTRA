"""Supervise the strict FEMTO/IMS -> COMSOL experiment."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_FILES = ("PREFLIGHT.json", "PROTOCOL.json", "REPORT.json", "MACRO.csv", "PREDICTIONS.csv", "DATA_MANIFEST.json")


def _write_status(path: Path, **values: object) -> None:
    payload = {"updated_at": time.time(), **values}
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _validate_attempt(path: Path, expected_sources: set[str]) -> dict[str, object]:
    missing = [name for name in EXPECTED_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Attempt is missing outputs: {missing}")
    report = json.loads((path / "REPORT.json").read_text(encoding="utf-8"))
    actual_sources = set(report.get("protocol", {}).get("source_arms", []))
    if actual_sources != expected_sources:
        raise RuntimeError(f"Unexpected source arms: {sorted(actual_sources)}")
    rows = report.get("macro_rows", [])
    if len(rows) != 120:
        raise RuntimeError(f"Expected 120 full-run macro rows, found {len(rows)}")
    if not rows or any(not all(key in row for key in ("raw_rmse", "mae", "bias", "normalized_rmse")) for row in rows):
        raise RuntimeError("Macro rows are incomplete")
    return {"macro_rows": len(rows), "sources": sorted(actual_sources), "report_elapsed_sec": report.get("elapsed_sec")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/femto_ims_to_comsol_transfer"))
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--heartbeat-sec", type=float, default=30.0)
    parser.add_argument("--", dest="separator", nargs="?")
    args, child_args = parser.parse_known_args()
    if "--" in child_args:
        child_args = child_args[child_args.index("--") + 1 :]
    if not child_args:
        child_args = ["--source", "both"]
    expected_sources = {"femto", "ims"} if "both" in child_args else set()
    if not expected_sources:
        for index, value in enumerate(child_args):
            if value == "--source" and index + 1 < len(child_args):
                expected_sources = {child_args[index + 1]}
    root = Path(__file__).resolve().parents[1]
    final = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    final.parent.mkdir(parents=True, exist_ok=True)
    status_path = final.with_name(final.name + ".SUPERVISOR_STATUS.json")
    if final.exists():
        try:
            summary = _validate_attempt(final, expected_sources)
        except Exception as exc:
            raise FileExistsError(f"Refusing to overwrite non-complete output: {final}: {exc}") from exc
        _write_status(
            status_path,
            schema="femto_ims_to_comsol_supervisor_v1",
            state="completed",
            attempt="existing",
            published_output=str(final),
            **summary,
        )
        (final / "SUPERVISOR_STATUS.json").write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(json.dumps({"state": "completed", "output": str(final), **summary}, ensure_ascii=False, indent=2))
        return
    attempts = []
    for attempt in range(1, max(1, args.max_retries) + 2):
        staging = final.parent / f".{final.name}.attempt_{attempt}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        command = [
            sys.executable,
            str(root / "scripts" / "exp_femto_ims_to_comsol.py"),
            *child_args,
            "--output", str(staging),
        ]
        log_path = staging / "experiment.log"
        started = time.monotonic()
        _write_status(
            status_path,
            schema="femto_ims_to_comsol_supervisor_v1",
            state="running",
            attempt=attempt,
            command=command,
            staging=str(staging),
            timeout_hours=args.timeout_hours,
        )
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, text=True)
            timed_out = False
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > args.timeout_hours * 3600.0:
                    timed_out = True
                    process.kill()
                    break
                _write_status(
                    status_path,
                    schema="femto_ims_to_comsol_supervisor_v1",
                    state="running",
                    attempt=attempt,
                    pid=process.pid,
                    elapsed_sec=elapsed,
                    timed_out=False,
                    staging=str(staging),
                )
                time.sleep(max(1.0, args.heartbeat_sec))
            return_code = process.wait()
        try:
            if timed_out:
                raise TimeoutError(f"Attempt {attempt} exceeded {args.timeout_hours} hours")
            if return_code != 0:
                raise RuntimeError(f"Attempt {attempt} exited with code {return_code}")
            summary = _validate_attempt(staging, expected_sources)
            staging.rename(final)
            _write_status(
                status_path,
                schema="femto_ims_to_comsol_supervisor_v1",
                state="completed",
                attempt=attempt,
                supervisor_elapsed_sec=time.monotonic() - started,
                published_output=str(final),
                **summary,
            )
            (final / "SUPERVISOR_STATUS.json").write_text(status_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(json.dumps({"state": "completed", "output": str(final), **summary}, ensure_ascii=False, indent=2))
            return
        except Exception as exc:
            attempts.append({"attempt": attempt, "error": str(exc), "return_code": return_code})
            _write_status(
                status_path,
                schema="femto_ims_to_comsol_supervisor_v1",
                state="retrying" if attempt <= args.max_retries else "failed",
                attempt=attempt,
                elapsed_sec=time.monotonic() - started,
                error=str(exc),
                attempts=attempts,
            )
            if attempt > args.max_retries:
                raise


if __name__ == "__main__":
    main()
