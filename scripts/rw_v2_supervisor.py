"""Supervise reaction-wheel v2 at one holdout/candidate cell per child process.

Each child owns one GPU and one logical experiment cell. The supervisor keeps
the authoritative checkpoint, emits a live status file, and kills/retries a
cell when its heartbeat or wall-clock budget expires.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import exp_rw_v2_full, exp_rw_v2_seq20
from scripts.rw_v2_common import (
    atomic_write_json,
    load_checkpoint,
    save_checkpoint,
    write_heartbeat,
)


@dataclass(frozen=True)
class RunConfig:
    mode: str
    experiment_schema: str
    script: Path
    checkpoint_name: str
    report_name: str
    macro_name: str
    seq_len: int
    candidates: tuple[Any, ...]
    holdouts: tuple[str, ...]


def config_for(mode: str) -> RunConfig:
    if mode == "full":
        exp = exp_rw_v2_full
        return RunConfig(
            mode=mode,
            experiment_schema="reaction_wheel_v2_full",
            script=ROOT / "scripts" / "exp_rw_v2_full.py",
            checkpoint_name="RW_V2_FULL_CHECKPOINT.json",
            report_name="RW_V2_FULL_REPORT.json",
            macro_name="rw_v2_full_macro.csv",
            seq_len=exp.SEQ_LEN,
            candidates=tuple(exp.CANDIDATES),
            holdouts=tuple(exp.BEARING_NAMES),
        )
    if mode == "seq20":
        exp = exp_rw_v2_seq20
        return RunConfig(
            mode=mode,
            experiment_schema="reaction_wheel_v2_seq20",
            script=ROOT / "scripts" / "exp_rw_v2_seq20.py",
            checkpoint_name="RW_V2_SEQ20_CHECKPOINT.json",
            report_name="RW_V2_SEQ20_REPORT.json",
            macro_name="rw_v2_seq20_macro.csv",
            seq_len=exp.SEQ_LEN,
            candidates=tuple(exp.CANDIDATES),
            holdouts=tuple(exp.BEARING_NAMES),
        )
    raise ValueError(f"Unknown mode: {mode}")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def count_results(results: dict[str, dict[str, dict[str, Any]]]) -> int:
    return sum(len(values) for values in results.values())


def missing_cells(
    results: dict[str, dict[str, dict[str, Any]]],
    config: RunConfig,
) -> list[tuple[str, str]]:
    return [
        (holdout, candidate.name)
        for holdout in config.holdouts
        for candidate in config.candidates
        if candidate.name not in results.get(holdout, {})
    ]


def tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])


def terminate_group(process: subprocess.Popen[Any], grace_seconds: float = 10.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def checkpoint_kwargs(
    config: RunConfig,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: tuple[int, ...],
    candidates: tuple[str, ...],
    holdouts: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "experiment_schema": config.experiment_schema,
        "seq_len": config.seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "seeds": seeds,
        "candidate_names": candidates,
        "holdout_names": holdouts,
    }


def load_results(
    path: Path,
    config: RunConfig,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: tuple[int, ...],
    candidates: tuple[str, ...],
    holdouts: tuple[str, ...],
) -> tuple[dict[str, dict[str, dict[str, Any]]], float]:
    if not path.exists():
        return {}, 1.0
    kwargs = checkpoint_kwargs(
        config,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seeds=seeds,
        candidates=candidates,
        holdouts=holdouts,
    )
    results = load_checkpoint(path, **kwargs)
    payload = read_json(path, {})
    return results, float(payload.get("rul_scale", 1.0))


def save_master(
    path: Path,
    config: RunConfig,
    results: dict[str, dict[str, dict[str, Any]]],
    rul_scale: float,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: tuple[int, ...],
) -> None:
    save_checkpoint(
        path,
        **checkpoint_kwargs(
            config,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            seeds=seeds,
            candidates=tuple(candidate.name for candidate in config.candidates),
            holdouts=config.holdouts,
        ),
        rul_scale=rul_scale,
        per_fold=results,
    )


def write_status(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(
        path,
        {
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **payload,
        },
    )


def acquire_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        old = read_json(path, {})
        old_pid = int(old.get("pid", 0))
        if old_pid:
            try:
                os.kill(old_pid, 0)
            except ProcessLookupError:
                path.unlink()
                return acquire_lock(path)
        raise RuntimeError(f"Supervisor lock already exists: {path} ({old})")
    os.write(fd, json.dumps({"pid": os.getpid(), "started_at": time.time()}).encode())
    os.close(fd)
    return os.getpid()


def merge_cell(
    master: dict[str, dict[str, dict[str, Any]]],
    cell: dict[str, dict[str, dict[str, Any]]],
    holdout: str,
) -> list[str]:
    merged: list[str] = []
    values = cell.get(holdout, {})
    target = master.setdefault(holdout, {})
    for name, result in values.items():
        if name == "ridge_full" or name not in target:
            target[name] = result
            merged.append(name)
    return merged


def run_cell(
    *,
    config: RunConfig,
    output_root: Path,
    device: str,
    data_zip: Path,
    holdout: str,
    candidate: str,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: tuple[int, ...],
    cell_timeout: float,
    stall_timeout: float,
    retries: int,
    status_path: Path,
    completed: int,
    total: int,
    failures: list[dict[str, Any]],
    seed_candidate_result: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, dict[str, Any]]] | None, float, bool]:
    cell_dir = output_root / "cells" / holdout / candidate
    cell_dir.mkdir(parents=True, exist_ok=True)
    child_checkpoint = cell_dir / config.checkpoint_name
    child_candidates = (candidate,)
    child_holdouts = (holdout,)
    child_kwargs = checkpoint_kwargs(
        config,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        seeds=seeds,
        candidates=child_candidates,
        holdouts=child_holdouts,
    )
    if seed_candidate_result is not None and not child_checkpoint.exists():
        save_checkpoint(
            child_checkpoint,
            **child_kwargs,
            rul_scale=1.0,
            per_fold={holdout: {candidate: seed_candidate_result}},
        )

    for attempt in range(1, retries + 2):
        heartbeat = cell_dir / "heartbeat.json"
        child_log = cell_dir / "child.log"
        write_heartbeat(
            heartbeat,
            phase="launching",
            mode=config.mode,
            holdout=holdout,
            candidate=candidate,
            attempt=attempt,
            device=device,
            completed=completed,
            total=total,
        )
        command = [
            sys.executable,
            str(config.script),
            "--device",
            device,
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--patience",
            str(patience),
            "--seeds",
            ",".join(str(seed) for seed in seeds),
            "--holdouts",
            holdout,
            "--candidates",
            candidate,
            "--data-zip",
            str(data_zip),
            "--output",
            str(cell_dir),
            "--heartbeat",
            str(heartbeat),
        ]
        started = time.monotonic()
        last_payload: dict[str, Any] = {}
        reason = None
        with child_log.open("a", encoding="utf-8") as log:
            log.write(f"\n=== attempt {attempt} started {time.time()} ===\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
            while process.poll() is None:
                now = time.monotonic()
                payload = read_json(heartbeat, {})
                if payload:
                    last_payload = payload
                heartbeat_age = (
                    time.time() - heartbeat.stat().st_mtime
                    if heartbeat.exists()
                    else now - started
                )
                if now - started > cell_timeout:
                    reason = f"cell_timeout_{cell_timeout:.0f}s"
                    terminate_group(process)
                    break
                if heartbeat_age > stall_timeout:
                    reason = f"heartbeat_stall_{stall_timeout:.0f}s"
                    terminate_group(process)
                    break
                write_status(
                    status_path,
                    {
                        "phase": "cell_running",
                        "mode": config.mode,
                        "device": device,
                        "current": {
                            "holdout": holdout,
                            "candidate": candidate,
                            "attempt": attempt,
                            "pid": process.pid,
                            "elapsed_sec": round(now - started, 1),
                            "heartbeat_age_sec": round(heartbeat_age, 1),
                            "heartbeat": last_payload,
                        },
                        "completed": completed,
                        "total": total,
                        "failures": failures,
                    },
                )
                time.sleep(5)
            return_code = process.wait()

        if return_code == 0 and reason is None:
            cell_results, rul_scale = load_results(
                child_checkpoint,
                config,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                seeds=seeds,
                candidates=child_candidates,
                holdouts=child_holdouts,
            )
            if candidate in cell_results.get(holdout, {}):
                return cell_results, rul_scale, True
            reason = "child_completed_without_candidate_result"
        if reason is None:
            reason = f"child_exit_{return_code}"
        failure = {
            "mode": config.mode,
            "holdout": holdout,
            "candidate": candidate,
            "attempt": attempt,
            "reason": reason,
            "return_code": return_code,
            "heartbeat": last_payload,
            "log_tail": tail(child_log),
            "at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        failures.append(failure)
        write_status(
            status_path,
            {
                "phase": "cell_failed",
                "current": failure,
                "completed": completed,
                "total": total,
                "failures": failures,
            },
        )
    return None, 1.0, False


def finalize(
    output_root: Path,
    config: RunConfig,
    results: dict[str, dict[str, dict[str, Any]]],
    rul_scale: float,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    seeds: tuple[int, ...],
    started_at: float,
) -> bool:
    candidate_names = [candidate.name for candidate in config.candidates]
    names = candidate_names + ["ridge_full"]
    missing = [
        f"{holdout}:{name}"
        for holdout in config.holdouts
        for name in names
        if name not in results.get(holdout, {})
    ]
    if missing:
        return False
    macro = {
        name: float(sum(results[holdout][name]["nrmse"] for holdout in config.holdouts) / len(config.holdouts))
        for name in names
    }
    best = min(macro, key=macro.get)
    report = {
        "schema": config.experiment_schema,
        "seq_len": config.seq_len,
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "seeds": list(seeds),
        "holdouts": list(config.holdouts),
        "rul_scale": rul_scale,
        "candidates": [asdict(candidate) for candidate in config.candidates],
        "macro_nrmse": macro,
        "best_candidate": best,
        "elapsed_min": round((time.time() - started_at) / 60.0, 1),
        "per_fold": results,
    }
    atomic_write_json(output_root / config.report_name, report)
    with (output_root / config.macro_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate", "macro_nrmse"])
        for name, value in sorted(macro.items(), key=lambda item: item[1]):
            writer.writerow([name, f"{value:.4f}"])
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "seq20"), required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--source-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-zip", default="data/processed/femto_bearing.zip")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seeds", default="42,123,456,2026,3407")
    parser.add_argument("--cell-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--stall-timeout-sec", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-cells", type=int, default=None)
    args = parser.parse_args()

    config = config_for(args.mode)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / "SUPERVISOR.lock"
    acquire_lock(lock_path)
    status_path = output_root / "SUPERVISOR_STATUS.json"
    supervisor_log = output_root / "SUPERVISOR.log"
    started_at = time.time()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    all_candidate_names = tuple(candidate.name for candidate in config.candidates)
    source_checkpoint = Path(args.source_output) / config.checkpoint_name
    master_checkpoint = output_root / config.checkpoint_name
    failures: list[dict[str, Any]] = []
    try:
        if master_checkpoint.exists():
            results, rul_scale = load_results(
                master_checkpoint,
                config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                seeds=seeds,
                candidates=all_candidate_names,
                holdouts=config.holdouts,
            )
        else:
            results, rul_scale = load_results(
                source_checkpoint,
                config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                seeds=seeds,
                candidates=all_candidate_names,
                holdouts=config.holdouts,
            )
            save_master(
                master_checkpoint,
                config,
                results,
                rul_scale,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                seeds=seeds,
            )

        total = len(config.holdouts) * (len(config.candidates) + 1)
        completed = count_results(results)
        with supervisor_log.open("a", encoding="utf-8") as log:
            log.write(
                f"start mode={config.mode} device={args.device} "
                f"completed={completed}/{total} at={time.time()}\n"
            )
        write_status(
            status_path,
            {
                "phase": "starting",
                "mode": config.mode,
                "device": args.device,
                "completed": completed,
                "total": total,
                "source_checkpoint": str(source_checkpoint),
                "master_checkpoint": str(master_checkpoint),
                "cell_timeout_sec": args.cell_timeout_sec,
                "stall_timeout_sec": args.stall_timeout_sec,
                "retries": args.retries,
                "failures": failures,
            },
        )

        tasks = missing_cells(results, config)
        for holdout in config.holdouts:
            fold = results.get(holdout, {})
            if "ridge_full" not in fold and all(
                candidate.name in fold for candidate in config.candidates
            ):
                tasks.append((holdout, config.candidates[0].name))
        cell_count = 0
        for holdout, candidate in tasks:
            if args.max_cells is not None and cell_count >= args.max_cells:
                break
            cell_results, cell_rul_scale, ok = run_cell(
                config=config,
                output_root=output_root,
                device=args.device,
                data_zip=Path(args.data_zip),
                holdout=holdout,
                candidate=candidate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                patience=args.patience,
                seeds=seeds,
                cell_timeout=args.cell_timeout_sec,
                stall_timeout=args.stall_timeout_sec,
                retries=args.retries,
                status_path=status_path,
                completed=completed,
                total=total,
                failures=failures,
                seed_candidate_result=(
                    results.get(holdout, {}).get(candidate)
                    if candidate in {item.name for item in config.candidates}
                    and candidate in results.get(holdout, {})
                    else None
                ),
            )
            cell_count += 1
            if ok and cell_results is not None:
                merged = merge_cell(results, cell_results, holdout)
                rul_scale = cell_rul_scale or rul_scale
                save_master(
                    master_checkpoint,
                    config,
                    results,
                    rul_scale,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    seeds=seeds,
                )
                completed = count_results(results)
                with supervisor_log.open("a", encoding="utf-8") as log:
                    log.write(
                        f"complete holdout={holdout} candidate={candidate} "
                        f"merged={merged} progress={completed}/{total}\n"
                    )

        complete = finalize(
            output_root,
            config,
            results,
            rul_scale,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            seeds=seeds,
            started_at=started_at,
        )
        write_status(
            status_path,
            {
                "phase": "complete" if complete else "incomplete",
                "mode": config.mode,
                "device": args.device,
                "completed": count_results(results),
                "total": total,
                "missing": [
                    f"{holdout}:{name}"
                    for holdout in config.holdouts
                    for name in [candidate.name for candidate in config.candidates] + ["ridge_full"]
                    if name not in results.get(holdout, {})
                ],
                "failures": failures,
                "report": str(output_root / config.report_name) if complete else None,
            },
        )
        return 0 if complete else 2
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
