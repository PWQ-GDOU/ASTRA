"""Container-friendly, auditable reproduction for FEMTO/IMS -> COMSOL."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INPUTS = {
    "processed/femto_bearing.zip": "e21bb22bd8d54fd18ebe98b4b4e094c0c40469bda19811a2a642d5cc84ebd81f",
    "processed/ims_processed": "887eea9cdb541ff40de8164f1913f9a37d22b12e3c48cc0eaa2d6dc7c100f4e6",
    "raw/competition/reaction_wheel_comsol_degradation.zip": "b628deb89634cf8732cd96bb62d675b820aedb0aa337fd36082db53bfebca8fe",
}
LOCKED_PACKAGES = {
    "torch": "2.3.1+cpu",
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "scikit-learn": "1.5.1",
    "pandas": "2.2.2",
    "h5py": "3.11.0",
}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        files = sorted(
            (item for item in path.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(path).as_posix(),
        )
        for item in files:
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
        return digest.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _code_digest() -> str:
    selected = [
        PROJECT_ROOT / "Dockerfile",
        PROJECT_ROOT / "docker-compose.yml",
        PROJECT_ROOT / "docker" / "requirements.lock.txt",
    ]
    for folder in (PROJECT_ROOT / "scripts", PROJECT_ROOT / "src"):
        selected.extend(item for item in folder.rglob("*.py") if item.is_file())
    digest = hashlib.sha256()
    for item in sorted(selected, key=lambda candidate: candidate.relative_to(PROJECT_ROOT).as_posix()):
        digest.update(item.relative_to(PROJECT_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    found = {}
    errors = []
    for package, expected in LOCKED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        found[package] = actual
        if actual != expected:
            errors.append(f"{package}: expected {expected}, found {actual}")
    if errors:
        raise RuntimeError("Container dependency lock failed:\n- " + "\n- ".join(errors))
    return found


def validate_inputs(data_root: Path) -> dict[str, object]:
    records = {}
    errors = []
    for relative, expected_hash in EXPECTED_INPUTS.items():
        path = data_root / relative
        if not path.exists():
            errors.append(f"Missing required input: {path}")
            continue
        actual_hash = _sha256_path(path)
        records[relative] = {
            "path": str(path),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "sha256_matches": actual_hash == expected_hash,
        }
        if actual_hash != expected_hash:
            errors.append(f"SHA-256 mismatch: {path}")
    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))
    return {"data_root": str(data_root), "inputs": records, "all_hashes_match": True}


def _environment_record(data_audit: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "femto_ims_to_comsol_reproduction_v1",
        "created_unix_sec": time.time(),
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "packages": _package_versions(),
        "determinism_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "code_sha256": _code_digest(),
        "input_audit": data_audit,
        "data_mount_must_be_read_only": True,
    }


def _assert_fresh_result(path: Path) -> None:
    generated = ("REPORT.json", "SUPERVISOR_STATUS.json", "MACRO.csv", "PREDICTIONS.csv")
    existing = [name for name in generated if (path / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing result at {path}: {existing}")


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _validate_report(path: Path, expected_rows: int) -> dict[str, object]:
    report_path = path / "REPORT.json"
    if not report_path.is_file():
        raise RuntimeError(f"Expected report was not created: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("macro_rows", [])
    arms = set(report.get("protocol", {}).get("source_arms", []))
    if len(rows) != expected_rows or arms != {"femto", "ims"}:
        raise RuntimeError(f"Unexpected report structure: rows={len(rows)}, source_arms={sorted(arms)}")
    return {"report": str(report_path), "macro_rows": len(rows), "source_arms": sorted(arms)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke", "full", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reproducibility" / "femto_ims_to_comsol",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--heartbeat-sec", type=float, default=30.0)
    args = parser.parse_args()

    data_audit = validate_inputs(args.data_root.resolve())
    environment = _environment_record(data_audit)
    output_root = args.output_root.resolve()
    _write_json(output_root / "REPRODUCTION_ENVIRONMENT.json", environment)
    completed = {"preflight": "passed"}

    data_args = [
        "--source",
        "both",
        "--device",
        args.device,
        "--femto-zip",
        str(args.data_root / "processed" / "femto_bearing.zip"),
        "--ims-path",
        str(args.data_root / "processed" / "ims_processed"),
        "--comsol-archive",
        str(args.data_root / "raw" / "competition" / "reaction_wheel_comsol_degradation.zip"),
    ]

    if args.phase in {"smoke", "all"}:
        smoke_output = output_root / "smoke"
        _assert_fresh_result(smoke_output)
        _run([sys.executable, str(PROJECT_ROOT / "scripts" / "exp_femto_ims_to_comsol.py"), "--quick", *data_args, "--output", str(smoke_output)])
        completed["smoke"] = _validate_report(smoke_output, expected_rows=66)
        _write_json(smoke_output / "REPRODUCTION_ENVIRONMENT.json", environment)

    if args.phase in {"full", "all"}:
        full_output = output_root / "full"
        _assert_fresh_result(full_output)
        _run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "run_femto_ims_to_comsol_supervised.py"),
                "--output",
                str(full_output),
                "--max-retries",
                str(args.max_retries),
                "--timeout-hours",
                str(args.timeout_hours),
                "--heartbeat-sec",
                str(args.heartbeat_sec),
                "--",
                *data_args,
            ]
        )
        completed["full"] = _validate_report(full_output, expected_rows=120)
        _write_json(full_output / "REPRODUCTION_ENVIRONMENT.json", environment)

    result = {
        "schema": "femto_ims_to_comsol_reproduction_result_v1",
        "phase": args.phase,
        "output_root": str(output_root),
        "completed": completed,
    }
    _write_json(output_root / "REPRODUCTION_STATUS.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
