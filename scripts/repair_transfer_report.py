"""Repair deterministic audit labels in an already completed transfer report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ims_strict import sha256_file


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted((child for child in path.rglob("*") if child.is_file()), key=lambda child: str(child.relative_to(path))):
        digest.update(str(item.relative_to(path)).replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _ensemble_label(row: dict[str, str]) -> str:
    method = row["method"]
    while method.endswith("_ensemble"):
        method = method[: -len("_ensemble")]
    seed = row.get("seed", "")
    if seed not in ("", "None", "null"):
        return method
    if method == "ridge":
        return "ridge" if row.get("ridge_alpha", "") not in ("", "None", "null") else "ridge_ensemble"
    return f"{method}_ensemble"


def _write_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ims-path", type=Path, default=None)
    args = parser.parse_args()
    macro_path = args.output / "MACRO.csv"
    report_path = args.output / "REPORT.json"
    rows = []
    with macro_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["method"] = _ensemble_label(row)
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    temp_macro = macro_path.with_suffix(macro_path.suffix + ".partial")
    with temp_macro.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_macro, macro_path)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for row in report.get("macro_rows", []):
        row["method"] = _ensemble_label({key: "" if value is None else str(value) for key, value in row.items()})
    if args.ims_path is not None and args.ims_path.exists():
        digest = _path_sha256(args.ims_path)
        report["sources"]["ims"]["manifest"]["sha256"] = digest
        report["preflight"]["source_manifests"]["ims"]["sha256"] = digest
    report["audit_repairs"] = {
        "ensemble_method_labels_repaired": True,
        "source_directory_hash_repaired": bool(args.ims_path is not None and args.ims_path.exists()),
    }
    _write_json(report_path, report)
    manifest = {
        "schema": "femto_ims_to_comsol_data_manifest_v1",
        "inputs": {
            "comsol": {
                "path": report["preflight"]["comsol_archive"],
                "sha256": report["preflight"]["comsol_sha256"],
                "feature_tier": report["preflight"]["comsol_feature_tier"],
                "trajectory_groups": report["preflight"]["comsol_groups"],
            },
            "femto": report["sources"]["femto"]["manifest"],
            "ims": report["sources"]["ims"]["manifest"],
        },
        "target_split": {
            "fine_tune_by_n": report["protocol"]["target_shots"],
            "validation": report["preflight"]["comsol_validation_groups"],
            "test": report["preflight"]["comsol_test_groups"],
            "test_labels_used_for_selection": False,
        },
    }
    _write_json(args.output / "DATA_MANIFEST.json", manifest)
    counts = {}
    for row in rows:
        key = (row["source"], row["n_shots"], row["method"])
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"rows": len(rows), "method_counts": {"|".join(key): value for key, value in counts.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
