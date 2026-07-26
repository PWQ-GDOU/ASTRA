"""Read-only audit and classical benchmark for the supplied COMSOL archive."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.data.reaction_wheel_sim import (
    archive_summary,
    load_simulation_archive,
    split_summary,
)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def bias(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def fit_ridge(x, y, alpha):
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1.0e-8] = 1.0
    z = (x - mean) / scale
    design = np.c_[np.ones(len(z)), z]
    ident = np.eye(design.shape[1])
    ident[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + alpha * ident, design.T @ y)
    return mean, scale, beta


def predict(x, mean, scale, beta):
    return np.c_[np.ones(len(x)), (x - mean) / scale] @ beta


def run_aligned(archive, tier, output):
    table = archive.aligned[tier]
    groups = sorted(set(table.group_ids.tolist()), key=lambda value: int(float(value)))
    split_by_group = {
        group: next(iter(table.split[table.group_ids == group]), "")
        for group in groups
    }
    rows = []
    for holdout in groups:
        train_mask = table.group_ids != holdout
        test_mask = table.group_ids == holdout
        x_train, y_train = table.features[train_mask], table.target[train_mask]
        x_test, y_test = table.features[test_mask], table.target[test_mask]
        best = None
        for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            fit_groups = sorted(set(table.group_ids[train_mask].tolist()), key=lambda value: int(float(value)))
            val_group = fit_groups[-1]
            fit_mask = train_mask & (table.group_ids != val_group)
            val_mask = table.group_ids == val_group
            mean, scale, beta = fit_ridge(table.features[fit_mask], table.target[fit_mask], alpha)
            val_pred = predict(table.features[val_mask], mean, scale, beta)
            score = rmse(table.target[val_mask], val_pred)
            if best is None or score < best[0]:
                best = (score, alpha)
        alpha = best[1]
        mean, scale, beta = fit_ridge(x_train, y_train, alpha)
        prediction = predict(x_test, mean, scale, beta)
        rows.append({
            "tier": tier,
            "holdout": holdout,
            "split": split_by_group[holdout],
            "alpha": alpha,
            "rmse": rmse(y_test, prediction),
            "mae": mae(y_test, prediction),
            "bias": bias(y_test, prediction),
            "n_test": int(test_mask.sum()),
        })
    macro = {
        "rmse": float(np.mean([row["rmse"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
        "bias": float(np.mean([row["bias"] for row in rows])),
    }
    return {"tier": tier, "folds": rows, "macro": macro}


def run_aligned_fixed_split(archive, tier):
    table = archive.aligned[tier]
    train_mask = np.isin(table.split, ["训练"])
    validation_mask = table.split == "验证"
    test_mask = table.split == "测试"
    if not np.any(train_mask) or not np.any(validation_mask) or not np.any(test_mask):
        raise ValueError(f"Aligned table {tier} does not contain train/validation/test trajectory splits")
    best = None
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        mean, scale, beta = fit_ridge(table.features[train_mask], table.target[train_mask], alpha)
        validation_pred = predict(table.features[validation_mask], mean, scale, beta)
        score = rmse(table.target[validation_mask], validation_pred)
        if best is None or score < best[0]:
            best = (score, alpha)
    validation_rmse, alpha = best
    mean, scale, beta = fit_ridge(table.features[train_mask], table.target[train_mask], alpha)
    test_pred = predict(table.features[test_mask], mean, scale, beta)
    return {
        "tier": tier,
        "protocol": "predeclared trajectory split: tracks 1-3 train, 4 validation, 5 test",
        "alpha": alpha,
        "validation_rmse": validation_rmse,
        "test_rmse": rmse(table.target[test_mask], test_pred),
        "test_mae": mae(table.target[test_mask], test_pred),
        "test_bias": bias(table.target[test_mask], test_pred),
        "train_groups": sorted(set(table.group_ids[train_mask].tolist())),
        "validation_groups": sorted(set(table.group_ids[validation_mask].tolist())),
        "test_groups": sorted(set(table.group_ids[test_mask].tolist())),
        "n_train": int(train_mask.sum()),
        "n_validation": int(validation_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


def run_low_error(archive, tier, output):
    train = archive.low_error[f"low_error_train_{tier}"]
    validation = archive.low_error[f"low_error_validation_{tier}"]
    test = archive.low_error[f"low_error_test_{tier}"]
    best = None
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        mean, scale, beta = fit_ridge(train.features, train.target, alpha)
        score = rmse(validation.target, predict(validation.features, mean, scale, beta))
        if best is None or score < best[0]:
            best = (score, alpha)
    val_score, alpha = best
    mean, scale, beta = fit_ridge(train.features, train.target, alpha)
    prediction = predict(test.features, mean, scale, beta)
    return {
        "tier": tier,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "alpha": alpha,
        "validation_rmse": val_score,
        "test_rmse": rmse(test.target, prediction),
        "test_mae": mae(test.target, prediction),
        "test_bias": bias(test.target, prediction),
        "train_groups": sorted(set(train.group_ids.tolist())),
        "validation_groups": sorted(set(validation.group_ids.tolist())),
        "test_groups": sorted(set(test.group_ids.tolist())),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", default="outputs/reaction_wheel_sim_audit")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    archive = load_simulation_archive(args.archive)
    result = {
        "schema": "reaction_wheel_sim_audit_v1",
        "archive": archive_summary(archive),
        "aligned_ridge_loo": [run_aligned(archive, tier, out) for tier in ("operational", "estimated", "oracle")],
        "aligned_ridge_fixed_split": [run_aligned_fixed_split(archive, tier) for tier in ("operational", "estimated", "oracle")],
        "low_error_ridge": [run_low_error(archive, tier, out) for tier in ("noisy", "observable", "oracle")],
        "leakage_warnings": [
            "Aligned oracle tier contains true degradation and precomputed particle-filter estimates; it is audit-only.",
            "Low-error oracle tier contains latent degradation parameters; it is audit-only.",
            "The supplied train/validation/test split is trajectory-disjoint (tracks 1-3/4/5).",
            "Metrics on the supplied simulation test split are simulation-domain results, not FEMTO or real reaction-wheel results.",
        ],
    }
    (out / "SIMULATION_AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "SIMULATION_RIDGE_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = []
        for item in result["aligned_ridge_loo"]:
            rows.extend({"dataset": "aligned_loo", **row} for row in item["folds"])
        rows.extend({"dataset": "aligned_fixed_split", **item} for item in result["aligned_ridge_fixed_split"])
        rows.extend({"dataset": "low_error", **item} for item in result["low_error_ridge"])
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result["aligned_ridge_fixed_split"], ensure_ascii=False, indent=2))
    print(json.dumps(result["low_error_ridge"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
