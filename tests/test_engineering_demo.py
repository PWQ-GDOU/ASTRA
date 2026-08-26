from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_engineering_demo.py"
SPEC = importlib.util.spec_from_file_location("build_engineering_demo", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _acceptance_row(n_shots: int | str) -> dict[str, object]:
    return {
        "source": "femto",
        "n_shots": n_shots,
        "method": "trend_residual_transfer_calibrated_ensemble",
        "outer_folds": 15 if n_shots == "all_predeclared" else 5,
        "macro_raw_rmse": 5.0,
        "macro_mae": 4.0,
        "macro_bias": 0.5,
        "macro_normalized_rmse": 0.02,
        "scratch_macro_raw_rmse": 6.0,
        "ridge_macro_raw_rmse": 6.2,
        "best_deterministic_trend_macro_raw_rmse": 6.4,
        "improvement_vs_scratch_pct": 16.7,
        "outer_fold_wins_vs_scratch": 4,
        "strict_positive_transfer_evidence": True,
        "acceptance_reason": "strict positive evidence",
    }


def _make_experiment(path: Path) -> Path:
    path.mkdir()
    (path / "ACCEPTANCE.json").write_text(
        json.dumps({"results": [_acceptance_row(2), _acceptance_row(3), _acceptance_row("all_predeclared")]}),
        encoding="utf-8",
    )
    (path / "PREFLIGHT.json").write_text(
        json.dumps(
            {
                "inputs_exist": True,
                "target_scaler_fit_only_on_fit_groups": True,
                "target_prior_fit_only_on_fit_groups": True,
                "full_life_min_max_used_for_target_input": False,
                "holdout_labels_used_for_selection": False,
                "holdout_labels_used_for_fit": False,
                "finite_checks": True,
                "transfer_checkpoint_requires_unfrozen_encoder": True,
                "outer_holdouts": ["1", "2", "3", "4", "5"],
                "target_model_shape_check": {"target_features": 13, "sequence_length": 20},
            }
        ),
        encoding="utf-8",
    )
    (path / "PROTOCOL.json").write_text(
        json.dumps({"experiment_designation": "exploratory_post_hoc_replication"}), encoding="utf-8"
    )
    (path / "DATA_MANIFEST.json").write_text(json.dumps({"inputs": {"comsol": {"sha256": "abc"}}}), encoding="utf-8")
    (path / "SUPERVISOR_STATUS.json").write_text(
        json.dumps({"state": "completed", "rows": 30, "elapsed_sec": 1.2}), encoding="utf-8"
    )
    fields = ["source", "n_shots", "outer_holdout", "method", "target_group", "endpoint", "true_rul", "prediction"]
    with (path / "PREDICTIONS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for shots in (2, 3):
            for holdout in range(1, 6):
                for endpoint in range(3):
                    writer.writerow(
                        {
                            "source": "femto",
                            "n_shots": shots,
                            "outer_holdout": holdout,
                            "method": "trend_residual_transfer_calibrated_ensemble",
                            "target_group": holdout,
                            "endpoint": endpoint,
                            "true_rul": 100 - endpoint,
                            "prediction": 99 - endpoint,
                        }
                    )
    return path


def test_dashboard_builds_operator_and_audit_replay(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path / "v5")
    output = tmp_path / "dashboard"
    report = MODULE.build_dashboard(experiment, output)

    assert report["status"] == "passed"
    assert (output / "index.html").is_file()
    assert (output / "DASHBOARD_DATA.json").is_file()
    assert (output / "ENGINEERING_DEMO_REPORT.json").is_file()
    data = json.loads((output / "DASHBOARD_DATA.json").read_text(encoding="utf-8"))
    html = (output / "index.html").read_text(encoding="utf-8")
    assert set(data["series"]) == {"2", "3"}
    assert set(data["series"]["2"]) == {"1", "2", "3", "4", "5"}
    assert "audit_reference_rul" in data["series"]["2"]["1"][0]
    assert "true_rul" not in data["series"]["2"]["1"][0]
    assert "Audit replay" in html
    assert "Reference RUL is hidden in normal replay" in html
    assert report["validation"]["operator_audit_separation"] is True


def test_dashboard_rejects_full_life_input_leakage(tmp_path: Path) -> None:
    experiment = _make_experiment(tmp_path / "v5")
    preflight_path = experiment / "PREFLIGHT.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["full_life_min_max_used_for_target_input"] = True
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    with pytest.raises(RuntimeError, match="full_life_min_max_used_for_target_input"):
        MODULE.build_dashboard(experiment, tmp_path / "dashboard")
