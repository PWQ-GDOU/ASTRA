"""Build an offline engineering lifetime-state dashboard from locked v5 evidence.

The generated HTML separates operator replay (predictions only) from audit replay
(held-out reference RUL revealed explicitly). It is a static artifact and does
not make a live-flight-telemetry claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "femto_ims_to_comsol_outerloo_v5_exploratory_fitonly_innerloo_full5x120_20260825"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "engineering_demo" / "femto_ims_to_comsol_v5"
REQUIRED_AUDIT_FILES = (
    "ACCEPTANCE.json",
    "DATA_MANIFEST.json",
    "PREFLIGHT.json",
    "PROTOCOL.json",
    "PREDICTIONS.csv",
    "SUPERVISOR_STATUS.json",
)
DISPLAY_METHODS = {
    2: "trend_residual_transfer_calibrated_ensemble",
    3: "trend_residual_transfer_calibrated_ensemble",
}
AGGREGATE_METHOD = "trend_residual_transfer_calibrated_ensemble"
OUTER_HOLDOUTS = ("1", "2", "3", "4", "5")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Cannot read JSON audit artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Expected finite numeric {label}, found {value!r}") from error
    if not math.isfinite(number):
        raise RuntimeError(f"Expected finite numeric {label}, found {value!r}")
    return number


def _required_file_hashes(experiment_output: Path) -> dict[str, str]:
    missing = [name for name in REQUIRED_AUDIT_FILES if not (experiment_output / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "The engineering demo requires a completed v5 output. Missing: " + ", ".join(missing)
        )
    return {name: _sha256(experiment_output / name) for name in REQUIRED_AUDIT_FILES}


def _find_result(
    acceptance: dict[str, Any], *, n_shots: int | str, method: str, label: str
) -> dict[str, Any]:
    rows = acceptance.get("results")
    if not isinstance(rows, list):
        raise RuntimeError("ACCEPTANCE.json has no result rows")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("source") == "femto"
        and row.get("n_shots") == n_shots
        and row.get("method") == method
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one acceptance row for {label}; found {len(matches)}")
    row = matches[0]
    if row.get("strict_positive_transfer_evidence") is not True:
        raise RuntimeError(f"{label} is not strict positive transfer evidence")
    for key in (
        "macro_raw_rmse",
        "macro_mae",
        "macro_bias",
        "macro_normalized_rmse",
        "scratch_macro_raw_rmse",
        "ridge_macro_raw_rmse",
        "best_deterministic_trend_macro_raw_rmse",
        "improvement_vs_scratch_pct",
    ):
        _finite(row.get(key), f"{label}.{key}")
    return row


def _validate_preflight(preflight: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    expected_values = {
        "inputs_exist": True,
        "target_scaler_fit_only_on_fit_groups": True,
        "target_prior_fit_only_on_fit_groups": True,
        "full_life_min_max_used_for_target_input": False,
        "holdout_labels_used_for_selection": False,
        "holdout_labels_used_for_fit": False,
        "finite_checks": True,
        "transfer_checkpoint_requires_unfrozen_encoder": True,
    }
    for key, expected in expected_values.items():
        if preflight.get(key) is not expected:
            raise RuntimeError(f"Preflight guard failed: {key} must be {expected}")
    if preflight.get("outer_holdouts") != list(OUTER_HOLDOUTS):
        raise RuntimeError("Preflight outer holdouts do not match the five trajectory protocol")
    shape = preflight.get("target_model_shape_check")
    if not isinstance(shape, dict) or shape.get("target_features") != 13 or shape.get("sequence_length") != 20:
        raise RuntimeError("Preflight target model shape must be 13 features and sequence length 20")
    if protocol.get("experiment_designation") != "exploratory_post_hoc_replication":
        raise RuntimeError("Dashboard only supports the locked exploratory v5 designation")
    return {
        "inputs_exist": True,
        "scaler_fit_on_fit_groups_only": True,
        "prior_fit_on_fit_groups_only": True,
        "holdout_labels_used_for_selection": False,
        "holdout_labels_used_for_fit": False,
        "full_life_min_max_used_for_target_input": False,
        "target_features": 13,
        "sequence_length": 20,
        "encoder_unfrozen_checkpoint_required": True,
    }


def _load_display_series(predictions_path: Path) -> dict[str, dict[str, list[dict[str, float]]]]:
    output = {str(shots): {holdout: [] for holdout in OUTER_HOLDOUTS} for shots in DISPLAY_METHODS}
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {
            "source", "n_shots", "outer_holdout", "method", "endpoint", "true_rul", "prediction"
        }
        if reader.fieldnames is None or not expected_columns.issubset(reader.fieldnames):
            raise RuntimeError("PREDICTIONS.csv does not provide the required v5 columns")
        for row in reader:
            if row.get("source") != "femto":
                continue
            shots = row.get("n_shots")
            if shots not in output or row.get("method") != DISPLAY_METHODS[int(shots)]:
                continue
            holdout = row.get("outer_holdout")
            if holdout not in output[shots]:
                continue
            endpoint = int(_finite(row.get("endpoint"), "prediction.endpoint"))
            output[shots][holdout].append(
                {
                    "endpoint": endpoint,
                    "prediction": _finite(row.get("prediction"), "prediction.prediction"),
                    "audit_reference_rul": _finite(row.get("true_rul"), "prediction.true_rul"),
                }
            )
    for shots, per_holdout in output.items():
        for holdout, series in per_holdout.items():
            series.sort(key=lambda point: point["endpoint"])
            endpoints = [point["endpoint"] for point in series]
            if len(series) < 2 or len(set(endpoints)) != len(endpoints):
                raise RuntimeError(f"Incomplete or duplicated display series: N={shots}, trajectory={holdout}")
            if any(right <= left for left, right in zip(endpoints, endpoints[1:])):
                raise RuntimeError(f"Non-chronological display series: N={shots}, trajectory={holdout}")
    return output


def _result_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "transfer_rmse": _finite(row["macro_raw_rmse"], "macro_raw_rmse"),
        "scratch_rmse": _finite(row["scratch_macro_raw_rmse"], "scratch_macro_raw_rmse"),
        "ridge_rmse": _finite(row["ridge_macro_raw_rmse"], "ridge_macro_raw_rmse"),
        "best_trend_rmse": _finite(
            row["best_deterministic_trend_macro_raw_rmse"], "best_deterministic_trend_macro_raw_rmse"
        ),
        "mae": _finite(row["macro_mae"], "macro_mae"),
        "bias": _finite(row["macro_bias"], "macro_bias"),
        "normalized_rmse": _finite(row["macro_normalized_rmse"], "macro_normalized_rmse"),
        "improvement_vs_scratch_pct": _finite(
            row["improvement_vs_scratch_pct"], "improvement_vs_scratch_pct"
        ),
        "outer_fold_wins": int(row["outer_fold_wins_vs_scratch"]),
        "outer_folds": int(row["outer_folds"]),
        "acceptance_reason": str(row["acceptance_reason"]),
    }


def build_dashboard(experiment_output: Path, output: Path) -> dict[str, Any]:
    experiment_output = experiment_output.resolve()
    output = output.resolve()
    hashes = _required_file_hashes(experiment_output)
    acceptance = _read_json(experiment_output / "ACCEPTANCE.json")
    manifest = _read_json(experiment_output / "DATA_MANIFEST.json")
    preflight = _read_json(experiment_output / "PREFLIGHT.json")
    protocol = _read_json(experiment_output / "PROTOCOL.json")
    supervisor = _read_json(experiment_output / "SUPERVISOR_STATUS.json")
    guards = _validate_preflight(preflight, protocol)
    metrics = {
        "2": _result_summary(
            _find_result(acceptance, n_shots=2, method=DISPLAY_METHODS[2], label="N=2 calibrated transfer")
        ),
        "3": _result_summary(
            _find_result(acceptance, n_shots=3, method=DISPLAY_METHODS[3], label="N=3 calibrated transfer")
        ),
        "aggregate": _result_summary(
            _find_result(
                acceptance,
                n_shots="all_predeclared",
                method=AGGREGATE_METHOD,
                label="predeclared aggregate calibrated transfer",
            )
        ),
    }
    series = _load_display_series(experiment_output / "PREDICTIONS.csv")
    data = {
        "schema": "engineering_lifetime_demo_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "title": "ASTRA Reaction-Wheel Lifetime Status",
        "designation": "exploratory_post_hoc_replication",
        "submission_eligibility": "requires_unseen_outer_holdout_confirmation",
        "boundary": "COMSOL reaction-wheel simulation proxy; not flight telemetry or a flight-release decision.",
        "operator_mode": "Predictions are shown without audit reference labels.",
        "audit_mode": "Audit replay reveals held-out reference RUL solely for validation review.",
        "policy": {
            "label": "Demonstration policy only; mission-specific thresholds require approval.",
            "critical_cycles": 30,
            "warning_cycles": 75,
            "watch_cycles": 150,
        },
        "metrics": metrics,
        "preflight": guards,
        "supervisor": {
            "state": supervisor.get("state"),
            "rows": supervisor.get("rows"),
            "elapsed_sec": supervisor.get("elapsed_sec"),
        },
        "source_manifest": manifest.get("inputs", {}),
        "audit_file_sha256": hashes,
        "series": series,
    }
    report = {
        "schema": "engineering_lifetime_demo_report_v1",
        "status": "passed",
        "generated_at_utc": data["generated_at_utc"],
        "experiment_output": str(experiment_output),
        "output": str(output),
        "validation": {
            "required_audit_files": list(REQUIRED_AUDIT_FILES),
            "strict_positive_transfer_evidence": {"N=2": True, "N=3": True, "aggregate": True},
            "outer_trajectory_count": 5,
            "preflight_guards": guards,
            "operator_audit_separation": True,
        },
        "audit_file_sha256": hashes,
    }
    _atomic_write_json(output / "DASHBOARD_DATA.json", data)
    _atomic_write_json(output / "ENGINEERING_DEMO_REPORT.json", report)
    _atomic_write_text(output / "index.html", _render_html(data))
    return report


def _render_html(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    document = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ASTRA | Reaction-Wheel Lifetime Status</title>
<style>
:root { --ink:#10242d; --muted:#61737a; --paper:#f4f6f4; --panel:#ffffff; --line:#cfdbd8; --teal:#006d70; --teal-dark:#004d51; --amber:#9c6200; --amber-bg:#fff2cf; --red:#b42318; --green:#197a4b; --navy:#263f57; }
* { box-sizing:border-box; }
body { margin:0; background:var(--paper); color:var(--ink); font:14px/1.45 Arial, Helvetica, sans-serif; }
button, select, input { font:inherit; }
button { cursor:pointer; }
.shell { max-width:1440px; margin:auto; padding:20px clamp(16px,3vw,40px) 48px; }
.topbar { display:flex; justify-content:space-between; align-items:flex-start; gap:20px; border-bottom:2px solid var(--ink); padding-bottom:18px; }
.brand { display:flex; gap:14px; align-items:flex-start; }
.mark { background:var(--teal-dark); color:#fff; font-weight:700; letter-spacing:1.4px; padding:7px 9px; border-radius:4px; }
h1 { font-size:24px; line-height:1.1; margin:2px 0 5px; letter-spacing:0; }
.subtitle { color:var(--muted); margin:0; max-width:770px; }
.run-meta { text-align:right; color:var(--muted); white-space:nowrap; }
.run-meta b { color:var(--ink); }
.boundary { background:var(--amber-bg); border-left:5px solid var(--amber); color:#513900; padding:12px 15px; margin:18px 0; border-radius:4px; }
.metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }
.metric { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:13px; min-height:104px; }
.metric .label { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.6px; }
.metric .value { font-size:25px; font-weight:700; margin:6px 0 2px; }
.metric .detail { color:var(--muted); font-size:12px; }
.good { color:var(--green); }
.flow { margin-top:20px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:14px 0; }
.flow h2, .panel h2 { font-size:14px; margin:0 0 10px; letter-spacing:.35px; }
.flow-track { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; }
.flow-step { min-height:62px; padding:10px; border-left:4px solid var(--teal); background:#e9f2f1; }
.flow-step b { display:block; font-size:12px; margin-bottom:3px; }
.flow-step span { color:var(--muted); font-size:12px; }
.workspace { display:grid; grid-template-columns:minmax(0,1.7fr) minmax(300px,.8fr); gap:16px; margin-top:20px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:16px; }
.controls { display:grid; grid-template-columns:auto minmax(150px,1fr) auto auto; align-items:center; gap:10px; margin-bottom:14px; }
.segmented { display:flex; border:1px solid var(--line); border-radius:5px; overflow:hidden; width:max-content; }
.segmented button { border:0; background:#fff; color:var(--muted); min-width:54px; padding:7px 10px; }
.segmented button + button { border-left:1px solid var(--line); }
.segmented button[aria-pressed="true"] { background:var(--teal-dark); color:#fff; }
select, .range { width:100%; min-height:34px; }
.icon-command { border:1px solid var(--teal-dark); background:#fff; color:var(--teal-dark); padding:7px 10px; border-radius:4px; font-weight:700; }
.icon-command:hover { background:#e9f2f1; }
.audit-label { display:flex; align-items:center; gap:7px; color:var(--ink); font-size:12px; white-space:nowrap; }
.audit-label input { width:16px; height:16px; accent-color:var(--amber); }
.timeline { display:grid; grid-template-columns:76px 1fr 84px; align-items:center; gap:10px; margin:8px 0 12px; color:var(--muted); font-size:12px; }
.timeline output { text-align:right; color:var(--ink); font-weight:700; }
.chart-wrap { border:1px solid var(--line); background:#fbfcfb; height:356px; position:relative; }
.chart { width:100%; height:100%; display:block; }
.legend { display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); font-size:12px; margin-top:10px; }
.legend i { display:inline-block; width:20px; border-top:3px solid var(--teal); vertical-align:middle; margin-right:5px; }
.legend i.audit { border-color:var(--amber); }
.status-grid { display:grid; gap:12px; }
.status { border-left:4px solid var(--teal); padding:8px 10px; background:#eef6f5; }
.status.critical { border-color:var(--red); background:#fff0ef; }
.status.warning { border-color:var(--amber); background:#fff8e7; }
.status.watch { border-color:#537c94; background:#eef5f8; }
.status .name { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.55px; }
.status .reading { font-size:24px; font-weight:700; margin:3px 0; }
.status .action { font-size:13px; }
.audit-only { display:none; }
.audit-active .audit-only { display:block; }
.comparison { margin-top:16px; overflow:auto; }
table { width:100%; border-collapse:collapse; min-width:620px; }
th, td { text-align:left; border-bottom:1px solid var(--line); padding:9px 7px; font-variant-numeric:tabular-nums; }
th { color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.4px; }
.audit-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; }
.audit-item { border:1px solid var(--line); padding:10px; min-height:65px; }
.audit-item b { display:block; font-size:12px; margin-bottom:3px; }
.audit-item span { color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
.hashes { margin:10px 0 0; padding:0; list-style:none; color:var(--muted); font:11px/1.45 Consolas, monospace; }
.footer { color:var(--muted); font-size:12px; margin-top:18px; }
@media (max-width:900px) { .metrics { grid-template-columns:repeat(2,minmax(0,1fr)); } .workspace { grid-template-columns:1fr; } .topbar { display:block; } .run-meta { text-align:left; margin-top:12px; white-space:normal; } }
@media (max-width:620px) { .shell { padding:14px 12px 30px; } h1 { font-size:21px; } .metrics { grid-template-columns:1fr; } .flow-track { grid-template-columns:1fr; } .flow-step { min-height:auto; } .controls { grid-template-columns:1fr 1fr; } .audit-label { grid-column:1 / -1; } .timeline { grid-template-columns:1fr; gap:4px; } .timeline output { text-align:left; } .chart-wrap { height:300px; } .audit-grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main class="shell" id="app">
  <header class="topbar">
    <div class="brand"><div class="mark">ASTRA</div><div><h1>Reaction-Wheel Lifetime Status</h1><p class="subtitle">FEMTO to COMSOL transfer replay. Engineering status is based on the deployed prediction trace, with audit labels intentionally withheld in operator mode.</p></div></div>
    <div class="run-meta">Protocol: <b>v5 outer LOO</b><br>Target: <b>COMSOL RW proxy</b><br>Run state: <b id="run-state"></b></div>
  </header>
  <section class="boundary"><b>Evidence boundary.</b> Exploratory post-hoc replication only. An unseen outer-holdout confirmation is still required. This dashboard is not flight telemetry and does not authorize a flight-release decision.</section>
  <section class="metrics" aria-label="Validated transfer metrics">
    <article class="metric"><div class="label">N=2 Transfer RMSE</div><div class="value" id="metric-n2"></div><div class="detail" id="metric-n2-detail"></div></article>
    <article class="metric"><div class="label">N=3 Transfer RMSE</div><div class="value" id="metric-n3"></div><div class="detail" id="metric-n3-detail"></div></article>
    <article class="metric"><div class="label">Aggregate Gain vs Scratch</div><div class="value good" id="metric-agg"></div><div class="detail" id="metric-agg-detail"></div></article>
    <article class="metric"><div class="label">Data Preflight</div><div class="value good">PASSED</div><div class="detail">Train-only scale / no holdout labels</div></article>
  </section>
  <section class="flow" aria-label="End-to-end run flow"><h2>Engineering Run Flow</h2><div class="flow-track">
    <div class="flow-step"><b>1. Data preflight</b><span>Hash, split, and leakage guards</span></div>
    <div class="flow-step"><b>2. Source pretraining</b><span>FEMTO degradation representation</span></div>
    <div class="flow-step"><b>3. Target adaptation</b><span>Fit-group-only calibration and validation</span></div>
    <div class="flow-step"><b>4. Online estimate</b><span>Replay endpoint into predicted RUL</span></div>
    <div class="flow-step"><b>5. Risk disposition</b><span>Demonstration policy, mission approval needed</span></div>
  </div></section>
  <section class="workspace">
    <article class="panel"><h2>Lifetime Replay</h2>
      <div class="controls">
        <div class="segmented" aria-label="Target fit group count"><button type="button" data-shots="2" aria-pressed="true">N=2</button><button type="button" data-shots="3" aria-pressed="false">N=3</button></div>
        <select id="trajectory" aria-label="Outer held-out trajectory"></select>
        <button class="icon-command" id="play" type="button" aria-label="Play lifetime replay">Play</button>
        <label class="audit-label"><input id="audit" type="checkbox"> Audit replay</label>
      </div>
      <div class="timeline"><span>Telemetry endpoint</span><input class="range" id="endpoint" type="range" min="0" value="0"><output id="endpoint-label"></output></div>
      <div class="chart-wrap"><svg class="chart" id="chart" role="img" aria-label="Predicted remaining useful life across replay endpoints"></svg></div>
      <div class="legend"><span><i></i>Predicted RUL</span><span class="audit-only"><i class="audit"></i>Audit reference RUL</span><span class="audit-only">Audit labels are excluded from operator status.</span></div>
    </article>
    <aside class="panel"><h2>Current Engineering Status</h2><div class="status-grid">
      <div class="status" id="risk-box"><div class="name">Risk disposition</div><div class="reading" id="risk-state"></div><div class="action" id="risk-action"></div></div>
      <div class="status"><div class="name">Predicted RUL</div><div class="reading" id="predicted-rul"></div><div class="action">Current model estimate in cycles</div></div>
      <div class="status"><div class="name">Observed life consumed</div><div class="reading" id="life-consumed"></div><div class="action">Endpoint progress only; not based on withheld RUL</div></div>
      <div class="status audit-only"><div class="name">Audit absolute error</div><div class="reading" id="audit-error"></div><div class="action">Visible only in audit replay</div></div>
    </div></aside>
  </section>
  <section class="panel comparison"><h2>Validated Comparison</h2><table><thead><tr><th>Protocol</th><th>Transfer RMSE</th><th>Scratch</th><th>Ridge</th><th>Best trend</th><th>Gain vs scratch</th><th>Outer wins</th></tr></thead><tbody id="comparison-body"></tbody></table></section>
  <section class="panel" style="margin-top:16px"><h2>Audit Trail</h2><div class="audit-grid" id="audit-grid"></div><ul class="hashes" id="hashes"></ul></section>
  <p class="footer">Risk thresholds are a demonstration policy: critical <= 30 cycles, warning <= 75, watch <= 150. Mission-specific limits, telemetry quality gates, and operational actions require formal approval.</p>
</main>
<script>
const data = __DASHBOARD_DATA__;
const state = { shots: '2', trajectory: '1', index: 0, audit: false, timer: null };
const $ = (id) => document.getElementById(id);
const fmt = (v, digits=1) => Number(v).toFixed(digits);
const series = () => data.series[state.shots][state.trajectory];
function metricCards() {
  const n2 = data.metrics['2'], n3 = data.metrics['3'], agg = data.metrics.aggregate;
  $('metric-n2').textContent = fmt(n2.transfer_rmse, 3) + ' cycles';
  $('metric-n2-detail').textContent = '+' + fmt(n2.improvement_vs_scratch_pct, 2) + '% vs scratch; ' + n2.outer_fold_wins + '/' + n2.outer_folds + ' folds';
  $('metric-n3').textContent = fmt(n3.transfer_rmse, 3) + ' cycles';
  $('metric-n3-detail').textContent = '+' + fmt(n3.improvement_vs_scratch_pct, 2) + '% vs scratch; ' + n3.outer_fold_wins + '/' + n3.outer_folds + ' folds';
  $('metric-agg').textContent = '+' + fmt(agg.improvement_vs_scratch_pct, 2) + '%';
  $('metric-agg-detail').textContent = fmt(agg.transfer_rmse, 3) + ' cycles RMSE; ' + agg.outer_fold_wins + '/' + agg.outer_folds + ' outer folds';
  $('run-state').textContent = String(data.supervisor.state || 'audited').toUpperCase();
}
function setupSelect() {
  const select = $('trajectory');
  select.innerHTML = '';
  Object.keys(data.series[state.shots]).forEach((key) => { const option = document.createElement('option'); option.value = key; option.textContent = 'Outer trajectory ' + key; select.append(option); });
  select.value = state.trajectory;
}
function risk(point) {
  const p = point.prediction, policy = data.policy;
  if (p <= policy.critical_cycles) return ['CRITICAL', 'Hold and inspect before next mission action.', 'critical'];
  if (p <= policy.warning_cycles) return ['WARNING', 'Schedule condition review and mitigation planning.', 'warning'];
  if (p <= policy.watch_cycles) return ['WATCH', 'Increase monitoring cadence; retain trending review.', 'watch'];
  return ['NOMINAL', 'Continue approved monitoring cadence.', 'nominal'];
}
function renderStatus() {
  const points = series(), point = points[state.index];
  const disposition = risk(point), consumed = point.endpoint / points[points.length - 1].endpoint * 100;
  $('predicted-rul').textContent = fmt(point.prediction, 1) + ' cycles';
  $('life-consumed').textContent = fmt(consumed, 1) + '%';
  $('risk-state').textContent = disposition[0]; $('risk-action').textContent = disposition[1];
  $('risk-box').className = 'status ' + disposition[2];
  $('audit-error').textContent = fmt(Math.abs(point.prediction - point.audit_reference_rul), 2) + ' cycles';
  $('endpoint-label').textContent = 'Cycle ' + point.endpoint + ' / ' + points[points.length - 1].endpoint;
}
function pathFor(points, key, width, height, pad, min, max) {
  const span = Math.max(max - min, 1), last = points.length - 1;
  return points.map((p, index) => { const x = pad + (width - 2 * pad) * index / last; const y = height - pad - (height - 2 * pad) * (p[key] - min) / span; return (index ? 'L' : 'M') + x.toFixed(2) + ' ' + y.toFixed(2); }).join(' ');
}
function svgNode(name, attrs) { const node = document.createElementNS('http://www.w3.org/2000/svg', name); Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value))); return node; }
function renderChart() {
  const svg = $('chart'), box = svg.getBoundingClientRect(), width = Math.max(280, Math.round(box.width)), height = Math.max(260, Math.round(box.height)), pad = 42;
  const points = series(), all = points.flatMap((p) => state.audit ? [p.prediction, p.audit_reference_rul] : [p.prediction]);
  const min = Math.min(0, ...all), max = Math.max(...all) * 1.08;
  svg.setAttribute('viewBox', '0 0 ' + width + ' ' + height); svg.innerHTML = '';
  for (let i = 0; i < 5; i += 1) { const y = pad + (height - 2 * pad) * i / 4; const value = max - (max - min) * i / 4; svg.append(svgNode('line', { x1:pad, x2:width-pad, y1:y, y2:y, stroke:'#d9e3e0', 'stroke-width':1 })); const label = svgNode('text', { x:pad-7, y:y+4, 'text-anchor':'end', fill:'#61737a', 'font-size':11 }); label.textContent = fmt(value, 0); svg.append(label); }
  const axis = svgNode('text', { x:12, y:22, fill:'#61737a', 'font-size':11 }); axis.textContent = 'RUL (cycles)'; svg.append(axis);
  svg.append(svgNode('path', { d:pathFor(points, 'prediction', width, height, pad, min, max), fill:'none', stroke:'#006d70', 'stroke-width':2.5 }));
  if (state.audit) svg.append(svgNode('path', { d:pathFor(points, 'audit_reference_rul', width, height, pad, min, max), fill:'none', stroke:'#9c6200', 'stroke-width':2, 'stroke-dasharray':'5 4' }));
  const point = points[state.index], x = pad + (width - 2 * pad) * state.index / (points.length - 1), y = height - pad - (height - 2 * pad) * (point.prediction - min) / Math.max(max - min, 1);
  svg.append(svgNode('line', { x1:x, x2:x, y1:pad, y2:height-pad, stroke:'#263f57', 'stroke-width':1, 'stroke-dasharray':'3 3' }));
  svg.append(svgNode('circle', { cx:x, cy:y, r:5, fill:'#006d70', stroke:'#fff', 'stroke-width':2 }));
  const marker = svgNode('text', { x:Math.min(width-pad, x+8), y:Math.max(pad+12, y-8), fill:'#10242d', 'font-size':11, 'font-weight':700 }); marker.textContent = 'Estimate ' + fmt(point.prediction, 1); svg.append(marker);
}
function renderAudit() {
  document.body.classList.toggle('audit-active', state.audit);
  const grid = $('audit-grid'); grid.innerHTML = '';
  const items = [
    ['Training data boundary', 'Target scaler and trend prior were fit on fit groups only.'],
    ['Holdout isolation', 'Held-out labels were excluded from fitting and model-selection.'],
    ['Input boundary', 'Target full-life min/max was not used for target input conversion.'],
    ['Model shape', data.preflight.target_features + ' target features; sequence length ' + data.preflight.sequence_length + '.'],
    ['Experiment designation', data.designation + '; confirmation status: ' + data.submission_eligibility + '.'],
    ['Operator / audit separation', 'Reference RUL is hidden in normal replay and revealed only by the audit control.']
  ];
  items.forEach((item) => { const div = document.createElement('div'); div.className = 'audit-item'; const b = document.createElement('b'); b.textContent = item[0]; const span = document.createElement('span'); span.textContent = item[1]; div.append(b, span); grid.append(div); });
  $('hashes').innerHTML = ''; Object.entries(data.audit_file_sha256).forEach(([name, hash]) => { const li = document.createElement('li'); li.textContent = name + ': sha256 ' + hash; $('hashes').append(li); });
}
function renderComparison() {
  const body = $('comparison-body'); body.innerHTML = '';
  [['N=2', data.metrics['2']], ['N=3', data.metrics['3']], ['Predeclared aggregate', data.metrics.aggregate]].forEach(([label, metric]) => {
    const row = document.createElement('tr'); [label, fmt(metric.transfer_rmse, 3), fmt(metric.scratch_rmse, 3), fmt(metric.ridge_rmse, 3), fmt(metric.best_trend_rmse, 3), '+' + fmt(metric.improvement_vs_scratch_pct, 2) + '%', metric.outer_fold_wins + '/' + metric.outer_folds].forEach((value) => { const cell = document.createElement('td'); cell.textContent = value; row.append(cell); }); body.append(row);
  });
}
function render() { const points = series(); $('endpoint').max = String(points.length - 1); $('endpoint').value = String(state.index); renderStatus(); renderChart(); renderAudit(); }
function stop() { if (state.timer) { clearInterval(state.timer); state.timer = null; $('play').textContent = 'Play'; } }
function play() { if (state.timer) return stop(); $('play').textContent = 'Pause'; state.timer = setInterval(() => { const points = series(); state.index = state.index >= points.length - 1 ? 0 : state.index + 1; render(); }, 55); }
function bind() {
  document.querySelectorAll('[data-shots]').forEach((button) => button.addEventListener('click', () => { stop(); state.shots = button.dataset.shots; state.trajectory = '1'; state.index = 0; document.querySelectorAll('[data-shots]').forEach((other) => other.setAttribute('aria-pressed', String(other === button))); setupSelect(); render(); }));
  $('trajectory').addEventListener('change', (event) => { stop(); state.trajectory = event.target.value; state.index = 0; render(); });
  $('endpoint').addEventListener('input', (event) => { stop(); state.index = Number(event.target.value); render(); });
  $('audit').addEventListener('change', (event) => { state.audit = event.target.checked; render(); });
  $('play').addEventListener('click', play); window.addEventListener('resize', renderChart);
}
metricCards(); setupSelect(); renderComparison(); bind(); render();
</script>
</body>
</html>'''
    return document.replace("__DASHBOARD_DATA__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-output", type=Path, default=DEFAULT_EXPERIMENT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_dashboard(args.experiment_output, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
