import numpy as np
import pytest
import torch

from scripts.exp_femto_ims_to_comsol_outerloo import (
    _fit_trend,
    _predict_trend,
    macro_rows_per_source_fold_shot,
    outer_schedule,
    prepare_outer_target,
    summarize_outer_folds,
)
from src.data.reaction_wheel_sim import SimulationTable
from src.models.comsol_sequence_baselines import build_comsol_sequence_baseline


def test_outer_schedule_covers_each_holdout_and_is_disjoint():
    groups = ("1", "2", "3", "4", "5")
    for n in (1, 2, 3):
        schedule = outer_schedule(groups, n)
        assert set(schedule) == set(groups)
        for holdout, split in schedule.items():
            assert split["test_groups"] == (holdout,)
            assert len(split["fit_groups"]) == n
            assert not set(split["fit_groups"]) & set(split["validation_groups"])
            assert holdout not in split["fit_groups"]
            assert holdout not in split["validation_groups"]


def test_outer_target_scaler_uses_fit_groups_only():
    rows = tuple(
        {"周期编号": str(i), "轨迹编号": group}
        for group in ("1", "2", "3", "4")
        for i in range(1, 31)
    )
    features = np.asarray([[float(group), float(i)] for group in (1, 2, 3, 4) for i in range(1, 31)], dtype=np.float32)
    target = np.asarray([100.0 - i for _group in (1, 2, 3, 4) for i in range(1, 31)], dtype=np.float32)
    table = SimulationTable("test", rows, features, target, ("f1", "f2"), np.asarray([row["轨迹编号"] for row in rows]), np.asarray([""] * len(rows)))
    train, validation, test, audit = prepare_outer_target(table, fit_groups=("1",), validation_groups=("2",), test_groups=("3",))
    assert audit["holdout_labels_used_for_fit"] is False
    assert audit["holdout_in_scaler_fit"] is False
    np.testing.assert_allclose(train.mean, np.asarray([1.0, 15.5], dtype=np.float32))
    assert test.scale == 99.0
    assert np.isfinite(train.x).all() and np.isfinite(validation.x).all() and np.isfinite(test.x).all()


def test_prepare_outer_target_rejects_overlap():
    rows = tuple({"周期编号": str(i), "轨迹编号": "1"} for i in range(1, 31))
    table = SimulationTable("test", rows, np.ones((30, 1), dtype=np.float32), np.ones(30, dtype=np.float32), ("f1",), np.asarray(["1"] * 30), np.asarray([""] * 30))
    with pytest.raises(ValueError):
        prepare_outer_target(table, fit_groups=("1",), validation_groups=("1",), test_groups=("1",))


def test_summary_requires_macro_gain_and_majority_fold_wins():
    rows = []
    for method, values in {
        "scratch_ensemble": (10.0, 10.0, 10.0, 10.0, 10.0),
        "transfer_ensemble": (7.0, 7.0, 7.0, 12.0, 12.0),
        "ridge_ensemble": (7.0, 7.0, 7.0, 7.0, 7.0),
    }.items():
        for index, value in enumerate(values, start=1):
            rows.append({
                "source": "femto", "n_shots": 1, "method": method, "outer_holdout": str(index),
                "raw_rmse": value, "mae": value / 2, "bias": 0.0, "normalized_rmse": value / 100,
            })
    summary = summarize_outer_folds(rows)
    transfer = next(row for row in summary if row["method"] == "transfer_ensemble")
    assert transfer["outer_fold_wins_vs_scratch"] == 3
    assert transfer["improvement_vs_scratch_pct"] == pytest.approx(10.0)
    assert transfer["positive_transfer_evidence"] is True


def test_extra_baseline_row_count_is_auditable():
    assert macro_rows_per_source_fold_shot(False, seeds=5) == 20
    assert macro_rows_per_source_fold_shot(True, seeds=5) == 44
    assert macro_rows_per_source_fold_shot(True, seeds=2) == 26


def test_time_trend_does_not_depend_on_holdout_labels():
    rows = tuple(
        {"周期编号": str(i), "轨迹编号": group}
        for group in ("1", "2", "3")
        for i in range(1, 31)
    )
    features = np.asarray(
        [[float(group), float(i)] for group in (1, 2, 3) for i in range(1, 31)],
        dtype=np.float32,
    )
    target = np.asarray(
        [100.0 - i for _group in (1, 2, 3) for i in range(1, 31)],
        dtype=np.float32,
    )
    table = SimulationTable(
        "test",
        rows,
        features,
        target,
        ("f1", "f2"),
        np.asarray([row["轨迹编号"] for row in rows]),
        np.asarray([""] * len(rows)),
    )
    train, _validation, test, _audit = prepare_outer_target(
        table,
        fit_groups=("1",),
        validation_groups=("2",),
        test_groups=("3",),
    )
    scale = max(float(np.max(train.endpoints)), 1.0)
    beta = _fit_trend(train, degree=1, endpoint_scale=scale)
    prediction = _predict_trend(beta, test, degree=1, endpoint_scale=scale)
    altered_test = type(test)(
        x=test.x,
        y=test.y + 1000.0,
        units=test.units,
        endpoints=test.endpoints,
        scale=test.scale,
        mean=test.mean,
        std=test.std,
    )
    np.testing.assert_allclose(
        prediction,
        _predict_trend(beta, altered_test, degree=1, endpoint_scale=scale),
    )


def test_target_sequence_baselines_have_finite_target_outputs():
    values = torch.randn(4, 20, 13)
    for name in ("gru", "tcn", "transformer"):
        model = build_comsol_sequence_baseline(name, n_features=13, sequence_length=20)
        output = model(values).rul
        assert output.shape == (4,)
        assert torch.isfinite(output).all()
