import numpy as np
import torch

from scripts.exp_femto_ims_to_comsol_outerloo_v4 import (
    _train_residual_model,
    _select_prior,
    _select_residual_blend_weight,
    _select_residual_blend_weight_raw,
    _select_transfer_trend_weight,
    _select_transfer_ridge_weight,
    _mix_transfer_with_trend,
    _fit_validation_affine_calibrator,
    _apply_validation_affine_calibrator,
    _align_target_adapter_to_source,
    _source_unit_weights,
    AFFINE_CALIBRATED_TRANSFER_METHOD,
    AFFINE_CALIBRATED_SCRATCH_METHOD,
    CALIBRATED_TRANSFER_METHOD,
    CALIBRATED_SCRATCH_METHOD,
    _summarize,
    _summarize_all_shots,
    _select_group_balanced_mixture_weight,
    _prepare_inner_group_loo_target,
    _select_inner_group_loo_weights,
)
from scripts.exp_femto_ims_to_comsol import PreparedData
from src.data.reaction_wheel_sim import SimulationTable
from src.models.cross_domain_transfer import CrossDomainRULModel, TrendResidualRULModel, copy_shared_encoder_to_trend_residual


def _data(y: np.ndarray) -> PreparedData:
    n = len(y)
    return PreparedData(
        x=np.zeros((n, 20, 3), dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        units=np.asarray(["1"] * n),
        endpoints=np.arange(n, dtype=np.int64),
        scale=100.0,
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
    )


def test_source_unit_weights_assign_equal_total_loss_mass_without_target_data():
    units = np.asarray(["short"] * 2 + ["medium"] * 4 + ["long"] * 8)
    weights, audit = _source_unit_weights(units)
    assert weights.shape == units.shape
    assert np.isfinite(weights).all()
    assert np.all(weights > 0.0)
    assert audit["policy"] == "unit_equal_weighted_loss"
    assert audit["target_data_used"] is False
    masses = [float(weights[units == unit].sum()) for unit in ("short", "medium", "long")]
    np.testing.assert_allclose(masses, np.repeat(masses[0], len(masses)))


def test_source_unit_weights_reject_empty_or_single_unit_inputs():
    for units in (np.asarray([], dtype=str), np.asarray(["only"] * 3)):
        try:
            _source_unit_weights(units)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid source unit weighting input must be rejected")


def test_transfer_checkpoint_is_post_unfreeze_when_required():
    train = _data(np.linspace(1.0, 0.1, 8))
    validation = _data(np.linspace(0.95, 0.05, 8))
    model = TrendResidualRULModel(target_features=3)
    _, diagnostics = _train_residual_model(
        model,
        train,
        validation,
        np.zeros(len(train.y), dtype=np.float32),
        np.zeros(len(validation.y), dtype=np.float32),
        device="cpu",
        seed=42,
        epochs=4,
        warmup_frozen_epochs=1,
        require_unfrozen_checkpoint=True,
        patience=20,
    )
    assert diagnostics["warmup_frozen_epochs"] == 1
    assert diagnostics["encoder_unfreeze_epoch"] == 2
    assert diagnostics["best_epoch"] > diagnostics["warmup_frozen_epochs"]
    assert diagnostics["best_checkpoint_after_encoder_unfreeze"] is True


def test_transfer_checkpoint_requirement_rejects_frozen_path():
    train = _data(np.linspace(1.0, 0.1, 4))
    validation = _data(np.linspace(0.95, 0.05, 4))
    try:
        _train_residual_model(
            TrendResidualRULModel(target_features=3),
            train,
            validation,
            np.zeros(len(train.y), dtype=np.float32),
            np.zeros(len(validation.y), dtype=np.float32),
            device="cpu",
            seed=42,
            epochs=2,
            frozen_encoder=True,
            require_unfrozen_checkpoint=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("frozen path must not satisfy unfrozen checkpoint requirement")


def test_trend_prior_selection_does_not_read_holdout_labels():
    train = _data(np.linspace(1.0, 0.1, 8))
    validation = _data(np.linspace(0.95, 0.05, 8))
    test = _data(np.linspace(0.9, 0.0, 8))
    altered = PreparedData(test.x, test.y + 999.0, test.units, test.endpoints, test.scale, test.mean, test.std)
    first, diagnostics = _select_prior(train, validation, test)
    second, altered_diagnostics = _select_prior(train, validation, altered)
    np.testing.assert_allclose(first["test"], second["test"])
    assert diagnostics["selected_method"] == altered_diagnostics["selected_method"]
    assert diagnostics["holdout_labels_used_for_prior"] is False


def test_target_adapter_alignment_uses_fit_inputs_only_and_preserves_source_encoder():
    torch.manual_seed(17)
    source_model = CrossDomainRULModel(source_features=4, target_features=3)
    target_first = TrendResidualRULModel(target_features=3)
    target_second = TrendResidualRULModel(target_features=3)
    copy_shared_encoder_to_trend_residual(source_model, target_first)
    target_second.load_state_dict(target_first.state_dict())
    rng = np.random.default_rng(17)
    source_train = PreparedData(
        x=rng.normal(size=(9, 20, 4)).astype(np.float32),
        y=rng.normal(size=9).astype(np.float32),
        units=np.asarray(["source"] * 9),
        endpoints=np.arange(9, dtype=np.int64),
        scale=100.0,
        mean=np.zeros(4, dtype=np.float32),
        std=np.ones(4, dtype=np.float32),
    )
    target_train = PreparedData(
        x=rng.normal(size=(8, 20, 3)).astype(np.float32),
        y=np.linspace(0.1, 0.8, 8, dtype=np.float32),
        units=np.asarray(["fit"] * 8),
        endpoints=np.arange(8, dtype=np.int64),
        scale=100.0,
        mean=np.zeros(3, dtype=np.float32),
        std=np.ones(3, dtype=np.float32),
    )
    altered_labels = PreparedData(
        target_train.x, target_train.y + 999.0, target_train.units,
        target_train.endpoints, target_train.scale, target_train.mean,
        target_train.std,
    )
    source_encoder_before = {
        name: value.detach().clone() for name, value in source_model.encoder.state_dict().items()
    }
    target_projection_before = [
        parameter.detach().clone() for parameter in target_first.target_projection.parameters()
    ]
    first = _align_target_adapter_to_source(
        source_model, target_first, source_train, target_train,
        device="cpu", seed=17, steps=3,
    )
    second = _align_target_adapter_to_source(
        source_model, target_second, source_train, altered_labels,
        device="cpu", seed=17, steps=3,
    )
    assert first["enabled"] is True
    assert first["target_labels_used"] is False
    assert first["validation_or_holdout_used"] is False
    assert first == second
    for name, value in source_model.encoder.state_dict().items():
        assert torch.equal(value, source_encoder_before[name])
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(target_projection_before, target_first.target_projection.parameters())
    )
    for first_parameter, second_parameter in zip(
        target_first.target_projection.parameters(), target_second.target_projection.parameters()
    ):
        assert torch.allclose(first_parameter, second_parameter)


def test_trend_residual_model_has_finite_bounded_residual_output():
    torch.manual_seed(42)
    output = TrendResidualRULModel(target_features=3).forward_target(
        torch.randn(4, 20, 3), torch.full((4,), 0.5)
    )
    assert output.rul.shape == (4,)
    assert output.residual.shape == (4,)
    assert torch.isfinite(output.rul).all()
    assert torch.isfinite(output.residual).all()
    assert float(output.residual.abs().max()) <= 0.5 + 1.0e-6


def test_residual_blend_selection_is_validation_only_and_can_keep_prior():
    validation = _data(np.linspace(1.0, 0.2, 8))
    prior = validation.y.copy()
    bad_model = np.zeros(len(validation.y), dtype=np.float32)
    weight, score = _select_residual_blend_weight(validation, prior, bad_model)
    assert weight == 0.0
    assert score == 0.0


def test_source_residual_head_initializes_target_prefix_and_extra_columns_zero():
    torch.manual_seed(7)
    source = CrossDomainRULModel(source_features=4, target_features=3)
    target = TrendResidualRULModel(target_features=3, context_features=2)
    copy_shared_encoder_to_trend_residual(source, target, copy_residual_head=True)
    assert torch.allclose(target.encoder.weight_ih_l0, source.encoder.weight_ih_l0)
    assert torch.allclose(target.residual_head[0].weight[:, : source.head[0].in_features], source.head[0].weight)
    assert torch.allclose(target.residual_head[0].weight[:, source.head[0].in_features :], torch.zeros_like(target.residual_head[0].weight[:, source.head[0].in_features :]))
    assert torch.allclose(target.residual_head[3].weight, source.head[3].weight)
    output = target.forward_target(torch.randn(5, 20, 3), torch.full((5,), 0.5), torch.zeros(5, 2))
    assert torch.isfinite(output.rul).all()
    assert torch.isfinite(output.residual).all()

def test_residual_blend_weight_honors_trust_cap_and_rejects_invalid_ranges():
    validation = _data(np.linspace(1.0, 0.2, 8))
    prior = np.zeros(len(validation.y), dtype=np.float32)
    model = validation.y * validation.scale
    weight, score = _select_residual_blend_weight_raw(
        validation,
        prior,
        model,
        max_weight=0.25,
    )
    assert weight == 0.25
    assert score > 0.0
    for invalid in (-0.01, 1.01):
        try:
            _select_residual_blend_weight_raw(validation, prior, model, max_weight=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid trust cap must fail")


def test_ridge_calibrated_transfer_weight_is_validation_only_and_bounded():
    validation = _data(np.asarray([0.0, 0.0], dtype=np.float32))
    transfer = np.asarray([10.0, 10.0], dtype=np.float32)
    ridge = np.asarray([0.0, 0.0], dtype=np.float32)
    low_weight, low_score = _select_transfer_ridge_weight(
        validation, transfer, ridge, min_weight=0.25, max_weight=1.0
    )
    assert low_weight == 0.25
    assert np.isfinite(low_score)

    altered_validation = _data(np.asarray([9.9, 9.9], dtype=np.float32))
    high_weight, high_score = _select_transfer_ridge_weight(
        altered_validation, transfer, ridge, min_weight=0.25, max_weight=1.0
    )
    assert high_weight == 1.0
    assert np.isfinite(high_score)
    assert low_weight != high_weight

    for invalid_bounds in ((-0.01, 1.0), (0.25, 1.01), (0.75, 0.25)):
        try:
            _select_transfer_ridge_weight(
                validation, transfer, ridge,
                min_weight=invalid_bounds[0], max_weight=invalid_bounds[1],
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Ridge-calibrated weight bounds must fail")


def test_calibrated_transfer_mix_is_fixed_weight_and_finite():
    transfer = np.asarray([10.0, 20.0], dtype=np.float32)
    trend = np.asarray([0.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(
        _mix_transfer_with_trend(transfer, trend, transfer_weight=0.5),
        np.asarray([5.0, 12.0], dtype=np.float32),
    )
    for invalid in (-0.01, 1.01):
        try:
            _mix_transfer_with_trend(transfer, trend, transfer_weight=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid calibrated transfer weight must fail")


def test_calibrated_validation_weight_is_bounded_and_validation_only():
    validation = _data(np.asarray([0.0, 0.0], dtype=np.float32))
    transfer = np.asarray([10.0, 10.0], dtype=np.float32)
    trend = np.asarray([0.0, 0.0], dtype=np.float32)
    low_weight, low_score = _select_transfer_trend_weight(
        validation, transfer, trend, min_weight=0.25, max_weight=1.0
    )
    assert low_weight == 0.25
    assert np.isfinite(low_score)

    altered_validation = _data(np.asarray([0.1, 0.1], dtype=np.float32))
    high_weight, high_score = _select_transfer_trend_weight(
        altered_validation, transfer, trend, min_weight=0.25, max_weight=1.0
    )
    assert high_weight == 1.0
    assert np.isfinite(high_score)
    assert low_weight != high_weight

    for invalid_bounds in ((-0.01, 1.0), (0.25, 1.01), (0.75, 0.25)):
        try:
            _select_transfer_trend_weight(
                validation,
                transfer,
                trend,
                min_weight=invalid_bounds[0],
                max_weight=invalid_bounds[1],
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid calibrated validation bounds must fail")


def test_calibrated_validation_weight_does_not_accept_holdout_labels():
    validation = _data(np.asarray([0.1, 0.1], dtype=np.float32))
    transfer = np.asarray([10.0, 10.0], dtype=np.float32)
    trend = np.asarray([0.0, 0.0], dtype=np.float32)
    selected, score = _select_transfer_trend_weight(validation, transfer, trend)
    altered_holdout = _data(np.asarray([999.0, -999.0], dtype=np.float32))
    altered_selected, altered_score = _select_transfer_trend_weight(validation, transfer, trend)
    assert selected == altered_selected
    assert score == altered_score
    assert altered_holdout.y.tolist() == [999.0, -999.0]


def test_validation_affine_calibrator_is_bounded_and_applies_finitely():
    raw_validation = np.asarray([20.0, 40.0, 60.0, 80.0], dtype=np.float32)
    truth = 1.2 * raw_validation + 5.0
    validation = _data(truth / 100.0)
    calibrator = _fit_validation_affine_calibrator(validation, raw_validation)
    calibrated = _apply_validation_affine_calibrator(
        np.asarray([25.0, 55.0], dtype=np.float32), calibrator
    )
    assert calibrator["selected_mapping"] == "affine"
    assert 0.5 <= calibrator["selected_slope"] <= 1.5
    assert abs(float(calibrator["selected_intercept_raw"])) <= 25.0
    assert calibrated.shape == (2,)
    assert np.all(np.isfinite(calibrated))


def test_validation_affine_calibrator_uses_validation_labels_and_keeps_identity_when_best():
    prediction = np.asarray([15.0, 35.0, 55.0], dtype=np.float32)
    identity_validation = _data(prediction / 100.0)
    shifted_validation = _data((prediction + 20.0) / 100.0)
    identity = _fit_validation_affine_calibrator(identity_validation, prediction)
    shifted = _fit_validation_affine_calibrator(shifted_validation, prediction)
    outer_holdout = _data(np.asarray([999.0, -999.0, 123.0], dtype=np.float32))
    repeated = _fit_validation_affine_calibrator(shifted_validation, prediction)
    assert identity["selected_mapping"] == "identity"
    assert identity["selected_slope"] == 1.0
    assert identity["selected_intercept_raw"] == 0.0
    assert shifted["selected_mapping"] == "affine"
    assert shifted["selected_intercept_raw"] != identity["selected_intercept_raw"]
    assert shifted == repeated
    assert shifted["holdout_used"] is False
    assert outer_holdout.y.tolist() == [999.0, -999.0, 123.0]


def test_explicit_prior_policy_does_not_read_holdout_labels():
    train = _data(np.linspace(1.0, 0.1, 8))
    validation = _data(np.linspace(0.95, 0.05, 8))
    test = _data(np.linspace(0.9, 0.0, 8))
    altered = PreparedData(test.x, test.y - 321.0, test.units, test.endpoints, test.scale, test.mean, test.std)
    for policy in ("ridge", "linear", "quadratic", "huber"):
        first, diagnostics = _select_prior(train, validation, test, policy=policy)
        second, altered_diagnostics = _select_prior(train, validation, altered, policy=policy)
        np.testing.assert_allclose(first["test"], second["test"])
        assert diagnostics["selected_method"] == altered_diagnostics["selected_method"]
        assert diagnostics["holdout_labels_used_for_prior"] is False

def test_all_shot_summary_is_predeclared_and_counts_all_fold_units():
    rows = []
    for n in (1, 2, 3):
        for holdout in ("1", "2"):
            for method, score in (
                ("trend_residual_scratch_ensemble", 10.0),
                ("trend_residual_transfer_ensemble", 8.0),
                ("ridge_ensemble", 9.0),
                ("trend_linear_ensemble", 11.0),
                ("trend_quadratic_ensemble", 12.0),
                ("trend_huber_ensemble", 10.5),
            ):
                rows.append({
                    "source": "femto", "n_shots": n, "method": method,
                    "outer_holdout": holdout, "raw_rmse": score,
                    "mae": score / 2.0, "bias": 0.0, "normalized_rmse": score / 100.0,
                })
    transfer = next(item for item in _summarize_all_shots(rows) if item["method"] == "trend_residual_transfer_ensemble")
    assert transfer["n_shots"] == "all_predeclared"
    assert transfer["outer_folds"] == 6
    assert transfer["positive_transfer_evidence"] is True
    assert transfer["baseline_comparison_pass"] is True

def test_v4_transfer_acceptance_uses_trend_residual_scratch_reference():
    rows = []
    values = {
        "trend_residual_scratch_ensemble": (10.0,) * 5,
        "trend_residual_transfer_ensemble": (8.0,) * 5,
        "ridge_ensemble": (9.0,) * 5,
        "trend_linear_ensemble": (9.5,) * 5,
        "trend_quadratic_ensemble": (10.0,) * 5,
        "trend_huber_ensemble": (10.5,) * 5,
    }
    for method, scores in values.items():
        for index, score in enumerate(scores, start=1):
            rows.append({"source": "femto", "n_shots": 1, "method": method, "outer_holdout": str(index), "raw_rmse": score, "mae": score / 2.0, "bias": 0.0, "normalized_rmse": score / 100.0})
    transfer = next(item for item in _summarize(rows) if item["method"] == "trend_residual_transfer_ensemble")
    assert transfer["improvement_vs_scratch_pct"] == 20.0
    assert transfer["outer_fold_wins_vs_scratch"] == 5
    assert transfer["positive_transfer_evidence"] is True
    assert transfer["macro_vs_ridge_pct"] > 0.0


def test_all_shot_summary_reports_affine_calibrated_transfer_separately():
    rows = []
    for n in (1, 2, 3):
        for holdout in ("1", "2"):
            for method, score in (
                ("trend_residual_scratch_ensemble", 10.0),
                (f"{AFFINE_CALIBRATED_SCRATCH_METHOD}_ensemble", 9.0),
                (f"{AFFINE_CALIBRATED_TRANSFER_METHOD}_ensemble", 8.0),
                ("ridge_ensemble", 9.0),
                ("trend_linear_ensemble", 11.0),
                ("trend_quadratic_ensemble", 12.0),
                ("trend_huber_ensemble", 10.5),
            ):
                rows.append({
                    "source": "femto", "n_shots": n, "method": method,
                    "outer_holdout": holdout, "raw_rmse": score,
                    "mae": score / 2.0, "bias": 0.0,
                    "normalized_rmse": score / 100.0,
                })
    affine = next(
        item for item in _summarize_all_shots(rows)
        if item["method"] == f"{AFFINE_CALIBRATED_TRANSFER_METHOD}_ensemble"
    )
    assert affine["affine_calibrated_transfer_evidence"] is True
    assert affine["baseline_comparison_pass"] is True


def test_hybrid_acceptance_uses_matching_scratch_hybrid_not_raw_scratch():
    rows = []
    values = {
        "trend_residual_scratch_ensemble": 10.0,
        f"{CALIBRATED_SCRATCH_METHOD}_ensemble": 8.0,
        f"{CALIBRATED_TRANSFER_METHOD}_ensemble": 7.0,
        "ridge_ensemble": 7.5,
        "trend_linear_ensemble": 8.5,
        "trend_quadratic_ensemble": 9.0,
        "trend_huber_ensemble": 8.2,
    }
    for holdout in ("1", "2", "3"):
        for method, score in values.items():
            rows.append({
                "source": "femto", "n_shots": 1, "method": method,
                "outer_holdout": holdout, "raw_rmse": score, "mae": score / 2.0,
                "bias": 0.0, "normalized_rmse": score / 100.0,
            })
    summary = next(
        item for item in _summarize(rows)
        if item["method"] == f"{CALIBRATED_TRANSFER_METHOD}_ensemble"
    )
    assert summary["matched_scratch_method"] == f"{CALIBRATED_SCRATCH_METHOD}_ensemble"
    assert summary["matched_scratch_macro_raw_rmse"] == 8.0
    assert summary["improvement_vs_matched_scratch_pct"] == 12.5
    assert summary["strict_positive_transfer_evidence"] is True


def test_hybrid_without_matching_scratch_is_diagnostic_even_if_raw_scratch_is_worse():
    rows = []
    values = {
        "trend_residual_scratch_ensemble": 10.0,
        f"{CALIBRATED_TRANSFER_METHOD}_ensemble": 6.0,
        "ridge_ensemble": 8.0,
        "trend_linear_ensemble": 9.0,
        "trend_quadratic_ensemble": 10.0,
        "trend_huber_ensemble": 9.5,
    }
    for holdout in ("1", "2", "3"):
        for method, score in values.items():
            rows.append({
                "source": "femto", "n_shots": 1, "method": method,
                "outer_holdout": holdout, "raw_rmse": score, "mae": score / 2.0,
                "bias": 0.0, "normalized_rmse": score / 100.0,
            })
    summary = next(
        item for item in _summarize(rows)
        if item["method"] == f"{CALIBRATED_TRANSFER_METHOD}_ensemble"
    )
    assert summary["matched_scratch_comparator_available"] is False
    assert summary["strict_positive_transfer_evidence"] is False
    assert "missing matched scratch comparator" in summary["acceptance_reason"]


def test_all_shot_summary_pairs_each_n_with_its_own_outer_fold_reference():
    rows = []
    per_n = {1: (10.0, 8.0, 9.0), 2: (20.0, 15.0, 16.0)}
    for n_shots, (scratch, transfer, ridge) in per_n.items():
        for holdout in ("1", "2"):
            for method, score in (
                ("trend_residual_scratch_ensemble", scratch),
                ("trend_residual_transfer_ensemble", transfer),
                ("ridge_ensemble", ridge),
                ("trend_linear_ensemble", ridge + 2.0),
                ("trend_quadratic_ensemble", ridge + 3.0),
                ("trend_huber_ensemble", ridge + 1.0),
            ):
                rows.append({
                    "source": "femto", "n_shots": n_shots, "method": method,
                    "outer_holdout": holdout, "raw_rmse": score, "mae": score / 2.0,
                    "bias": 0.0, "normalized_rmse": score / 100.0,
                })
    summary = next(
        item for item in _summarize_all_shots(rows)
        if item["method"] == "trend_residual_transfer_ensemble"
    )
    assert summary["outer_folds"] == 4
    assert summary["matched_scratch_macro_raw_rmse"] == 15.0
    assert summary["ridge_macro_raw_rmse"] == 12.5
    assert summary["improvement_vs_matched_scratch_pct"] == (100.0 * (15.0 - 11.5) / 15.0)


def test_inner_group_loo_mixture_selection_is_group_balanced_and_deterministic():
    # Group A has many easy windows while group B has few hard windows. A pooled
    # window loss would favor the primary predictor; group-balanced RMSE must not.
    truth = np.asarray([0.0] * 100 + [10.0] * 2, dtype=np.float32)
    primary = np.asarray([0.0] * 100 + [30.0] * 2, dtype=np.float32)
    baseline = np.asarray([1.0] * 100 + [10.0] * 2, dtype=np.float32)
    groups = np.asarray(["A"] * 100 + ["B"] * 2)
    first = _select_group_balanced_mixture_weight(
        truth, primary, baseline, groups, min_weight=0.0, max_weight=1.0, grid_size=3
    )
    second = _select_group_balanced_mixture_weight(
        truth, primary, baseline, groups, min_weight=0.0, max_weight=1.0, grid_size=3
    )
    weight, score, audit = first
    assert first == second
    assert weight == 0.0
    assert score == 0.5
    assert audit["selection"] == "inner_leave_one_fit_group_out"
    assert audit["outer_holdout_used"] is False
    assert audit["outer_validation_used_for_selection"] is False
    assert audit["inner_groups"] == ["A", "B"]


def test_inner_group_loo_mixture_selection_rejects_bad_arrays_and_bounds():
    truth = np.asarray([1.0, 2.0], dtype=np.float32)
    prediction = np.asarray([1.0, 2.0], dtype=np.float32)
    groups = np.asarray(["A", "B"])
    invalid = (
        (truth, prediction[:-1], prediction, groups, 0.0, 1.0, 3),
        (truth, np.asarray([np.nan, 2.0], dtype=np.float32), prediction, groups, 0.0, 1.0, 3),
        (truth, prediction, prediction, np.asarray(["A", "A"]), 0.0, 1.0, 3),
        (truth, prediction, prediction, groups, 0.8, 0.2, 3),
        (truth, prediction, prediction, groups, 0.0, 1.0, 1),
    )
    for y, primary, baseline, ids, low, high, grid in invalid:
        try:
            _select_group_balanced_mixture_weight(
                y, primary, baseline, ids, min_weight=low, max_weight=high, grid_size=grid
            )
        except (ValueError, FloatingPointError):
            pass
        else:
            raise AssertionError("invalid inner group-LOO mixture inputs must fail")


def test_inner_group_loo_target_scaler_excludes_heldout_group():
    rows = tuple(
        {"周期编号": str(cycle), "轨迹编号": group}
        for group, _level in (("1", 1.0), ("2", 100.0), ("3", 1000.0))
        for cycle in range(1, 26)
    )
    table = SimulationTable(
        "inner-loo",
        rows,
        np.asarray([[level, float(cycle)] for _group, level in (("1", 1.0), ("2", 100.0), ("3", 1000.0)) for cycle in range(1, 26)], dtype=np.float32),
        np.asarray([30.0 - cycle for _group in ("1", "2", "3") for cycle in range(1, 26)], dtype=np.float32),
        ("level", "cycle"),
        np.asarray([row["轨迹编号"] for row in rows]),
        np.asarray([""] * len(rows)),
    )
    train, heldout, audit = _prepare_inner_group_loo_target(
        table, fit_groups=("1", "2"), heldout_group="3"
    )
    assert audit["target_scaler_fit_groups"] == ["1", "2"]
    assert audit["target_label_scale_fit_groups"] == ["1", "2"]
    assert audit["heldout_in_scaler_fit"] is False
    assert audit["outer_holdout_materialized"] is False
    assert set(train.units.tolist()) == {"1", "2"}
    assert set(heldout.units.tolist()) == {"3"}
    assert float(train.mean[0]) < 100.0


def test_inner_group_loo_target_rejects_overlap():
    table = SimulationTable(
        "tiny",
        tuple(),
        np.empty((0, 1), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        ("f",),
        np.asarray([], dtype=str),
        np.asarray([], dtype=str),
    )
    try:
        _prepare_inner_group_loo_target(table, fit_groups=("1",), heldout_group="1")
    except ValueError:
        pass
    else:
        raise AssertionError("inner heldout group must not be a scaler-fit group")


def test_inner_group_loo_n1_returns_explicit_validation_fallback():
    audit = _select_inner_group_loo_weights(
        source_bundle=None,
        source_models={},
        table=None,
        outer_holdout="5",
        n_shots=1,
        fit_groups=("2",),
        seeds=(42,),
        epochs=1,
        device="cpu",
        prior_policy="validation",
        use_context=False,
        max_blend_weight=1.0,
        transfer_residual_head=False,
        encoder_anchor_weight=0.0,
        transfer_warmup_epochs=1,
        target_include_deltas=False,
        target_adapter_alignment_steps=0,
    )
    assert audit["available"] is False
    assert audit["selection"] == "chronological_validation_fallback"
    assert audit["reason"] == "inner_group_loo_requires_at_least_two_outer_fit_groups"
    assert audit["outer_fit_groups"] == ["2"]
    assert audit["outer_holdout"] == "5"
    assert audit["outer_holdout_used"] is False
    assert audit["outer_validation_used_for_selection"] is True
