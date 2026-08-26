from scripts.exp_femto_ims_to_comsol_outerloo_v5 import PINNED_V4_ARGS, build_v4_argv


def test_v5_pins_the_audited_exploratory_condition_and_allows_runtime_controls():
    forwarded = build_v4_argv((
        "--output", "outputs/example",
        "--temp-dir", "outputs/example_tmp",
        "--device", "cpu",
        "--quick",
        "--seeds", "42,123",
        "--source-epochs", "8",
        "--target-epochs", "8",
    ))
    assert tuple(forwarded[:len(PINNED_V4_ARGS)]) == PINNED_V4_ARGS
    assert forwarded[len(PINNED_V4_ARGS):] == [
        "--output", "outputs/example",
        "--temp-dir", "outputs/example_tmp",
        "--device", "cpu",
        "--quick",
        "--seeds", "42,123",
        "--source-epochs", "8",
        "--target-epochs", "8",
    ]


def test_v5_rejects_condition_drift_in_both_argument_forms():
    locked = (
        "--source", "--source=femto",
        "--target-feature-tier", "--target-feature-tier=operational",
        "--target-prior-policy", "--target-prior-policy=ridge",
        "--blend-selection-policy", "--blend-selection-policy=inner_group_loo",
        "--calibrated-transfer-weight", "--calibrated-transfer-weight=0.5",
        "--calibrated-transfer-weight-policy", "--calibrated-transfer-weight-policy=fit_only_innerloo",
        "--experiment-designation", "--experiment-designation=confirmatory",
        "--target-include-deltas", "--use-context", "--target-adapter-alignment-steps=0",
    )
    for argument in locked:
        try:
            build_v4_argv((argument,))
        except ValueError as exc:
            assert argument in str(exc)
        else:
            raise AssertionError(f"locked option {argument} must be rejected")
