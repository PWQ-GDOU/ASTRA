"""Pinned exploratory replication for FEMTO/IMS-to-COMSOL outer-LOO transfer.

The five COMSOL outer folds informed the v5 condition design.  This entry point
therefore writes an explicit exploratory designation and prevents callers from
silently changing the source, target feature tier, or fit-only selection policy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import exp_femto_ims_to_comsol_outerloo_v4 as v4


PINNED_V4_ARGS = (
    "--source", "femto",
    "--target-feature-tier", "operational",
    "--target-prior-policy", "ridge",
    "--blend-selection-policy", "inner_group_loo",
    "--calibrated-transfer-weight-policy", "fit_only_innerloo",
    "--experiment-designation", "exploratory_post_hoc_replication",
)

_LOCKED_OPTIONS = frozenset({
    "--source",
    "--target-feature-tier",
    "--target-prior-policy",
    "--blend-selection-policy",
    "--calibrated-transfer-weight",
    "--calibrated-transfer-weight-policy",
    "--experiment-designation",
    "--target-include-deltas",
    "--use-context",
    "--target-adapter-alignment-steps",
})


def build_v4_argv(user_argv: Sequence[str]) -> list[str]:
    """Prepend the audited v5 condition and reject drift in locked arguments."""
    conflicts = [arg for arg in user_argv if arg.split("=", 1)[0] in _LOCKED_OPTIONS]
    if conflicts:
        raise ValueError(
            "v5 locks experimental-condition arguments; remove: "
            + ", ".join(conflicts)
        )
    return [*PINNED_V4_ARGS, *user_argv]


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        forwarded = build_v4_argv(arguments)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    previous_argv = sys.argv
    try:
        sys.argv = [str(v4.__file__), *forwarded]
        v4.main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
