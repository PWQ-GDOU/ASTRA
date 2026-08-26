from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reproduce_femto_ims_to_comsol.py"
SPEC = importlib.util.spec_from_file_location("reproduce_femto_ims_to_comsol", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reproduction_manifest_has_three_distinct_hashed_inputs() -> None:
    assert len(MODULE.EXPECTED_INPUTS) == 3
    assert len(set(MODULE.EXPECTED_INPUTS.values())) == 3


def test_reproduction_driver_refuses_to_overwrite_a_completed_result(tmp_path: Path) -> None:
    result = tmp_path / "full"
    result.mkdir()
    (result / "REPORT.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        MODULE._assert_fresh_result(result)
