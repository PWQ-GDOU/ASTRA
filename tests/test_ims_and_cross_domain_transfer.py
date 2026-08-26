from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.data.ims_strict import IMS_COMMON_CHANNELS, load_ims_archive, load_ims_processed_dir
from scripts.exp_femto_ims_to_comsol import _new_seeded_model, _new_target_model
from src.models.cross_domain_transfer import CrossDomainRULModel, copy_shared_encoder


def _write_measurement(path: Path, offset: float) -> None:
    t = np.linspace(0.0, 1.0, 128)
    values = np.column_stack(
        [
            np.sin(2.0 * np.pi * t) + offset,
            np.cos(4.0 * np.pi * t) + offset * 0.5,
            np.sin(8.0 * np.pi * t) * 0.25,
        ]
    )
    np.savetxt(path, values)


def test_ims_loader_groups_units_and_builds_ordinal_labels() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for unit in ("1st_test", "2nd_test"):
            folder = root / unit
            folder.mkdir()
            for index in range(4):
                _write_measurement(folder / f"{index + 1:05d}", float(index))
        units = load_ims_archive(root)
    assert sorted(units) == ["1st_test", "2nd_test"]
    assert units["1st_test"].features.shape == (4, 21)
    np.testing.assert_array_equal(units["1st_test"].raw_rul, [3, 2, 1, 0])


def test_ims_loader_uses_common_channels_across_8_and_4_channel_units() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for unit, channels in (("1st_test", 8), ("2nd_test", 4)):
            folder = root / unit
            folder.mkdir()
            for index in range(3):
                t = np.linspace(0.0, 1.0, 128)
                values = np.column_stack([np.sin((channel + 1) * t) + index for channel in range(channels)])
                np.savetxt(folder / f"{index + 1:05d}", values)
        units = load_ims_archive(root)
    assert units["1st_test"].features.shape[1] == IMS_COMMON_CHANNELS * 7
    assert units["2nd_test"].features.shape[1] == IMS_COMMON_CHANNELS * 7
    assert units["1st_test"].feature_names == units["2nd_test"].feature_names


def test_cross_domain_model_has_independent_adapters_and_finite_outputs() -> None:
    source = CrossDomainRULModel(21, 13)
    target = CrossDomainRULModel(21, 13)
    source_x = torch.randn(5, 20, 21)
    target_x = torch.randn(5, 20, 13)
    assert source.forward_source(source_x).rul.shape == (5,)
    assert target.forward_target(target_x).rul.shape == (5,)
    copy_shared_encoder(source, target)
    for parameter in target.encoder.parameters():
        parameter.requires_grad = False
    assert all(not parameter.requires_grad for parameter in target.encoder.parameters())
    assert torch.isfinite(target.forward_target(target_x).rul).all()


def test_model_initialization_is_stable_after_unrelated_rng_use() -> None:
    torch.manual_seed(1)
    _ = torch.rand(17)
    first = _new_seeded_model(21, 13, seed=2026)
    torch.manual_seed(999)
    _ = torch.rand(31)
    second = _new_seeded_model(21, 13, seed=2026)
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.equal(left, right)


def test_target_model_uses_explicit_initialization_seed() -> None:
    torch.manual_seed(11)
    first = _new_target_model(None, 21, 13, "scratch", "cpu", initialization_seed=3407)
    torch.manual_seed(22)
    second = _new_target_model(None, 21, 13, "scratch", "cpu", initialization_seed=3407)
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        assert torch.equal(left, right)


def test_processed_ims_loader_keeps_two_source_runs_disjoint() -> None:
    pytest = __import__("pytest")
    h5py = pytest.importorskip("h5py")
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for unit, n in (("2nd_test", 6), ("3rd_test", 7)):
            x_name = "x_train_2.hdf5" if unit == "2nd_test" else "x_train_3.hdf5"
            y_name = "y_train_2.hdf5" if unit == "2nd_test" else "y_train_3.hdf5"
            with h5py.File(root / x_name, "w") as handle:
                handle.create_dataset(x_name[:-5], data=np.ones((n, 20)))
            labels = np.column_stack([np.arange(n), np.arange(n), np.arange(n, 0, -1)])
            with h5py.File(root / y_name, "w") as handle:
                handle.create_dataset(y_name[:-5], data=labels)
        units = load_ims_processed_dir(root)
    assert tuple(units) == ("2nd_test", "3rd_test")
    assert units["2nd_test"].features.shape == (6, 20)
    np.testing.assert_array_equal(units["3rd_test"].raw_rul, np.arange(7, 0, -1))
