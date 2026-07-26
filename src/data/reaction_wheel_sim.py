"""Loader and audit helpers for the supplied COMSOL reaction-wheel simulation data.

The archive contains two distinct products:

* ``三_对齐周期数据集``: five complete trajectories with physical cycle/hour RUL.
* ``五_低误差窗口特征数据集``: four fault modes, five trajectories, and
  preassigned train/validation/test trajectory splits.

Feature tiers are explicit.  ``observable``/``operational`` exclude simulated
latent degradation state; ``oracle`` is only for leakage/sensitivity audits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from zipfile import ZipFile
import csv
import io
import hashlib
import json

import numpy as np


ALIGNED_FILE = "三_对齐周期数据集/全部周期特征.csv"
ALIGNED_META_FILE = "三_对齐周期数据集/退化轨迹元数据.csv"
LOW_ERROR_FILES = {
    "train": "五_低误差窗口特征数据集/训练集.csv",
    "validation": "五_低误差窗口特征数据集/验证集.csv",
    "test": "五_低误差窗口特征数据集/测试集.csv",
}

ALIGNED_OPERATIONAL_FEATURES = (
    "周期编号", "等效服役时长_小时", "拉升阶段时长_秒", "恒速阶段时长_秒",
    "指令角速度_弧度每秒", "拉升角速度斜率", "恒速角速度均值_弧度每秒",
    "恒速角速度标准差_弧度每秒", "拉升电流峰值_安培", "恒速电流均值_安培",
    "恒速电流有效值_安培", "电压均值_伏", "绕组温度均值_摄氏度",
)
ALIGNED_ESTIMATED_FEATURES = ALIGNED_OPERATIONAL_FEATURES + (
    "自适应扩展卡尔曼估计转矩常数", "自适应扩展卡尔曼估计黏性摩擦系数",
    "自适应扩展卡尔曼转矩常数协方差", "自适应扩展卡尔曼黏性摩擦协方差",
    "特征性能比", "性能裕度", "漂移核异常评分",
)
ALIGNED_ORACLE_FEATURES = ALIGNED_ESTIMATED_FEATURES + (
    "退化程度", "真实转矩常数", "真实黏性摩擦系数", "设计失效阈值比",
    "漂移核粒子滤波剩余寿命估计_周期", "漂移核粒子滤波剩余寿命第十百分位_周期",
    "漂移核粒子滤波剩余寿命第九十百分位_周期",
)

LOW_ERROR_META_COLUMNS = {
    "样本编号", "源文件", "故障类型", "轨迹编号", "退化等级", "窗口编号",
    "窗口开始时间_秒", "窗口结束时间_秒", "剩余寿命比例", "剩余寿命_等级",
}
LOW_ERROR_LATENT_TERMS = (
    "退化程度", "轴承磨损退化程度", "匝间短路退化程度", "永磁退磁退化程度",
    "附加磨损摩擦转矩", "电阻下降比例", "剩磁衰减比例", "相电阻", "相电感",
    "剩磁比例", "转动惯量",
)
LOW_ERROR_OBSERVABLE_EXTRA = (
    "三相电流有效值均值_安培", "转矩裕度_牛米", "摩擦功率_瓦", "直流电功率_瓦",
    "热裕度_摄氏度", "角速度均值_转每分", "振动健康指标",
)


@dataclass(frozen=True)
class SimulationTable:
    name: str
    rows: tuple[Mapping[str, str], ...]
    features: np.ndarray
    target: np.ndarray
    feature_names: tuple[str, ...]
    group_ids: np.ndarray
    split: np.ndarray

    def __len__(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class SimulationArchive:
    path: str
    sha256: str
    aligned: Mapping[str, SimulationTable]
    low_error: Mapping[str, SimulationTable]
    aligned_metadata: tuple[Mapping[str, str], ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "\x00" not in text[:2000] and text.count(",") > 5:
            return text
    return raw.decode("utf-8", errors="replace")


def _read_rows(archive: ZipFile, name: str) -> list[dict[str, str]]:
    text = _decode(archive.read(name))
    return list(csv.DictReader(io.StringIO(text)))


def _numeric(rows: Sequence[Mapping[str, str]], columns: Sequence[str]) -> np.ndarray:
    values = []
    for row in rows:
        values.append([float(row.get(column, "0") or 0.0) for column in columns])
    output = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _table(name: str, rows: Sequence[Mapping[str, str]], columns: Sequence[str], target_column: str, group_column: str, split_column: str | None) -> SimulationTable:
    split = np.asarray([row.get(split_column, "") if split_column else "" for row in rows])
    return SimulationTable(
        name=name,
        rows=tuple(rows),
        features=_numeric(rows, columns),
        target=np.asarray([float(row[target_column]) for row in rows], dtype=np.float32),
        feature_names=tuple(columns),
        group_ids=np.asarray([row[group_column] for row in rows]),
        split=split,
    )


def aligned_feature_names(tier: str) -> tuple[str, ...]:
    groups = {
        "operational": ALIGNED_OPERATIONAL_FEATURES,
        "estimated": ALIGNED_ESTIMATED_FEATURES,
        "oracle": ALIGNED_ORACLE_FEATURES,
    }
    if tier not in groups:
        raise ValueError(f"Unknown aligned feature tier: {tier}")
    return groups[tier]


def low_error_feature_names(rows: Sequence[Mapping[str, str]], tier: str) -> tuple[str, ...]:
    if not rows:
        raise ValueError("No low-error rows")
    header = list(rows[0])
    noisy = tuple(column for column in header if "带噪" in column)
    if tier == "noisy":
        return noisy
    if tier == "observable":
        return noisy + tuple(column for column in LOW_ERROR_OBSERVABLE_EXTRA if column in header)
    if tier == "oracle":
        latent = tuple(
            column for column in header
            if column not in LOW_ERROR_META_COLUMNS
            and column not in noisy
            and column not in ("剩余寿命比例", "剩余寿命_等级")
        )
        return noisy + tuple(column for column in LOW_ERROR_OBSERVABLE_EXTRA if column in header) + latent
    raise ValueError(f"Unknown low-error feature tier: {tier}")


def load_simulation_archive(path: str | Path) -> SimulationArchive:
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    with ZipFile(archive_path) as archive:
        aligned_rows = _read_rows(archive, ALIGNED_FILE)
        metadata = tuple(_read_rows(archive, ALIGNED_META_FILE))
        aligned = {
            tier: _table(
                f"aligned_{tier}",
                aligned_rows,
                aligned_feature_names(tier),
                "剩余寿命_周期",
                "轨迹编号",
                "数据划分",
            )
            for tier in ("operational", "estimated", "oracle")
        }
        low_error = {}
        for split, member in LOW_ERROR_FILES.items():
            rows = _read_rows(archive, member)
            for tier in ("noisy", "observable", "oracle"):
                low_error[f"low_error_{split}_{tier}"] = _table(
                    f"low_error_{split}_{tier}",
                    rows,
                    low_error_feature_names(rows, tier),
                    "剩余寿命_等级",
                    "轨迹编号",
                    None,
                )
    return SimulationArchive(
        path=str(archive_path),
        sha256=sha256_file(archive_path),
        aligned=aligned,
        low_error=low_error,
        aligned_metadata=metadata,
    )


def split_summary(table: SimulationTable) -> dict[str, object]:
    return {
        "name": table.name,
        "rows": len(table),
        "features": len(table.feature_names),
        "feature_names": list(table.feature_names),
        "groups": sorted(set(table.group_ids.tolist())),
        "splits": sorted(set(table.split.tolist())),
        "target_min": float(np.min(table.target)),
        "target_max": float(np.max(table.target)),
        "target_mean": float(np.mean(table.target)),
    }


def archive_summary(archive: SimulationArchive) -> dict[str, object]:
    return {
        "path": archive.path,
        "sha256": archive.sha256,
        "aligned_metadata_rows": len(archive.aligned_metadata),
        "aligned": {key: split_summary(value) for key, value in archive.aligned.items()},
        "low_error": {key: split_summary(value) for key, value in archive.low_error.items()},
    }
