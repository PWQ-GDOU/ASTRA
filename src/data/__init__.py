"""Data package exports with lazy loading for lightweight experiment entrypoints."""
from __future__ import annotations

__all__ = [
    "DegradationDataset", "MultiDomainDataset", "create_dataloaders",
    "TimeAugmentation", "FrequencyAugmentation",
    "load_nozzle_data", "load_cmapss", "load_battery_data",
]


def __getattr__(name: str):
    if name in {"DegradationDataset", "MultiDomainDataset", "create_dataloaders"}:
        from .dataset import DegradationDataset, MultiDomainDataset, create_dataloaders
        return {
            "DegradationDataset": DegradationDataset,
            "MultiDomainDataset": MultiDomainDataset,
            "create_dataloaders": create_dataloaders,
        }[name]
    if name in {"TimeAugmentation", "FrequencyAugmentation"}:
        from .augmentation import TimeAugmentation, FrequencyAugmentation
        return {"TimeAugmentation": TimeAugmentation, "FrequencyAugmentation": FrequencyAugmentation}[name]
    if name in {"load_nozzle_data", "load_cmapss", "load_battery_data"}:
        from .preprocess import load_nozzle_data, load_cmapss, load_battery_data
        return {
            "load_nozzle_data": load_nozzle_data,
            "load_cmapss": load_cmapss,
            "load_battery_data": load_battery_data,
        }[name]
    raise AttributeError(name)
