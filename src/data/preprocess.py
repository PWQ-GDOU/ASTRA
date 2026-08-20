"""
Data loading and preprocessing for all datasets.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


def load_nozzle_data(data_dir: str) -> Dict[str, np.ndarray]:
    """
    Load user's nozzle ablation simulation data.
    
    Returns dict with:
        - throat_degradation: time series of throat degradation (N_timesteps, 6)
        - wall_ablation: wall ablation rate distribution (N_spatial, 6)
    """
    throat_file = os.path.join(data_dir, "喷管喉部退化时程.csv")
    wall_file = os.path.join(data_dir, "壁面烧蚀速率分布数据.csv")
    
    throat_data = pd.read_csv(throat_file, encoding='utf-8-sig')
    wall_data = pd.read_csv(wall_file, encoding='utf-8-sig')
    
    return {
        "throat_degradation": throat_data.values,
        "wall_ablation": wall_data.values,
        "throat_columns": throat_data.columns.tolist(),
        "wall_columns": wall_data.columns.tolist(),
    }


def load_cmapss(data_dir: str, subset: str = "FD001") -> Dict[str, np.ndarray]:
    """
    Load C-MAPSS turbofan engine degradation dataset.
    
    Expected format: .txt files with space-separated values.
    Columns: unit, time, op_setting_1, op_setting_2, op_setting_3, sensor_1..sensor_21
    """
    import h5py
    
    # Try HDF5 format first (common preprocessed version)
    h5_path = os.path.join(data_dir, "CMAPSSData.h5")
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            if subset in f:
                data = f[subset]['data'][:]
                labels = f[subset]['labels'][:]
                return {"data": data, "labels": labels, "subset": subset}
    
    # Fallback: load from raw txt
    train_file = os.path.join(data_dir, f"train_{subset}.txt")
    test_file = os.path.join(data_dir, f"test_{subset}.txt")
    rul_file = os.path.join(data_dir, f"RUL_{subset}.txt")
    
    if os.path.exists(train_file):
        train_df = pd.read_csv(train_file, sep='\s+', header=None)
        train_data = train_df.values
        return {"data": train_data, "subset": subset}
    
    print(f"Warning: C-MAPSS {subset} not found at {data_dir}")
    return {"data": None, "subset": subset}


def load_battery_data(data_dir: str, dataset: str = "nasa") -> Dict[str, np.ndarray]:
    """
    Load battery degradation data.
    
    Args:
        data_dir: Path to battery data directory
        dataset: "nasa" or "calce"
    
    Returns dict with:
        - cycles: dict mapping battery_id -> array of (cycle_num, capacity, voltage, current, temp)
    """
    import h5py
    
    h5_path = os.path.join(data_dir, f"{dataset}_battery.h5")
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            cycles = {}
            for key in f.keys():
                cycles[key] = f[key][:]
            return {"cycles": cycles, "dataset": dataset}
    
    # Try CSV format
    csv_path = os.path.join(data_dir, f"{dataset}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {"dataframe": df, "dataset": dataset}
    
    print(f"Warning: {dataset} battery data not found at {data_dir}")
    return {"dataset": dataset}


def extract_sliding_windows(
    data: np.ndarray,
    seq_len: int,
    stride: int = 1,
    feature_cols: Optional[List[int]] = None,
    label_col: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract sliding windows from a single degradation trajectory.
    
    Args:
        data: 2D array of shape (timesteps, features)
        seq_len: Window length
        stride: Stride between windows
        feature_cols: Which columns to use as input features
        label_col: Which column to use as RUL label
    
    Returns:
        X: (n_windows, seq_len, n_features)
        y: (n_windows,) RUL labels
    """
    if feature_cols is None:
        feature_cols = list(range(data.shape[1] - 1))  # Exclude time column by default
    
    if label_col is None:
        label_col = data.shape[1] - 1  # Last column as label by default
    
    n_timesteps = data.shape[0]
    windows = []
    labels = []
    
    for start in range(0, n_timesteps - seq_len + 1, stride):
        end = start + seq_len
        window = data[start:end, feature_cols]
        
        # RUL label: remaining time from the END of window to end of trajectory
        rul = n_timesteps - end
        
        windows.append(window)
        labels.append(rul)
    
    if len(windows) == 0:
        return np.array([]).reshape(0, seq_len, len(feature_cols)), np.array([])
    
    return np.stack(windows), np.array(labels)


def extract_sliding_windows_multi_traj(
    trajectories: List[np.ndarray],
    seq_len: int,
    stride: int = 1,
    feature_cols: Optional[List[int]] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sliding windows from multiple trajectories.
    
    Returns:
        X: (total_windows, seq_len, n_features)
        y: (total_windows,) RUL labels
        domain_ids: (total_windows,) domain/trajectory identifiers
    """
    all_X, all_y, all_domains = [], [], []
    
    for domain_id, traj in enumerate(trajectories):
        X, y = extract_sliding_windows(traj, seq_len, stride, feature_cols)
        if len(X) > 0:
            if normalize:
                # Normalize per trajectory
                X = (X - X.mean(axis=(0, 1), keepdims=True)) / (X.std(axis=(0, 1), keepdims=True) + 1e-8)
            
            all_X.append(X)
            all_y.append(y)
            all_domains.append(np.full(len(y), domain_id))
    
    if len(all_X) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return (
        np.concatenate(all_X, axis=0),
        np.concatenate(all_y, axis=0),
        np.concatenate(all_domains, axis=0),
    )


def build_piecewise_rul(y: np.ndarray, max_rul: int = 130) -> np.ndarray:
    """
    Piecewise linear RUL labels (standard for C-MAPSS).
    RUL is capped at max_rul to avoid extremely long-range prediction.
    """
    y = np.clip(y, 0, max_rul)
    return y.astype(np.float32)
