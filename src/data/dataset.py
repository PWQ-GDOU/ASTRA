"""
Dataset classes for degradation time series data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, List
from .augmentation import TimeAugmentation, FrequencyAugmentation, TimestampLatentMask
from .preprocess import extract_sliding_windows, build_piecewise_rul


class DegradationDataset(Dataset):
    """
    Single-domain degradation dataset for supervised learning.
    
    Args:
        data: Raw time series data (n_timesteps, n_features) or list of trajectories
        seq_len: Sequence length for sliding windows
        stride: Stride between windows
        max_rul: Maximum RUL value for piecewise labeling (None = unlimited)
        feature_cols: Column indices to use as features
        normalize: Whether to normalize data
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 128,
        stride: int = 1,
        max_rul: Optional[int] = None,
        feature_cols: Optional[List[int]] = None,
        normalize: bool = True,
    ):
        self.seq_len = seq_len
        
        # Handle single trajectory vs list of trajectories
        if isinstance(data, list):
            trajectories = data
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            trajectories = [data]
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        # Extract sliding windows from all trajectories
        all_X, all_y = [], []
        for traj in trajectories:
            X, y = extract_sliding_windows(
                traj, seq_len, stride, feature_cols, label_col=None
            )
            if len(X) > 0:
                if normalize:
                    mean = X.mean(axis=(0, 1), keepdims=True)
                    std = X.std(axis=(0, 1), keepdims=True) + 1e-8
                    X = (X - mean) / std
                
                all_X.append(X)
                all_y.append(y)
        
        if len(all_X) == 0:
            raise ValueError("No valid windows extracted from data")
        
        self.X = np.concatenate(all_X, axis=0).astype(np.float32)
        self.y = np.concatenate(all_y, axis=0).astype(np.float32)
        
        # Apply piecewise RUL if specified
        if max_rul is not None:
            self.y = build_piecewise_rul(self.y, max_rul)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])


class ContrastiveDegradationDataset(Dataset):
    """
    Dataset for contrastive self-supervised learning (SHRDL).
    
    Returns pairs of augmented views of the same sample.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        seq_len: int = 128,
        stride: int = 1,
        feature_cols: Optional[List[int]] = None,
        normalize: bool = True,
        crop_min: float = 0.5,
        crop_max: float = 0.8,
        mask_ratio: float = 0.15,
    ):
        # Extract windows
        if isinstance(data, list):
            trajectories = data
        elif isinstance(data, np.ndarray) and data.ndim == 2:
            trajectories = [data]
        else:
            trajectories = []
        
        all_X = []
        for traj in trajectories:
            X, _ = extract_sliding_windows(traj, seq_len, stride, feature_cols)
            if len(X) > 0:
                if normalize:
                    X = (X - X.mean()) / (X.std() + 1e-8)
                all_X.append(X)
        
        self.X = np.concatenate(all_X, axis=0).astype(np.float32) if all_X else np.array([])
        
        self.time_aug = TimeAugmentation(crop_min=crop_min, crop_max=crop_max)
        self.freq_aug = FrequencyAugmentation(n_fft=seq_len // 2)
        self.mask_aug = TimestampLatentMask(mask_ratio=mask_ratio)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.FloatTensor(self.X[idx])
        
        # Generate two augmented views
        view1 = self.time_aug(x, mode='random')
        view2 = self.time_aug(x, mode='random')
        
        return view1, view2


class MultiDomainDataset(Dataset):
    """
    Multi-domain dataset for domain adaptation (PEUDA).
    
    Each sample includes a domain label.
    Source domain has RUL labels; target domain may or may not.
    """
    
    def __init__(
        self,
        source_data: List[np.ndarray],
        target_data: List[np.ndarray],
        seq_len: int = 128,
        stride: int = 1,
        feature_cols: Optional[List[int]] = None,
        normalize: bool = True,
        max_rul: Optional[int] = None,
        target_has_labels: bool = False,
    ):
        self.seq_len = seq_len
        self.target_has_labels = target_has_labels
        
        # Process source domain
        self.source_X, self.source_y = self._process_domain(
            source_data, seq_len, stride, feature_cols, normalize, max_rul
        )
        
        # Process target domain
        if target_has_labels:
            self.target_X, self.target_y = self._process_domain(
                target_data, seq_len, stride, feature_cols, normalize, max_rul
            )
        else:
            self.target_X, _ = self._process_domain(
                target_data, seq_len, stride, feature_cols, normalize, None
            )
            self.target_y = None
        
        self.n_source = len(self.source_X)
        self.n_target = len(self.target_X)
    
    def _process_domain(
        self, data: List[np.ndarray], seq_len: int, stride: int,
        feature_cols: Optional[List[int]], normalize: bool,
        max_rul: Optional[int]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        all_X, all_y = [], []
        for traj in data:
            X, y = extract_sliding_windows(traj, seq_len, stride, feature_cols)
            if len(X) > 0:
                if normalize:
                    X = (X - X.mean()) / (X.std() + 1e-8)
                all_X.append(X)
                all_y.append(y)
        
        if not all_X:
            return np.array([]), np.array([])
        
        X = np.concatenate(all_X, axis=0).astype(np.float32)
        y = np.concatenate(all_y, axis=0).astype(np.float32) if all_y else None
        
        if y is not None and max_rul is not None:
            y = build_piecewise_rul(y, max_rul)
        
        return X, y
    
    def __len__(self) -> int:
        return max(self.n_source, self.n_target)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Source sample (cycled if needed)
        source_idx = idx % self.n_source
        source_x = torch.FloatTensor(self.source_X[source_idx])
        source_y = torch.FloatTensor([self.source_y[source_idx]])
        
        # Target sample (cycled if needed)
        target_idx = idx % self.n_target
        target_x = torch.FloatTensor(self.target_X[target_idx])
        
        result = {
            "source_x": source_x,
            "source_y": source_y,
            "target_x": target_x,
        }
        
        if self.target_has_labels and self.target_y is not None:
            result["target_y"] = torch.FloatTensor([self.target_y[target_idx]])
        
        return result


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Important for batch norm and contrastive learning
    )
