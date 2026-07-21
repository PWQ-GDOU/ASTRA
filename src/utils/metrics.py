"""
Evaluation metrics for RUL prediction.

Includes the standard NASA scoring function that asymmetrically
penalizes over-estimation (predicting later failure than actual).
"""

import torch
import numpy as np
from typing import Dict


def compute_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root Mean Square Error."""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2)).item()


def compute_mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return torch.mean(torch.abs(y_true - y_pred)).item()


def compute_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    NASA Prognostics Scoring Function.
    
    Asymmetric penalty: over-estimation (d > 0, predicting later failure)
    is penalized more heavily than under-estimation.
    
    score = sum(exp(d/13) - 1) for d >= 0  (over-estimate)
          = sum(exp(-d/10) - 1) for d < 0  (under-estimate)
    
    where d = RUL_predicted - RUL_actual (in cycles)
    """
    d = y_pred - y_true
    
    score_parts = torch.where(
        d >= 0,
        torch.exp(d / 13.0) - 1.0,
        torch.exp(-d / 10.0) - 1.0
    )
    return torch.sum(score_parts).item()


def compute_rul_metrics(y_true: torch.Tensor, y_pred: torch.Tensor,
                         y_std: torch.Tensor = None) -> Dict[str, float]:
    """
    Compute comprehensive RUL prediction metrics.
    
    Args:
        y_true: Ground truth RUL values
        y_pred: Predicted RUL values
        y_std: Prediction std (for uncertainty metrics)
    
    Returns:
        Dictionary of metrics
    """
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()
    
    metrics = {
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "Score": compute_score(y_true, y_pred),
    }
    
    # Relative accuracy
    metrics["MAPE"] = torch.mean(
        torch.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
    ).item()
    
    # R²
    ss_res = torch.sum((y_true - y_pred) ** 2).item()
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2).item()
    metrics["R2"] = 1 - ss_res / (ss_tot + 1e-8)
    
    # Uncertainty metrics (if std provided)
    if y_std is not None:
        y_std = y_std.detach().cpu()
        # Prediction Interval Coverage Probability (PICP)
        lower = y_pred - 1.96 * y_std
        upper = y_pred + 1.96 * y_std
        coverage = torch.mean(
            ((y_true >= lower) & (y_true <= upper)).float()
        ).item()
        metrics["PICP_95"] = coverage
        
        # Mean Prediction Interval Width (MPIW)
        metrics["MPIW_95"] = torch.mean(upper - lower).item()
        
        # Sharpness: normalized MPIW
        metrics["Sharpness"] = metrics["MPIW_95"] / (torch.mean(y_true).item() + 1e-8)
    
    return metrics


def compute_advance_warning(y_true: torch.Tensor, y_pred: torch.Tensor,
                             threshold: float = 0.15) -> float:
    """
    Compute advance warning capability: how early can we reliably predict failure.
    
    Args:
        y_true: True RUL values
        y_pred: Predicted RUL values
        threshold: Relative error threshold for "reliable" prediction
    
    Returns:
        Fraction of lifetime where prediction is within threshold
    """
    relative_error = torch.abs(y_true - y_pred) / (y_true + 1e-8)
    reliable = relative_error <= threshold
    return torch.mean(reliable.float()).item()
