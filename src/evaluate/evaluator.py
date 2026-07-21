"""
Evaluation module for RUL prediction results.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.metrics import compute_rul_metrics


class RULEvaluator:
    """Comprehensive evaluator for comparing RUL prediction methods."""
    
    def __init__(self, method_names: List[str]):
        self.method_names = method_names
        self.results = {name: {} for name in method_names}
    
    def add_results(self, method: str, y_true: np.ndarray, 
                    y_pred: np.ndarray, y_std: Optional[np.ndarray] = None):
        """Add prediction results for a method."""
        y_true_t = torch.FloatTensor(y_true)
        y_pred_t = torch.FloatTensor(y_pred)
        y_std_t = torch.FloatTensor(y_std) if y_std is not None else None
        
        self.results[method] = compute_rul_metrics(y_true_t, y_pred_t, y_std_t)
        self.results[method]["predictions"] = y_pred
        self.results[method]["ground_truth"] = y_true
    
    def compare(self) -> str:
        """Generate comparison table."""
        header = f"{'Method':<30} {'RMSE':>10} {'MAE':>10} {'Score':>12} {'R2':>8}"
        sep = "-" * 75
        lines = [header, sep]
        
        for name in self.method_names:
            if name in self.results and self.results[name]:
                r = self.results[name]
                line = f"{name:<30} {r.get('RMSE', 0):>10.4f} {r.get('MAE', 0):>10.4f} {r.get('Score', 0):>12.2f} {r.get('R2', 0):>8.4f}"
                lines.append(line)
        
        return "\n".join(lines)
    
    def get_best_method(self, metric: str = "RMSE") -> str:
        """Get the best-performing method by metric."""
        best_name = None
        best_value = float('inf') if metric != 'R2' else float('-inf')
        
        for name, r in self.results.items():
            if not r:
                continue
            val = r.get(metric)
            if val is None:
                continue
            if (metric != 'R2' and val < best_value) or \
               (metric == 'R2' and val > best_value):
                best_value = val
                best_name = name
        
        return best_name
