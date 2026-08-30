"""
Standard LSTM baseline for RUL prediction comparison.
No pretraining, no domain adaptation — pure supervised learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMBaseline(nn.Module):
    """
    Standard LSTM-based RUL predictor.
    
    Used as baseline for comparison (scoring item 3.2).
    """
    
    def __init__(self, n_features: int, hidden_dim: int = 128, 
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Softplus(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            rul: (B, 1)
        """
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        h_last = lstm_out[:, -1, :]  # (B, hidden)
        rul = self.regressor(h_last)
        return rul


class CNNLSTMBaseline(nn.Module):
    """
    CNN-LSTM hybrid baseline.
    """
    
    def __init__(self, n_features: int, hidden_dim: int = 128,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim // 2, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> transpose for Conv1d: (B, F, T)
        x_conv = self.conv(x.transpose(1, 2))  # (B, hidden, T)
        x_conv = x_conv.transpose(1, 2)  # (B, T, hidden)
        
        lstm_out, _ = self.lstm(x_conv)
        return self.regressor(lstm_out[:, -1, :])
