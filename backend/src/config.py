"""
Global Configuration for Spacecraft Component RUL Prediction System
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# ==================== Paths ====================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")

# User's nozzle ablation data
NOZZLE_DATA_DIR = r"D:\数据集\航天器"

# ==================== SHRDL Pretraining Config ====================
@dataclass
class SHRDLConfig:
    """Configuration for SHRDL self-supervised pre-training."""
    # Encoder
    d_model: int = 128
    n_heads: int = 4
    n_adn_blocks: int = 2
    d_feedforward: int = 256
    dropout: float = 0.1
    
    # WDL (Working Condition Decomposition)
    wdl_num_heads: int = 4
    
    # NDL (Noise Decomposition)
    ndl_window_sizes: List[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    
    # CIMCL Contrastive Learning
    temperature: float = 0.07
    momentum: float = 0.999
    memory_bank_size: int = 4096
    feature_dim: int = 64  # Projection head output dim
    cimcl_beta: float = 400.0
    cimcl_gamma: float = 100.0
    
    # Training
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 200
    warmup_epochs: int = 10
    
    # Data
    seq_len: int = 128
    crop_min_ratio: float = 0.5
    crop_max_ratio: float = 0.8
    mask_ratio: float = 0.15

# ==================== PEUDA Domain Adaptation Config ====================
@dataclass
class PEUDAConfig:
    """Configuration for PEUDA cross-domain adaptation."""
    # Feature extractor
    d_model: int = 128
    n_heads: int = 4
    n_transformer_layers: int = 2
    dropout: float = 0.1
    
    # Frequency domain
    n_fft: int = 64
    hop_length: int = 16
    
    # Adversarial training
    discriminator_hidden: int = 256
    grl_lambda: float = 1.0  # Gradient reversal scaling
    
    # Momentum contrastive (MCL)
    mcl_momentum: float = 0.999
    mcl_queue_size: int = 2048
    mcl_temperature: float = 0.07
    
    # Self-supervised pretraining (target domain only)
    ssl_temperature: float = 0.5
    
    # Loss weights
    lambda_adv: float = 1.0
    lambda_mcl: float = 0.5
    lambda_ssl: float = 1.0
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    ssl_epochs: int = 50  # Target-only pretraining
    da_epochs: int = 100  # Domain adaptation
    
    # Data
    seq_len: int = 128

# ==================== PCBNN Uncertainty Config ====================
@dataclass
class PCBNNConfig:
    """Configuration for PCBNN uncertainty quantification."""
    # BiLSTM encoder
    bilstm_hidden: int = 128
    bilstm_layers: int = 2
    segment_size: int = 16
    
    # HGRR (Hierarchical Gated Recurrent Regressor)
    hgrr_hidden: int = 64
    hgrr_complex_dim: int = 32
    
    # Bayesian inference
    prior_sigma: float = 0.1
    n_monte_carlo_samples: int = 100
    
    # Weibull output
    min_beta: float = 0.5  # Minimum shape parameter
    
    # Physics constraints (DeepHPM)
    lambda_physics: float = 0.1
    monotonicity_penalty: float = 1.0
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 100
    kl_annealing_start: int = 10
    kl_annealing_end: int = 50

# ==================== Training Pipeline Config ====================
@dataclass
class TrainingConfig:
    """Master configuration for the entire training pipeline."""
    shrdl: SHRDLConfig = field(default_factory=SHRDLConfig)
    peuda: PEUDAConfig = field(default_factory=PEUDAConfig)
    pcbnn: PCBNNConfig = field(default_factory=PCBNNConfig)
    
    # Source domains (for pretraining)
    source_datasets: List[str] = field(default_factory=lambda: [
        "nozzle_ablation",  # User's own data
        "cmapss",           # Turbofan engine
        "nasa_battery",     # NASA battery
        "calce_battery",    # CALCE battery
    ])
    
    # Target domains (for adaptation)
    target_datasets: List[str] = field(default_factory=lambda: [
        "reaction_wheel",
        "battery",
    ])
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50

# ==================== Dataset-specific Configs ====================
@dataclass
class DatasetConfig:
    """Configuration for a specific dataset."""
    name: str
    n_features: int
    seq_len: int = 128
    stride: int = 1
    normalize: bool = True
    # RUL label type: "linear" or "piecewise"
    rul_mode: str = "linear"

# Predefined dataset configs
DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "nozzle_ablation": DatasetConfig(
        name="nozzle_ablation",
        n_features=5,  # time, ablation_depth, throat_radius, ablation_rate, mach
        seq_len=128,
        stride=2,
        rul_mode="linear",
    ),
    "cmapss": DatasetConfig(
        name="cmapss",
        n_features=21,  # 21 sensor channels
        seq_len=128,
        stride=1,
        rul_mode="piecewise",  # Piecewise linear degradation
    ),
    "nasa_battery": DatasetConfig(
        name="nasa_battery",
        n_features=4,  # voltage, current, temperature, capacity
        seq_len=64,
        stride=4,
        rul_mode="linear",
    ),
    "calce_battery": DatasetConfig(
        name="calce_battery",
        n_features=4,
        seq_len=64,
        stride=4,
        rul_mode="linear",
    ),
    "reaction_wheel": DatasetConfig(
        name="reaction_wheel",
        n_features=3,  # speed, current, temperature
        seq_len=128,
        stride=2,
        rul_mode="linear",
    ),
    "battery": DatasetConfig(
        name="battery",
        n_features=4,  # voltage, current, temperature, cycle
        seq_len=64,
        stride=4,
        rul_mode="linear",
    ),
}
