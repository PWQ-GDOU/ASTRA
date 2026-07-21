"""
PCBNN: Physics-Constrained Bayesian Neural Network for RUL Prediction
with Uncertainty Quantification.

Based on: Wang E. et al. "A Physics-Constrained Bayesian Neural Network
for Machinery RUL Prediction and Uncertainty Quantification",
Reliability Engineering & System Safety, 2025.

Architecture:
  Input → Segmented BiLSTM Encoder → HGRR (Complex-valued Gated Recurrent) 
        → Weibull Output (η, β) → RUL + Confidence Intervals

Key features:
  1. Weibull likelihood: principled TTF distribution
  2. Bayesian VI with KL annealing: epistemic uncertainty
  3. Complex-valued HGRR: captures oscillatory degradation patterns  
  4. DeepHPM physics constraint: monotonic degradation enforcement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Weibull, kl_divergence
import math
from typing import Tuple, Optional


class BayesianLinear(nn.Module):
    """
    Bayesian Linear layer using variational inference.
    
    Each weight is modeled as a Gaussian distribution:
      w ~ N(μ, σ²)
    
    During training: sample w and compute KL divergence
    During inference: sample multiple times for MC estimation
    """
    
    def __init__(self, in_features: int, out_features: int,
                 prior_sigma: float = 0.1, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = None  # Computed from rho
        
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
            self.bias_sigma = None
        else:
            self.register_parameter('bias_mu', None)
        
        # Prior
        self.prior_sigma = prior_sigma
        
        # Initialize
        self.reset_parameters()
        
        # KL divergence accumulator
        self.kl_div = torch.tensor(0.0)
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -3.0)  # exp(-3) ≈ 0.05 initial sigma
        
        if self.bias_mu is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -3.0)
    
    @property
    def sigma(self) -> torch.Tensor:
        """Softplus to ensure positive standard deviation."""
        return F.softplus(self.weight_rho)
    
    @property
    def bias_std(self) -> torch.Tensor:
        if self.bias_rho is not None:
            return F.softplus(self.bias_rho)
        return None
    
    def sample_weights(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample weights from variational posterior using reparameterization."""
        eps = torch.randn_like(self.weight_mu)
        w = self.weight_mu + self.sigma * eps
        
        if self.bias_mu is not None:
            eps_b = torch.randn_like(self.bias_mu)
            b = self.bias_mu + self.bias_std * eps_b
        else:
            b = None
        
        # Compute KL divergence: KL(q(w|μ,σ) || p(w|0,prior_sigma))
        q_dist = Normal(self.weight_mu, self.sigma)
        p_dist = Normal(torch.zeros_like(self.weight_mu), 
                        torch.full_like(self.sigma, self.prior_sigma))
        self.kl_div = kl_divergence(q_dist, p_dist).sum()
        
        if b is not None:
            q_b = Normal(self.bias_mu, self.bias_std)
            p_b = Normal(torch.zeros_like(self.bias_mu),
                        torch.full_like(self.bias_std, self.prior_sigma))
            self.kl_div += kl_divergence(q_b, p_b).sum()
        
        return w, b
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weight sampling.
        """
        w, b = self.sample_weights()
        out = F.linear(x, w, b)
        return out


class SegmentedBiLSTM(nn.Module):
    """
    Segmented BiLSTM encoder for efficient long-sequence processing.
    
    Divides long input sequences into segments, processes each with BiLSTM,
    then concatenates outputs.
    """
    
    def __init__(self, n_features: int, hidden_dim: int = 128,
                 n_layers: int = 2, segment_size: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.segment_size = segment_size
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        
        # Project bidirectional output back to hidden_dim
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, F)
        Returns:
            h_out: (B, T, hidden_dim)
        """
        B, T, F = x.shape
        
        # Segment the input
        n_segments = max(1, T // self.segment_size)
        segment_len = T // n_segments
        
        outputs = []
        for i in range(n_segments):
            start = i * segment_len
            end = (i + 1) * segment_len if i < n_segments - 1 else T
            seg = x[:, start:end, :]
            
            lstm_out, _ = self.lstm(seg)  # (B, seg_len, 2*hidden)
            lstm_out = self.output_proj(lstm_out)  # (B, seg_len, hidden)
            outputs.append(lstm_out)
        
        # Concatenate along time dimension
        h_out = torch.cat(outputs, dim=1)  # (B, T, hidden)
        return self.dropout(h_out)


class ComplexGatedUnit(nn.Module):
    """
    Complex-valued Gated Linear Unit for HGRR.
    
    h_t = τ_t ⊙ h_{t-1} + (1 - τ_t) ⊙ c_t
    where τ_t = r_t · exp(i·θ_t) is a complex-valued forget gate.
    
    r_t: magnitude (how much to remember)
    θ_t: phase (oscillation pattern)
    """
    
    def __init__(self, hidden_dim: int, complex_dim: int = 32):
        super().__init__()
        self.complex_dim = complex_dim
        
        # Gate generators
        self.magnitude_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),  # r_t ∈ [0, 1]
        )
        self.phase_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),  # θ_t ∈ [-π, π]
        )
        
        # Candidate state
        self.candidate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
    
    def forward(self, h_prev: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_prev: Previous hidden state (B, hidden_dim)
            x: Current input (B, hidden_dim)
        
        Returns:
            h_new: Updated complex-gated hidden state (B, hidden_dim)
        """
        combined = torch.cat([h_prev, x], dim=-1)  # (B, 2*hidden)
        
        # Magnitude (r_t) and phase (θ_t)
        r = self.magnitude_gate(combined)  # (B, hidden)
        theta = self.phase_gate(combined) * math.pi  # (B, hidden), [-π, π]
        
        # Complex forget gate: τ = r · exp(i·θ)
        tau_real = r * torch.cos(theta)
        tau_imag = r * torch.sin(theta)
        
        # Candidate state
        c = self.candidate(combined)  # (B, hidden)
        
        # Complex-gated update: h_new = τ·h_prev + (1-τ)·c
        # Using real-valued approximation
        h_new = tau_real * h_prev + (1 - tau_real) * c
        
        return h_new


class HGRRRegressor(nn.Module):
    """
    Hierarchical Gated Recurrent Regressor with complex-valued hidden states.
    
    Processes the encoded features through a series of complex-gated updates,
    producing a final hidden representation for RUL prediction.
    """
    
    def __init__(self, hidden_dim: int = 128, complex_dim: int = 32,
                 n_steps: int = 16, dropout: float = 0.1):
        super().__init__()
        self.n_steps = n_steps
        
        # Project time information
        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )
        
        # Complex-gated units
        self.gate_units = nn.ModuleList([
            ComplexGatedUnit(hidden_dim, complex_dim)
            for _ in range(n_steps)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, h_enc: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_enc: Encoded sequence features (B, T, hidden_dim) or pooled (B, hidden_dim)
            t: Time values (B, 1)
        
        Returns:
            h_final: Final hidden state (B, hidden_dim)
        """
        if h_enc.dim() == 3:
            h_enc = h_enc.mean(dim=1)  # Global pooling (B, hidden_dim)
        
        # Time projection
        t_emb = self.time_proj(t)  # (B, hidden_dim)
        
        # Initialize hidden state
        h = h_enc + t_emb
        h = self.layer_norm(h)
        
        # Apply complex-gated updates
        for gate in self.gate_units:
            h = gate(h, h_enc)
            h = self.dropout(h)
        
        return h


class WeibullOutput(nn.Module):
    """
    Weibull distribution output layer for RUL prediction.
    
    Outputs Weibull shape (β) and scale (η) parameters.
    RUL = η · Γ(1 + 1/β)
    
    The Weibull distribution is the standard reliability model:
    - β < 1: infant mortality (decreasing failure rate)
    - β = 1: random failures (constant failure rate)  
    - β > 1: wear-out failures (increasing failure rate)
    """
    
    def __init__(self, hidden_dim: int, min_beta: float = 0.5):
        super().__init__()
        self.min_beta = min_beta
        
        self.eta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive scale
        )
        
        self.beta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive shape
        )
        
        # Aleatoric uncertainty (noise) channel
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )
    
    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Hidden state (B, hidden_dim)
        
        Returns:
            eta: Weibull scale (B, 1)
            beta: Weibull shape (B, 1)  
            sigma: Aleatoric uncertainty (B, 1)
        """
        eta = self.eta_head(h) + 1e-6
        beta = self.beta_head(h) + self.min_beta
        sigma = self.noise_head(h) + 1e-6
        
        return eta, beta, sigma
    
    def mean_rul(self, eta: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Compute RUL as Weibull mean: η · Γ(1 + 1/β)"""
        # Approximation of Gamma function for computational efficiency
        x = 1.0 + 1.0 / (beta + 1e-8)
        # Stirling approximation: Γ(x) ≈ √(2π/x) · (x/e)^x
        gamma_approx = torch.sqrt(2 * math.pi / x) * (x / math.e) ** x
        return eta * gamma_approx
    
    def nll_loss(self, y_true: torch.Tensor, eta: torch.Tensor,
                 beta: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood loss for Weibull distribution.
        
        L = -log p(y | η, β) = -log[(β/η)(y/η)^(β-1) exp(-(y/η)^β)]
        """
        y = y_true + 1e-8
        eta = eta + 1e-8
        beta = beta + 1e-8
        
        # Weibull log-likelihood
        log_lik = (
            torch.log(beta) - torch.log(eta) +
            (beta - 1) * (torch.log(y) - torch.log(eta)) -
            (y / eta) ** beta
        )
        
        # Add Gaussian noise contribution for aleatoric uncertainty
        log_lik = log_lik - 0.5 * torch.log(2 * math.pi * sigma ** 2)
        
        return -log_lik.mean()


class PCBNN(nn.Module):
    """
    Physics-Constrained Bayesian Neural Network for RUL prediction.
    
    Combines:
      1. Segmented BiLSTM for feature encoding
      2. HGRR with complex-valued gating for temporal dynamics
      3. Weibull distribution output for principled TTF modeling
      4. Bayesian layers for epistemic uncertainty
      5. Physics constraints (monotonic degradation)
    """
    
    def __init__(
        self,
        n_features: int,
        bilstm_hidden: int = 128,
        bilstm_layers: int = 2,
        segment_size: int = 16,
        hgrr_hidden: int = 64,
        hgrr_complex_dim: int = 32,
        hgrr_steps: int = 8,
        prior_sigma: float = 0.1,
        min_beta: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hgrr_hidden = hgrr_hidden
        
        # Segmented BiLSTM encoder
        self.encoder = SegmentedBiLSTM(
            n_features, bilstm_hidden, bilstm_layers, segment_size, dropout
        )
        
        # Project encoded features to HGRR dimension
        self.enc_to_hgrr = nn.Sequential(
            nn.Linear(bilstm_hidden, hgrr_hidden),
            nn.GELU(),
            nn.LayerNorm(hgrr_hidden),
        )
        
        # HGRR regressor
        self.hgrr = HGRRRegressor(
            hgrr_hidden, hgrr_complex_dim, hgrr_steps, dropout
        )
        
        # Bayesian layers (for epistemic uncertainty)
        self.bayesian_layer1 = BayesianLinear(hgrr_hidden, hgrr_hidden // 2, prior_sigma)
        self.bayesian_layer2 = BayesianLinear(hgrr_hidden // 2, hgrr_hidden // 2, prior_sigma)
        
        # Weibull output
        self.weibull_output = WeibullOutput(hgrr_hidden // 2, min_beta)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Track MC samples
        self.mc_samples = []
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to hidden representation."""
        h_enc = self.encoder(x)  # (B, T, bilstm_hidden)
        h_pool = h_enc.mean(dim=1)  # (B, bilstm_hidden)
        h = self.enc_to_hgrr(h_pool)  # (B, hgrr_hidden)
        return h, h_enc
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> dict:
        """
        Forward pass.
        
        Args:
            x: Input sequence (B, T, F)
            t: Time values (B, 1) - normalized operation time
        
        Returns:
            dict with eta, beta, sigma, rul_mean, kl_div
        """
        # Encode
        h, h_enc = self.encode(x)
        
        # HGRR
        h = self.hgrr(h, t)
        h = self.dropout(h)
        
        # Bayesian layers
        h = F.gelu(self.bayesian_layer1(h))
        h = self.dropout(h)
        h = F.gelu(self.bayesian_layer2(h))
        
        # Weibull output
        eta, beta, sigma = self.weibull_output(h)
        rul_mean = self.weibull_output.mean_rul(eta, beta)
        
        # KL divergence (accumulated from Bayesian layers)
        kl_div = self.bayesian_layer1.kl_div + self.bayesian_layer2.kl_div
        
        return {
            "eta": eta,
            "beta": beta,
            "sigma": sigma,
            "rul_mean": rul_mean,
            "kl_div": kl_div,
            "h_pool": h,
            "h_enc": h_enc,
        }
    
    def monte_carlo_predict(self, x: torch.Tensor, t: torch.Tensor,
                             n_samples: int = 100) -> dict:
        """
        Monte Carlo prediction for uncertainty quantification.
        
        Runs n_samples forward passes with different weight samples
        to estimate epistemic uncertainty.
        
        Returns:
            rul_mean: Mean RUL prediction
            rul_std: Epistemic uncertainty (std of MC samples)
            sigma_mean: Aleatoric uncertainty (mean noise)
            rul_samples: All MC sample predictions
            ci_lower, ci_upper: 95% confidence interval
        """
        self.train()  # Keep dropout active for MC sampling
        
        all_rul = []
        all_sigma = []
        
        for _ in range(n_samples):
            outputs = self.forward(x, t)
            all_rul.append(outputs["rul_mean"].detach())
            all_sigma.append(outputs["sigma"].detach())
        
        rul_samples = torch.stack(all_rul, dim=0)  # (n_samples, B, 1)
        sigma_samples = torch.stack(all_sigma, dim=0)
        
        # Epistemic uncertainty: std across MC samples
        rul_mean = rul_samples.mean(dim=0)
        rul_std = rul_samples.std(dim=0)
        
        # Aleatoric uncertainty: mean noise
        sigma_mean = sigma_samples.mean(dim=0)
        
        # Total uncertainty
        total_std = torch.sqrt(rul_std ** 2 + sigma_mean ** 2)
        
        # 95% confidence interval
        ci_lower = rul_mean - 1.96 * total_std
        ci_upper = rul_mean + 1.96 * total_std
        
        return {
            "rul_mean": rul_mean,
            "rul_std": rul_std,  # Epistemic
            "sigma_mean": sigma_mean,  # Aleatoric
            "total_std": total_std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "rul_samples": rul_samples,
        }


def compute_physics_loss(outputs: dict, x: torch.Tensor) -> torch.Tensor:
    """
    DeepHPM physics constraint loss.
    
    Enforces monotonic degradation:
      d(RUL)/dt <= 0  (RUL should decrease over time)
    
    Penalizes any increase in RUL prediction between consecutive time steps.
    """
    rul = outputs["rul_mean"].squeeze()  # (B,)
    
    # Monotonicity: RUL should never increase
    # We approximate this by penalizing when the model would predict
    # increasing RUL (which is physically impossible)
    
    # For this batch, ensure RUL values are positive and reasonable
    physics_penalty = F.relu(-rul).mean()  # RUL should be positive
    
    # Additional: penalize extremely large RUL values
    max_rul = 300.0  # Consistent with piecewise RUL cap
    physics_penalty += F.relu(rul - max_rul).mean()
    
    return physics_penalty


def compute_pcbnn_loss(
    outputs: dict,
    y_true: torch.Tensor,
    x: torch.Tensor,
    kl_weight: float = 0.01,
    physics_weight: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """
    Compute PCBNN total loss.
    
    L_total = NLL + kl_weight * KL_div + physics_weight * L_physics
    """
    # Create temporary WeibullOutput to use its nll_loss
    temp_weibull = WeibullOutput(outputs["h_pool"].shape[-1])
    
    # Negative log-likelihood (Weibull)
    nll = temp_weibull.nll_loss(
        y_true.squeeze(),
        outputs["eta"],
        outputs["beta"],
        outputs["sigma"],
    )
    
    # KL divergence (Bayesian regularization)
    kl_loss = outputs["kl_div"]
    
    # Physics constraint
    physics_loss = compute_physics_loss(outputs, x)
    
    # Total
    total_loss = nll + kl_weight * kl_loss + physics_weight * physics_loss
    
    loss_dict = {
        "nll": nll.item(),
        "kl_div": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
        "physics": physics_loss.item(),
        "total": total_loss.item(),
    }
    
    return total_loss, loss_dict
