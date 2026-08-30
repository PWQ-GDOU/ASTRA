"""
Wiener Process RUL Predictor (baseline for comparison).

Based on: Lin et al. "A Novel Interactive Prognosis Framework with
Nonlinear Wiener Process and Multi-Sensor Fusion for RUL Prediction",
Journal of Process Control, 2024.

Implements a nonlinear Wiener process with multi-sensor fusion
for degradation modeling and RUL prediction.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Optional


class WienerRULPredictor:
    """
    Nonlinear Wiener process-based RUL predictor.
    
    Degradation model:
      X(t) = X(0) + η · Λ(t; γ) + σ · B(t)
    
    where:
      X(t): degradation state at time t
      Λ(t; γ): nonlinear time transformation (power law: t^γ)
      B(t): standard Brownian motion
      η: drift coefficient
      σ: diffusion coefficient
    """
    
    def __init__(self, failure_threshold: float = 1.0):
        self.failure_threshold = failure_threshold
        self.eta = None  # Drift coefficient
        self.sigma = None  # Diffusion coefficient
        self.gamma = None  # Nonlinear parameter
        self.fusion_weights = None  # Multi-sensor fusion weights
    
    def _time_transform(self, t: np.ndarray, gamma: float) -> np.ndarray:
        """Power-law time transformation."""
        return t ** gamma
    
    def _degradation_path(self, t: np.ndarray, eta: float, 
                          gamma: float) -> np.ndarray:
        """Expected degradation path."""
        return eta * self._time_transform(t, gamma)
    
    def _log_likelihood(self, params: np.ndarray, times: np.ndarray,
                        degradations: np.ndarray) -> float:
        """Negative log-likelihood for parameter estimation."""
        eta, sigma, gamma = float(params[0]), float(params[1]), float(params[2])
        
        if eta <= 0 or sigma <= 0 or gamma <= 0:
            return 1e10
        
        t = np.asarray(times, dtype=np.float64)
        d = np.asarray(degradations, dtype=np.float64)
        
        # Time differences
        dt = np.diff(t)
        dlambda = np.diff(t ** gamma)
        dx = np.diff(d)
        
        n = len(dt)
        ll = -0.5 * n * np.log(2 * np.pi * sigma ** 2)
        ll -= 0.5 * np.sum(np.log(np.maximum(dt, 1e-10)))
        ll -= np.sum((dx - eta * dlambda) ** 2) / (2.0 * sigma ** 2)
        
        return float(-ll)
    
    def fit(self, times: np.ndarray, degradations: np.ndarray):
        """
        Fit Wiener process parameters via MLE.
        
        Args:
            times: Time points
            degradations: Degradation measurements (1D array or fused HI)
        """
        if degradations.ndim > 1:
            # Multi-sensor: fuse using equal weights initially
            degradations = np.mean(degradations, axis=1)
        
        # Initial guess
        eta_init = degradations[-1] / (times[-1] ** 1.0 + 1e-8)
        sigma_init = np.std(np.diff(degradations))
        
        # Optimize
        result = minimize(
            self._log_likelihood,
            x0=[eta_init, sigma_init, 1.0],
            args=(times, degradations),
            bounds=[(1e-8, None), (1e-8, None), (0.1, 3.0)],
            method='L-BFGS-B',
        )
        
        self.eta, self.sigma, self.gamma = result.x
        return self
    
    def predict_rul(self, current_time: float, 
                    current_degradation: float) -> Tuple[float, float]:
        """
        Predict RUL at current time.
        
        Args:
            current_time: Current operation time
            current_degradation: Current degradation state
        
        Returns:
            rul_mean: Expected RUL
            rul_std: RUL standard deviation
        """
        if self.eta is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Remaining degradation to reach threshold
        remaining = self.failure_threshold - current_degradation
        if remaining <= 0:
            return 0.0, 0.0
        
        # First passage time distribution
        # For nonlinear Wiener process:
        # RUL ~ IG(remaining/η, remaining²/σ²) approximately
        # Mean ≈ remaining / η (simplified)
        
        # More accurate: solve for time when E[X(t)] reaches threshold
        # η · t^γ = remaining  →  t = (remaining/η)^(1/γ)
        rul_mean = (remaining / (self.eta + 1e-8)) ** (1.0 / self.gamma)
        
        # Variance approximation
        rul_var = (remaining * self.sigma ** 2) / (self.eta ** 3 + 1e-8)
        rul_std = max(0, np.sqrt(rul_var))
        
        return max(0, rul_mean - current_time), rul_std
    
    def predict_rul_distribution(self, current_time: float,
                                  current_degradation: float,
                                  n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict RUL probability distribution.
        
        Returns:
            t_grid: Time grid
            pdf: Probability density function values
        """
        rul_mean, rul_std = self.predict_rul(current_time, current_degradation)
        
        if rul_mean <= 0:
            return np.array([0]), np.array([1.0])
        
        t_grid = np.linspace(max(0, rul_mean - 3 * rul_std),
                             rul_mean + 3 * rul_std, n_points)
        pdf = norm.pdf(t_grid, loc=rul_mean, scale=max(rul_std, 1e-6))
        
        return t_grid, pdf


class MultiSensorWienerPredictor(WienerRULPredictor):
    """
    Extended Wiener predictor with multi-sensor fusion.
    
    Uses sparse Bayesian learning to find optimal fusion weights
    for combining multiple sensor signals into a Health Indicator.
    """
    
    def __init__(self, failure_threshold: float = 1.0):
        super().__init__(failure_threshold)
        self.sensor_weights = None
    
    def _fuse_sensors(self, sensor_data: np.ndarray) -> np.ndarray:
        """
        Fuse multiple sensor signals into a single HI.
        
        HI(t) = Σ w_j · S_j(t)
        """
        if self.sensor_weights is None:
            # Equal weights as default
            n_sensors = sensor_data.shape[1] if sensor_data.ndim > 1 else 1
            self.sensor_weights = np.ones(n_sensors) / n_sensors
        
        if sensor_data.ndim == 1:
            return sensor_data
        
        return np.dot(sensor_data, self.sensor_weights)
    
    def _optimize_weights(self, sensor_data: np.ndarray,
                          times: np.ndarray) -> np.ndarray:
        """
        Optimize sensor fusion weights using ARD sparse prior.
        (Simplified version for baseline)
        """
        n_sensors = sensor_data.shape[1]
        
        def objective(weights):
            weights = np.abs(weights)
            weights = weights / (weights.sum() + 1e-8)
            hi = np.dot(sensor_data, weights)
            
            # Fit Wiener on fused HI
            self.fit(times, hi)
            
            # Evaluate monotonicity (higher is better)
            monotonicity = np.mean(np.diff(hi) > 0)
            
            # Evaluate trendability (correlation with time)
            trendability = np.abs(np.corrcoef(times, hi)[0, 1])
            
            return -(monotonicity + trendability)  # Minimize negative
        
        result = minimize(
            objective,
            x0=np.ones(n_sensors) / n_sensors,
            bounds=[(0, 1)] * n_sensors,
            method='L-BFGS-B',
        )
        
        weights = np.abs(result.x)
        return weights / (weights.sum() + 1e-8)
    
    def fit_with_fusion(self, times: np.ndarray, sensor_data: np.ndarray):
        """Fit with automatic sensor weight optimization."""
        if sensor_data.ndim == 1:
            return self.fit(times, sensor_data)
        
        self.sensor_weights = self._optimize_weights(sensor_data, times)
        hi = np.dot(sensor_data, self.sensor_weights)
        return self.fit(times, hi)
