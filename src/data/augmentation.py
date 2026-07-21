"""
Data augmentation strategies for contrastive self-supervised learning
on degradation time series.

Based on:
- SHRDL (Wang et al., RESS 2023): Random Cropping + Timestamp Latent Masking
- PEUDA (Li et al., AEI 2024): Time-domain + Frequency-domain augmentations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class TimeAugmentation:
    """
    Time-domain augmentations for degradation signals.
    
    Strategies:
    1. Scaling: Multiply signal by random factor in [0.8, 1.2]
    2. Dithering: Add small Gaussian noise
    3. Warping: Local time warping via random shift
    4. Time-shifting: Circular shift along time axis
    5. Random Cropping: Extract random subsequence (SHRDL)
    """
    
    def __init__(self, scale_range=(0.8, 1.2), noise_std=0.01, 
                 crop_min=0.5, crop_max=0.8):
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.crop_min = crop_min
        self.crop_max = crop_max
    
    def scale(self, x: torch.Tensor) -> torch.Tensor:
        """Random scaling augmentation."""
        factor = torch.FloatTensor(1).uniform_(*self.scale_range).to(x.device)
        return x * factor
    
    def dither(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return x + noise
    
    def warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Local time warping: randomly shift a segment of the sequence.
        Input shape: (batch, seq_len, features) or (seq_len, features)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single = True
        else:
            single = False
        
        B, T, C = x.shape
        warped = x.clone()
        
        for b in range(B):
            # Pick a random segment to shift
            seg_len = np.random.randint(T // 8, T // 4)
            seg_start = np.random.randint(0, T - seg_len)
            shift = np.random.randint(-seg_len // 2, seg_len // 2 + 1)
            
            new_start = max(0, min(T - seg_len, seg_start + shift))
            seg = warped[b, seg_start:seg_start + seg_len].clone()
            warped[b, seg_start:seg_start + seg_len] = warped[b, new_start:new_start + seg_len]
            warped[b, new_start:new_start + seg_len] = seg
        
        if single:
            warped = warped.squeeze(0)
        return warped
    
    def random_crop(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SHRDL-style random cropping: generates two overlapping crops
        from the same original sequence.
        Returns (crop1, crop2) with overlapping region.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single = True
        else:
            single = False
        
        B, T, C = x.shape
        crop_len = int(T * np.random.uniform(self.crop_min, self.crop_max))
        
        crops = []
        for _ in range(2):
            max_start = T - crop_len
            start = np.random.randint(0, max_start + 1)
            crop = x[:, start:start + crop_len, :]
            # Resize back to original length via interpolation
            crop = F.interpolate(
                crop.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            crops.append(crop)
        
        if single:
            crops = [c.squeeze(0) for c in crops]
        return crops[0], crops[1]
    
    def shift(self, x: torch.Tensor) -> torch.Tensor:
        """Circular time shift."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
            single = True
        else:
            single = False
        
        B, T, C = x.shape
        shift_amount = np.random.randint(0, T)
        shifted = torch.roll(x, shifts=shift_amount, dims=1)
        
        if single:
            shifted = shifted.squeeze(0)
        return shifted
    
    def __call__(self, x: torch.Tensor, mode: str = 'random') -> torch.Tensor:
        """
        Apply a random time-domain augmentation.
        mode: 'random', 'scale', 'dither', 'warp', 'shift'
        """
        if mode == 'random':
            mode = np.random.choice(['scale', 'dither', 'warp', 'shift'])
        
        if mode == 'scale':
            return self.scale(x)
        elif mode == 'dither':
            return self.dither(x)
        elif mode == 'warp':
            return self.warp(x)
        elif mode == 'shift':
            return self.shift(x)
        else:
            return x


class FrequencyAugmentation:
    """
    Frequency-domain augmentations (PEUDA).
    
    Strategies:
    1. Add frequency: Inject random frequency components
    2. Remove frequency: Mask out random frequency bands
    """
    
    def __init__(self, n_fft: int = 64, mask_ratio: float = 0.1):
        self.n_fft = n_fft
        self.mask_ratio = mask_ratio
    
    def to_frequency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert time-domain signal to frequency domain via FFT.
        Input: (..., seq_len, features)
        """
        return torch.fft.rfft(x, n=self.n_fft, dim=-2)
    
    def to_time(self, x_fft: torch.Tensor, original_len: int) -> torch.Tensor:
        """Convert back to time domain."""
        x_time = torch.fft.irfft(x_fft, n=original_len, dim=-2)
        return x_time
    
    def add_frequency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add random frequency components.
        1. FFT -> 2. Inject random magnitudes at random freqs -> 3. IFFT
        """
        x_fft = self.to_frequency(x)
        # Random magnitudes at random frequency bins
        mask = torch.rand_like(x_fft.real) < 0.1
        random_magnitudes = torch.randn_like(x_fft.real) * 0.1
        x_fft = x_fft + mask * (random_magnitudes + 1j * torch.randn_like(x_fft.imag) * 0.1)
        return self.to_time(x_fft, x.shape[-2])
    
    def remove_frequency(self, x: torch.Tensor) -> torch.Tensor:
        """
        Remove random frequency bands.
        1. FFT -> 2. Zero out random frequency bins -> 3. IFFT
        """
        x_fft = self.to_frequency(x)
        mask = torch.rand_like(x_fft.real) > self.mask_ratio
        x_fft = x_fft * mask.float()
        return self.to_time(x_fft, x.shape[-2])
    
    def frequency_enhance(self, x: torch.Tensor) -> torch.Tensor:
        """
        PEUDA frequency enhancement: generates frequency-augmented version
        for the frequency-domain branch of the dual-stream encoder.
        """
        return self.add_frequency(x)
    
    def __call__(self, x: torch.Tensor, mode: str = 'random') -> torch.Tensor:
        if mode == 'random':
            mode = np.random.choice(['add', 'remove'])
        
        if mode == 'add':
            return self.add_frequency(x)
        elif mode == 'remove':
            return self.remove_frequency(x)
        else:
            return x


class TimestampLatentMask:
    """
    SHRDL Timestamp Latent Masking:
    Random binary mask applied to the latent representation.
    """
    
    def __init__(self, mask_ratio: float = 0.15):
        self.mask_ratio = mask_ratio
    
    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """Apply random mask to latent features."""
        mask = torch.bernoulli(torch.full_like(z, 1 - self.mask_ratio))
        return z * mask
