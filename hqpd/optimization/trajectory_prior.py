"""
Universal Trajectory Prior for Diffusion

This module implements a meta-predictor that learns universal anime denoising
patterns. Given (z_t, timestep), it predicts the noise/delta, allowing the
UNet to act as a "corrector" for faster inference.

Key insight: The predictor sees the actual latent state (not just embedding),
making prediction much more tractable than H10's blind approach.

Components:
- TrajectoryPredictor: Lightweight ConvNet (~5-10M params)
- TrajectoryDataset: Loads (z_t, t, noise_pred) training pairs
- TrajectoryTrainer: Training loop with MSE + cosine loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import math


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding (same as diffusion models use)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) tensor of timestep values
        Returns:
            embeddings: (B, dim) sinusoidal embeddings
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with timestep conditioning."""
    
    def __init__(self, channels: int, t_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.t_proj = nn.Linear(t_emb_dim, channels)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add timestep conditioning
        t = self.t_proj(t_emb)[:, :, None, None]
        h = h + t
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return x + h


class DownBlock(nn.Module):
    """Downsample block."""
    
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.res = ResBlock(out_ch, t_emb_dim)
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.res(x, t_emb)
        return x


class UpBlock(nn.Module):
    """Upsample block."""
    
    def __init__(self, in_ch: int, out_ch: int, t_emb_dim: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.res = ResBlock(out_ch, t_emb_dim)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x)
        x = x + skip  # Skip connection
        x = self.res(x, t_emb)
        return x


class TrajectoryPredictor(nn.Module):
    """
    Lightweight predictor for denoising trajectory.
    
    Given current latent z_t and timestep t, predicts the noise.
    Much smaller than UNet but learns domain-specific priors.
    
    Architecture:
    - Input: (B, 4, 64, 64) latent + timestep
    - Encoder: 64 -> 128 -> 256 (downsample to 16x16)
    - Bottleneck: 256 channels at 16x16
    - Decoder: 256 -> 128 -> 64 (upsample back to 64x64)
    - Output: (B, 4, 64, 64) predicted noise
    
    Parameters: ~5M
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        t_emb_dim: int = 256,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Timestep embedding
        self.t_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )
        
        # Input projection
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder (downsample)
        self.down1 = DownBlock(base_channels, base_channels * 2, t_emb_dim)      # 64 -> 32
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, t_emb_dim)  # 32 -> 16
        
        # Bottleneck
        self.mid1 = ResBlock(base_channels * 4, t_emb_dim)
        self.mid2 = ResBlock(base_channels * 4, t_emb_dim)
        
        # Decoder (upsample)
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, t_emb_dim)  # 16 -> 32
        self.up2 = UpBlock(base_channels * 2, base_channels, t_emb_dim)      # 32 -> 64
        
        # Output projection
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to produce near-zero output initially."""
        # Zero-init output layer for stable start
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)
    
    def forward(self, z_t: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Predict noise from current latent and timestep.
        
        Args:
            z_t: (B, 4, 64, 64) current noisy latent
            timestep: (B,) timestep values
            
        Returns:
            noise_pred: (B, 4, 64, 64) predicted noise
        """
        # Timestep embedding
        t_emb = self.t_embed(timestep)
        
        # Input
        h = self.in_conv(z_t)
        
        # Encoder with skip connections
        h1 = h  # Skip at 64x64
        h = self.down1(h, t_emb)
        h2 = h  # Skip at 32x32
        h = self.down2(h, t_emb)
        
        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        
        # Decoder with skip connections
        h = self.up1(h, h2, t_emb)
        h = self.up2(h, h1, t_emb)
        
        # Output
        noise_pred = self.out_conv(h)
        
        return noise_pred
    
    def save(self, path: str):
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "in_channels": self.in_channels,
                "base_channels": self.base_channels,
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TrajectoryPredictor":
        """Load model from checkpoint."""
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model


class TrajectoryDataset(torch.utils.data.Dataset):
    """
    Dataset for training TrajectoryPredictor.
    
    Each sample: (z_t, timestep, noise_pred)
    - z_t: Current latent state
    - timestep: Current timestep value
    - noise_pred: UNet's noise prediction (ground truth)
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = sorted(self.data_dir.glob("*.pt"))
        
        if len(self.samples) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = torch.load(self.samples[idx], map_location="cpu")
        return (
            sample["z_t"].float(),           # (4, 64, 64)
            sample["timestep"].float(),       # scalar
            sample["noise_pred"].float(),     # (4, 64, 64)
        )


class TrajectoryTrainer:
    """
    Trainer for TrajectoryPredictor.
    
    Uses MSE loss + optional cosine similarity loss for better gradient flow.
    """
    
    def __init__(
        self,
        model: TrajectoryPredictor,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    def train_step(
        self,
        z_t: torch.Tensor,
        timestep: torch.Tensor, 
        noise_target: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        z_t = z_t.to(self.device).float()
        timestep = timestep.to(self.device).float()
        noise_target = noise_target.to(self.device).float()
        
        # Forward
        noise_pred = self.model(z_t, timestep)
        
        # MSE loss
        loss_mse = F.mse_loss(noise_pred, noise_target)
        
        # Cosine similarity loss (helps with direction)
        B = noise_pred.shape[0]
        pred_flat = noise_pred.view(B, -1)
        target_flat = noise_target.view(B, -1)
        cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
        loss_cos = 1.0 - cos_sim
        
        loss = loss_mse + 0.1 * loss_cos
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "loss_mse": loss_mse.item(),
            "loss_cos": loss_cos.item(),
            "cos_sim": cos_sim.item(),
        }
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        totals = {"loss": 0, "loss_mse": 0, "cos_sim": 0}
        num_batches = 0
        
        for z_t, timestep, noise_target in dataloader:
            metrics = self.train_step(z_t, timestep, noise_target)
            for k in totals:
                totals[k] += metrics.get(k, 0)
            num_batches += 1
            
            if verbose and num_batches % 50 == 0:
                print(f"  Batch {num_batches}: loss={metrics['loss']:.4f}, cos={metrics['cos_sim']:.3f}")
        
        return {f"avg_{k}": v / max(1, num_batches) for k, v in totals.items()}


def predictor_assisted_denoise(
    predictor: TrajectoryPredictor,
    unet,
    scheduler,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    added_cond_kwargs: dict,
    predictor_steps: List[int] = [0, 2],  # Which steps use predictor only
    blend_weight: float = 0.0,  # 0 = predictor only for predictor_steps, 1 = always UNet
) -> torch.Tensor:
    """
    Hybrid inference: predictor for some steps, UNet for others.
    
    Args:
        predictor: Trained TrajectoryPredictor
        unet: Full UNet model
        scheduler: Scheduler
        latents: Initial latent
        timesteps: Timestep schedule
        prompt_embeds: Encoded prompt
        added_cond_kwargs: Additional conditioning
        predictor_steps: Which step indices use predictor-only
        blend_weight: 0 = pure predictor for predictor_steps, 1 = always blend
    
    Returns:
        Final denoised latent
    """
    device = latents.device
    
    for i, t in enumerate(timesteps):
        t_tensor = torch.tensor([t], device=device)
        latent_model_input = scheduler.scale_model_input(latents, t)
        
        if i in predictor_steps and blend_weight < 1.0:
            # Use predictor (fast)
            with torch.no_grad():
                noise_pred = predictor(latent_model_input, t_tensor)
                
                if blend_weight > 0:
                    # Blend with UNet
                    unet_pred = unet(
                        latent_model_input, t,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = (1 - blend_weight) * noise_pred + blend_weight * unet_pred
        else:
            # Use UNet (accurate)
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
        
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    return latents
