"""
Structured Initialization for Diffusion

This module implements learned structured initialization to accelerate
diffusion by predicting a "smart starting point" from prompt embeddings.

Key Concept:
- Learn low-frequency structure from prompts
- Combine with random high-frequency noise for variation
- Start closer to target → fewer steps needed

Components:
- StructuredInitializer: Lightweight network (~1MB)
- smart_latent_init(): Inference function
- Training utilities for offline learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import json


class StructuredInitializer(nn.Module):
    """
    Learns to predict initial latent structure from prompt embeddings.
    
    Architecture:
    - Input: CLIP pooled embedding (1280-dim) or context mean
    - Hidden: MLP with residual connections
    - Output: Low-res structure → upsampled to latent size
    
    Parameters: ~1.5M (~6MB fp32, ~3MB fp16)
    """
    
    def __init__(
        self,
        embed_dim: int = 1280,  # CLIP pooled dim
        hidden_dim: int = 512,
        latent_channels: int = 4,
        latent_size: int = 64,  # 64x64 for 512x512 images
        low_res: int = 8,  # Predict at 8x8, upsample
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.low_res = low_res
        
        # Project embedding to hidden
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Predict low-res structure
        self.structure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_channels * low_res * low_res),
        )
        
        # Optional: learnable upsampling refinement
        self.refine = nn.Sequential(
            nn.Conv2d(latent_channels, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, latent_channels, 3, padding=1),
        )
        
        # Initialize output layers to near-zero for safe starting point
        self._init_weights()
    
    def _init_weights(self):
        """Initialize to produce near-zero output initially."""
        for module in self.structure_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Last layer very small
        last_linear = self.structure_head[-1]
        nn.init.normal_(last_linear.weight, std=0.001)
        nn.init.zeros_(last_linear.bias)
        
        # Refine layers also small
        for module in self.refine.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, pooled_embed: torch.Tensor) -> torch.Tensor:
        """
        Predict structure from pooled embedding.
        
        Args:
            pooled_embed: (B, embed_dim) - CLIP pooled or context mean
            
        Returns:
            structure: (B, 4, latent_size, latent_size)
        """
        B = pooled_embed.shape[0]
        
        # Project embedding
        h = self.embed_proj(pooled_embed)  # (B, hidden_dim)
        
        # Predict low-res structure
        low_res_flat = self.structure_head(h)  # (B, C*H*W)
        low_res = low_res_flat.view(B, self.latent_channels, self.low_res, self.low_res)
        
        # Upsample to full latent size
        structure = F.interpolate(
            low_res,
            size=(self.latent_size, self.latent_size),
            mode='bilinear',
            align_corners=False,
        )
        
        # Optional refinement
        structure = structure + self.refine(structure) * 0.1
        
        return structure
    
    def save(self, path: str):
        """Save model and config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "embed_dim": self.embed_proj[0].in_features,
            "hidden_dim": self.embed_proj[0].out_features,
            "latent_channels": self.latent_channels,
            "latent_size": self.latent_size,
            "low_res": self.low_res,
        }
        
        torch.save({
            "state_dict": self.state_dict(),
            "config": config,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "StructuredInitializer":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint["config"]
        
        model = cls(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        
        return model


def gaussian_blur_2d(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Apply Gaussian blur to extract low-frequency components."""
    # Create Gaussian kernel
    kernel_size = int(6 * sigma) | 1  # Ensure odd
    kernel_size = max(3, kernel_size)
    
    # Create 1D Gaussian
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords = coords - kernel_size // 2
    kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Create 2D kernel via outer product
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.expand(x.shape[1], 1, -1, -1)
    
    # Apply blur
    padding = kernel_size // 2
    blurred = F.conv2d(x, kernel_2d, padding=padding, groups=x.shape[1])
    
    return blurred


def smart_latent_init(
    initializer: StructuredInitializer,
    pooled_embed: torch.Tensor,
    seed: int,
    init_noise_sigma: float = 1.0,
    variation_scale: float = 0.7,
    structure_scale: float = 0.3,
    blur_sigma: float = 4.0,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate smart initial latent combining learned structure with random variation.
    
    Args:
        initializer: Trained StructuredInitializer model
        pooled_embed: (1, embed_dim) CLIP pooled embedding
        seed: Random seed for variation
        init_noise_sigma: Scheduler's initial noise sigma
        variation_scale: How much random variation to add (0-1)
        structure_scale: How much learned structure to use (0-1)
        blur_sigma: Sigma for frequency separation blur
        device: Device to use
        
    Returns:
        init_latent: (1, 4, H, W) initialized latent
    """
    with torch.no_grad():
        # Get predicted structure
        structure = initializer(pooled_embed.to(device))  # (1, 4, 64, 64)
        
        # Extract low-frequency component
        structure_lowfreq = gaussian_blur_2d(structure, sigma=blur_sigma)
        
        # Generate random noise with seed
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            structure.shape,
            generator=generator,
            dtype=structure.dtype,
        ).to(device)
        
        # Extract high-frequency component of noise
        noise_lowfreq = gaussian_blur_2d(noise, sigma=blur_sigma)
        noise_highfreq = noise - noise_lowfreq
        
        # Combine: learned structure (low-freq) + random variation (high-freq)
        init_latent = (
            structure_scale * structure_lowfreq +
            variation_scale * noise_highfreq +
            (1.0 - structure_scale - variation_scale) * noise
        )
        
        # Scale by scheduler's init sigma
        init_latent = init_latent * init_noise_sigma
        
    return init_latent


class StructuredInitDataset(torch.utils.data.Dataset):
    """
    Dataset for training StructuredInitializer.
    
    Each sample: (pooled_embedding, target_latent)
    - pooled_embedding: CLIP pooled embedding of the prompt
    - target_latent: Final denoised latent from the pipeline
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = list(self.data_dir.glob("*.pt"))
        
        if len(self.samples) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.load(self.samples[idx], map_location="cpu")
        return sample["pooled_embed"], sample["target_latent"]


class StructuredInitTrainer:
    """
    Trainer for StructuredInitializer.
    
    Uses low-frequency reconstruction loss to train the structure predictor
    while ignoring high-frequency details (handled by random noise).
    """
    
    def __init__(
        self,
        model: StructuredInitializer,
        lr: float = 1e-4,
        blur_sigma: float = 4.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.blur_sigma = blur_sigma
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    def train_step(
        self,
        pooled_embeds: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        pooled_embeds = pooled_embeds.to(self.device)
        target_latents = target_latents.to(self.device)
        
        # Forward pass
        predicted = self.model(pooled_embeds)
        
        # Low-frequency reconstruction loss
        target_lowfreq = gaussian_blur_2d(target_latents, self.blur_sigma)
        pred_lowfreq = gaussian_blur_2d(predicted, self.blur_sigma)
        
        loss_lowfreq = F.mse_loss(pred_lowfreq, target_lowfreq)
        
        # Optional: small penalty on full prediction to avoid divergence
        loss_reg = F.mse_loss(predicted, target_latents) * 0.1
        
        loss = loss_lowfreq + loss_reg
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "loss_lowfreq": loss_lowfreq.item(),
            "loss_reg": loss_reg.item(),
        }
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        total_lowfreq = 0.0
        num_batches = 0
        
        for pooled_embeds, target_latents in dataloader:
            metrics = self.train_step(pooled_embeds, target_latents)
            total_loss += metrics["loss"]
            total_lowfreq += metrics["loss_lowfreq"]
            num_batches += 1
            
            if verbose and num_batches % 10 == 0:
                print(f"  Batch {num_batches}: loss={metrics['loss']:.4f}")
        
        return {
            "avg_loss": total_loss / max(1, num_batches),
            "avg_loss_lowfreq": total_lowfreq / max(1, num_batches),
        }
