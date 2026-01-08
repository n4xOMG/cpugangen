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
        
        # Normalization statistics (will be computed during training)
        # These handle the large variance in latent deltas
        self.register_buffer('delta_mean', torch.zeros(1, latent_channels, 1, 1))
        self.register_buffer('delta_std', torch.ones(1, latent_channels, 1, 1))
        
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
        
        # Initialize with scale-aware weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with reasonable scale for delta prediction."""
        for module in self.structure_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Last layer: scale to produce reasonable initial outputs (~0 mean, moderate std)
        last_linear = self.structure_head[-1]
        nn.init.normal_(last_linear.weight, std=0.1)  # Increased from 0.001
        nn.init.zeros_(last_linear.bias)
        
        # Refine layers: small residual
        for module in self.refine.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, pooled_embed: torch.Tensor, use_normalization: bool = True) -> torch.Tensor:
        """
        Predict structure from pooled embedding.
        
        Args:
            pooled_embed: (B, embed_dim) - CLIP pooled or context mean
            use_normalization: Whether to denormalize output (True during inference)
            
        Returns:
            structure: (B, 4, latent_size, latent_size)
        """
        B = pooled_embed.shape[0]
        
        # Project embedding
        h = self.embed_proj(pooled_embed)  # (B, hidden_dim)
        
        # Predict low-res structure (normalized space)
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
        
        # Denormalize to original delta scale
        if use_normalization and self.delta_std.sum() > 0:
            structure = structure * self.delta_std + self.delta_mean
        
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
    delta_scale: float = 1.0,  # How much of predicted delta to apply (1.0 = full)
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate smart initial latent by predicting what step 1 would do.
    
    The model predicts the DELTA that step 1 applies to random noise.
    We apply this delta to new random noise to get a "head start".
    
    Args:
        initializer: Trained StructuredInitializer model (predicts delta)
        pooled_embed: (1, embed_dim) CLIP pooled embedding
        seed: Random seed for noise generation
        init_noise_sigma: Scheduler's initial noise sigma
        delta_scale: How much of the predicted delta to apply (0-1)
        device: Device to use
        
    Returns:
        init_latent: (1, 4, H, W) initialized latent ready for step 2+
    """
    with torch.no_grad():
        # Generate random noise (same as normal init)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            (1, 4, 64, 64),
            generator=generator,
            dtype=torch.float32,
        ).to(device)
        
        # Scale noise by init sigma (standard initialization)
        initial_latent = noise * init_noise_sigma
        
        # Predict what step 1 would do to this noise
        predicted_delta = initializer(pooled_embed.to(device).float())
        
        # Apply the delta: simulate having run step 1
        smart_latent = initial_latent + delta_scale * predicted_delta
        
    return smart_latent




class StructuredInitDataset(torch.utils.data.Dataset):
    """
    Dataset for training StructuredInitializer.
    
    Each sample: (pooled_embedding, initial_latent, target_latent)
    - pooled_embedding: CLIP pooled embedding of the prompt
    - initial_latent: Random noise latent (scaled)
    - target_latent: Latent after step 1 of denoising
    
    We train to predict: delta = target_latent - initial_latent
    This delta represents "what step 1 did" to the noise.
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = list(self.data_dir.glob("*.pt"))
        
        if len(self.samples) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = torch.load(self.samples[idx], map_location="cpu")
        # Handle both old format (no initial_latent) and new format
        initial_latent = sample.get("initial_latent", torch.zeros_like(sample["target_latent"]))
        return sample["pooled_embed"], initial_latent, sample["target_latent"]


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
        initial_latents: torch.Tensor,
        target_latents: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        We train to predict the DELTA: target_latent - initial_latent
        This represents what denoising step 1 does to the noise.
        """
        self.model.train()
        
        # Convert to float32 in case data was collected with fp16
        pooled_embeds = pooled_embeds.to(self.device).float()
        initial_latents = initial_latents.to(self.device).float()
        target_latents = target_latents.to(self.device).float()
        
        # Compute the actual delta (what step 1 did)
        target_delta = target_latents - initial_latents
        
        # Normalize target delta for stable training
        # Compute per-channel mean and std
        delta_mean = target_delta.mean(dim=(0, 2, 3), keepdim=True)
        delta_std = target_delta.std(dim=(0, 2, 3), keepdim=True) + 1e-6
        target_delta_norm = (target_delta - delta_mean) / delta_std
        
        # Update model's normalization stats (EMA)
        with torch.no_grad():
            momentum = 0.1
            self.model.delta_mean = (1 - momentum) * self.model.delta_mean + momentum * delta_mean.mean(dim=0)
            self.model.delta_std = (1 - momentum) * self.model.delta_std + momentum * delta_std.mean(dim=0)
        
        # Forward pass - predict the delta (normalized)
        predicted_delta_norm = self.model(pooled_embeds, use_normalization=False)
        
        # Loss on normalized delta
        loss_delta = F.mse_loss(predicted_delta_norm, target_delta_norm)
        
        # Optional: low-frequency delta loss for stability
        target_delta_lowfreq = gaussian_blur_2d(target_delta_norm, self.blur_sigma)
        pred_delta_lowfreq = gaussian_blur_2d(predicted_delta_norm, self.blur_sigma)
        loss_lowfreq = F.mse_loss(pred_delta_lowfreq, target_delta_lowfreq)
        
        loss = loss_delta + 0.5 * loss_lowfreq
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "loss_delta": loss_delta.item(),
            "loss_lowfreq": loss_lowfreq.item(),
            "pred_std": predicted_delta_norm.std().item(),
            "target_std": target_delta_norm.std().item(),
            "grad_norm": grad_norm.item(),
        }
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        total_delta = 0.0
        total_pred_std = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        
        for batch in dataloader:
            # Handle both old (2-value) and new (3-value) format
            if len(batch) == 3:
                pooled_embeds, initial_latents, target_latents = batch
            else:
                pooled_embeds, target_latents = batch
                initial_latents = torch.zeros_like(target_latents)
            
            metrics = self.train_step(pooled_embeds, initial_latents, target_latents)
            total_loss += metrics["loss"]
            total_delta += metrics["loss_delta"]
            total_pred_std += metrics.get("pred_std", 0)
            total_grad_norm += metrics.get("grad_norm", 0)
            num_batches += 1
            
            if verbose and num_batches % 10 == 0:
                print(f"  Batch {num_batches}: loss={metrics['loss']:.4f}")
        
        return {
            "avg_loss": total_loss / max(1, num_batches),
            "avg_loss_delta": total_delta / max(1, num_batches),
            "avg_pred_std": total_pred_std / max(1, num_batches),
            "avg_grad_norm": total_grad_norm / max(1, num_batches),
        }

