"""
Validation utilities for VAE training.

Saves reconstructed images during training to monitor quality.
"""

import torch
from pathlib import Path
from PIL import Image
import torchvision.utils as vutils
import numpy as np


@torch.no_grad()
def save_validation_images(
    model,
    val_loader,
    epoch: int,
    output_dir: Path,
    num_samples: int = 8,
    device='cuda'
):
    """
    Save validation reconstructions to monitor quality during training.
    
    CRITICAL: Visual inspection is the best way to catch quality degradation!
    
    Args:
        model: TinyVAE decoder model
        val_loader: Validation DataLoader
        epoch: Current epoch number
        output_dir: Directory to save images
        num_samples: Number of validation samples to save
        device: Device to use
    """
    model.eval()
    
    # Get a batch
    batch = next(iter(val_loader))
    latents = batch['latent'][:num_samples].to(device)
    targets = batch['target'][:num_samples].to(device)
    filenames = batch['filename'][:num_samples]
    
    # Decode
    latents_unscaled = latents / 0.13025
    reconstructed = model(latents_unscaled)
    
    # Create comparison grid
    # Format: [target, reconstructed] for each sample
    comparison = []
    for i in range(num_samples):
        comparison.append(targets[i])
        comparison.append(reconstructed[i])
    
    # Stack into grid
    grid = vutils.make_grid(
        comparison,
        nrow=2,  # 2 images per row (target, reconstruction)
        normalize=True,
        value_range=(-1, 1),
        padding=2
    )
    
    # Save grid
    grid_np = grid.cpu().permute(1, 2, 0).numpy()
    grid_np = (grid_np * 255).astype(np.uint8)
    grid_img = Image.fromarray(grid_np)
    
    output_path = output_dir / f"validation_epoch{epoch:03d}.png"
    grid_img.save(output_path)
    
    print(f"  💾 Saved validation images: {output_path}")
    
    # Also save individual pairs for detailed inspection
    pairs_dir = output_dir / f"epoch{epoch:03d}"
    pairs_dir.mkdir(exist_ok=True)
    
    for i in range(min(4, num_samples)):  # Save 4 individual pairs
        # Target
        target_np = ((targets[i].cpu() + 1) / 2 * 255).clamp(0, 255).byte()
        target_np = target_np.permute(1, 2, 0).numpy()
        target_img = Image.fromarray(target_np)
        target_img.save(pairs_dir / f"sample{i}_target.png")
        
        # Reconstruction
        recon_np = ((reconstructed[i].cpu() + 1) / 2 * 255).clamp(0, 255).byte()
        recon_np = recon_np.permute(1, 2, 0).numpy()
        recon_img = Image.fromarray(recon_np)
        recon_img.save(pairs_dir / f"sample{i}_reconstructed.png")
    
    model.train()
    
    return output_path


@torch.no_grad()
def compute_reconstruction_metrics(
    model,
    val_loader,
    device='cuda',
    num_batches: int = 10
):
    """
    Compute reconstruction quality metrics on validation set.
    
    Metrics:
    - MSE (Mean Squared Error)
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index) - optional
    
    Args:
        model: Decoder model
        val_loader: Validation dataloader
        device: Device
        num_batches: Number of batches to evaluate
        
    Returns:
        Dict with average metrics
    """
    model.eval()
    
    mse_total = 0.0
    psnr_total = 0.0
    count = 0
    
    for i, batch in enumerate(val_loader):
        if i >= num_batches:
            break
        
        latents = batch['latent'].to(device)
        targets = batch['target'].to(device)
        
        # Decode
        latents_unscaled = latents / 0.13025
        reconstructed = model(latents_unscaled)
        
        # Compute MSE
        mse = torch.mean((reconstructed - targets) ** 2, dim=[1, 2, 3])
        mse_total += mse.sum().item()
        
        # Compute PSNR
        # PSNR = 10 * log10(max^2 / MSE)
        # For images in [-1, 1], max = 2, so max^2 = 4
        psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
        psnr_total += psnr.sum().item()
        
        count += latents.size(0)
    
    metrics = {
        'mse': mse_total / count,
        'psnr': psnr_total / count
    }
    
    model.train()
    
    return metrics
