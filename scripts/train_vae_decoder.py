#!/usr/bin/env python3
"""
Train TinyVAE Decoder on cached latents.

Enhanced with:
- Pretrained initialization from SD 1.5
- LPIPS perceptual loss (CRITICAL for quality!)
- Validation image generation
- Gradient checkpointing
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import wandb

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.tiny_vae import TinyVAEDecoder, count_parameters
from hqpd.models.vae_enhancements import (
    EnhancedVAELoss,
    load_pretrained_illustrious_decoder,
    initialize_from_pretrained
)
from hqpd.models.vae_validation import (
    save_validation_images,
    compute_reconstruction_metrics
)


class CachedLatentDataset(Dataset):
    """Dataset for loading pre-cached latents and target images."""
    
    def __init__(
        self,
        latent_metadata_path: Path,
        latents_dir: Path,
        images_dir: Path,
        masks_dir: Optional[Path] = None,
        resolution: int = 1024
    ):
        with open(latent_metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.latents_dir = latents_dir
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.resolution = resolution
        
        # Image transform (to tensor and normalize)
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)  # [0, 1] -> [-1, 1]
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load cached latent
        latent_filename = item['latent_filename']
        latent_path = self.latents_dir / latent_filename
        latent = torch.load(latent_path)
        
        # Load target image
        image_filename = item.get('preprocessed_filename', item.get('filename'))
        image_path = self.images_dir / image_filename
        image = Image.open(image_path).convert('RGB')
        target = self.transform(image)
        
        # Load mask if available (for padded images)
        mask = None
        if self.masks_dir and item.get('mask_filename'):
            mask_path = self.masks_dir / item['mask_filename']
            if mask_path.exists():
                mask = torch.load(mask_path)
        
        result = {
            'latent': latent,
            'target': target,
            'filename': image_filename
        }
        
        if mask is not None:
            result['mask'] = mask
        
        return result


# VAEDecoderLoss is now imported from vae_enhancements.py as EnhancedVAELoss
# Old basic loss removed - using enhanced version with LPIPS


def train_one_epoch(
    model: TinyVAEDecoder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: EnhancedVAELoss,
    device: torch.device,
    epoch: int,
    config: Dict
) -> float:
    """Train for one epoch."""
    
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        latents = batch['latent'].to(device)
        targets = batch['target'].to(device)
        masks = batch.get('mask', None)
        if masks is not None:
            masks = masks.to(device)
        
        # Forward pass with mixed precision
        with autocast():
            # Unscale latents (they're scaled in cache)
            latents_unscaled = latents / 0.13025
            
            outputs = model(latents_unscaled)
            
            # Compute loss
            losses = loss_fn(outputs, targets, masks)
            loss = losses['total']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config['training']['max_grad_norm'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['max_grad_norm']
            )
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Logging
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'l1': f"{losses['l1'].item():.4f}",
            'l2': f"{losses['l2'].item():.4f}",
            'lpips': f"{losses.get('lpips', 0).item():.4f}"
        })
        
        # Log to wandb
        if step % config['logging']['log_every'] == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/l1_loss': losses['l1'].item(),
                'train/l2_loss': losses['l2'].item(),
                'train/lpips_loss': losses.get('lpips', torch.tensor(0.0)).item(),
                'train/epoch': epoch,
                'train/step': step
            })
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: TinyVAEDecoder,
    dataloader: DataLoader,
    loss_fn: EnhancedVAELoss,
    device: torch.device,
    epoch: int
) -> float:
    """Validate the model."""
    
    model.eval()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Validating"):
        latents = batch['latent'].to(device)
        targets = batch['target'].to(device)
        masks = batch.get('mask', None)
        if masks is not None:
            masks = masks.to(device)
        
        # Unscale latents
        latents_unscaled = latents / 0.13025
        
        outputs = model(latents_unscaled)
        losses = loss_fn(outputs, targets, masks)
        
        total_loss += losses['total'].item()
    
    avg_loss = total_loss / len(dataloader)
    
    wandb.log({
        'val/loss': avg_loss,
        'val/epoch': epoch
    })
    
    return avg_loss


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['wandb_project'],
            name=config['logging']['run_name'],
            config=config
        )
    
    print("\n" + "="*60)
    print("TinyVAE Decoder Training")
    print("="*60)
    
    # Create model
    model = TinyVAEDecoder(
        latent_channels=4,
        base_channels=config['model']['base_channels'],
        max_channels=config['model']['max_channels'],
        num_upsample_blocks=config['model']['num_upsample_blocks']
    ).to(device)
    
    # CRITICAL: Initialize from pretrained Illustrious VAE (NOT random!)
    if config['model'].get('use_pretrained_init', True):
        print("\n⚠️  IMPORTANT: Initializing from pretrained Illustrious VAE...")
        print("   (Matches your teacher model: martineux/janku6)")
        pretrained_decoder = load_pretrained_illustrious_decoder(device)
        model = initialize_from_pretrained(
            model,
            pretrained_decoder,
            strategy='copy_matching'
        )
        del pretrained_decoder
        torch.cuda.empty_cache()
    else:
        print("\n⚠️  WARNING: Using random initialization (not recommended!)")
    
    # Enable gradient checkpointing if requested
    if config['model'].get('use_gradient_checkpointing', False):
        print("✅ Gradient checkpointing enabled")
        # Note: TinyVAEDecoder would need checkpointing support added
    
    total_params = count_parameters(model)
    print(f"\nModel: {total_params:,} parameters")
    print(f"Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # Load datasets
    print("\n" + "="*60)
    print("Loading Datasets")
    print("="*60)
    
    train_dataset = CachedLatentDataset(
        latent_metadata_path=Path(config['data']['train_latent_metadata']),
        latents_dir=Path(config['data']['train_latents_dir']),
        images_dir=Path(config['data']['train_images_dir']),
        masks_dir=Path(config['data'].get('train_masks_dir')),
        resolution=config['data']['resolution']
    )
    
    val_dataset = CachedLatentDataset(
        latent_metadata_path=Path(config['data']['val_latent_metadata']),
        latents_dir=Path(config['data']['val_latents_dir']),
        images_dir=Path(config['data']['val_images_dir']),
        masks_dir=Path(config['data'].get('val_masks_dir')),
        resolution=config['data']['resolution']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Setup training with enhanced loss (includes LPIPS!)
    loss_fn = EnhancedVAELoss(
        l1_weight=config['loss']['l1_weight'],
        l2_weight=config['loss']['l2_weight'],
        lpips_weight=config['loss'].get('lpips_weight', 0.1),
        device=device
    )
    
    print("\n✅ Using EnhancedVAELoss with LPIPS perceptual loss")
    print(f"   L1 weight: {config['loss']['l1_weight']}")
    print(f"   L2 weight: {config['loss']['l2_weight']}")
    print(f"   LPIPS weight: {config['loss'].get('lpips_weight', 0.1)}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    best_val_loss = float('inf')
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create validation images directory
    val_images_dir = output_dir / "validation_images"
    val_images_dir.mkdir(exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            loss_fn, device, epoch, config
        )
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(
            model, val_loader, loss_fn, device, epoch
        )
        print(f"Val Loss: {val_loss:.4f}")
        
        # CRITICAL: Save validation images to monitor quality!
        if (epoch + 1) % config['logging'].get('save_images_every', 1) == 0:
            save_validation_images(
                model, val_loader, epoch,
                val_images_dir,
                num_samples=8,
                device=device
            )
        
        # Compute reconstruction metrics
        if (epoch + 1) % config['logging'].get('compute_metrics_every', 5) == 0:
            metrics = compute_reconstruction_metrics(
                model, val_loader, device, num_batches=10
            )
            print(f"  Reconstruction MSE: {metrics['mse']:.6f}")
            print(f"  Reconstruction PSNR: {metrics['psnr']:.2f} dB")
            
            if config['logging']['use_wandb']:
                wandb.log({
                    'val/mse': metrics['mse'],
                    'val/psnr': metrics['psnr'],
                    'val/epoch': epoch
                })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = output_dir / "best_tiny_vae_decoder.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"✅ Saved best model! Val loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = output_dir / f"tiny_vae_decoder_epoch{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    if config['logging']['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TinyVAE Decoder")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/vae_decoder_config.json',
        help='Path to config file'
    )
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Test mode (short run for validation)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        help='Max steps for test mode'
    )
    
    args = parser.parse_args()
    main(args)
