#!/usr/bin/env python3
"""
Cache latents for training by pre-encoding all preprocessed images.

Uses the frozen SDXL VAE encoder to encode images to (4, 128, 128) latent tensors.
Saves latents to disk for fast training without re-encoding.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
import numpy as np


class ImageDataset(Dataset):
    """Simple dataset for loading preprocessed images."""
    
    def __init__(self, metadata: List[Dict], images_dir: Path, resolution: int = 1024):
        self.metadata = metadata
        self.images_dir = images_dir
        self.resolution = resolution
        
        # Transform: resize, crop to exact size, to tensor and normalize
        # CenterCrop ensures ALL images are exactly resolution x resolution
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),  # Ensure exact square size
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        filename = item.get('preprocessed_filename', item.get('filename'))
        
        img_path = self.images_dir / filename
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.transform(image)
        
        return {
            'pixel_values': pixel_values,
            'filename': filename,
            'metadata': item  # Keep as single dict, not batched
        }


def custom_collate_fn(batch):
    """Custom collate that handles varying metadata structures."""
    # Stack only the pixel_values tensor
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Keep filenames and metadata as lists (don't try to collate dicts)
    filenames = [item['filename'] for item in batch]
    metadata_list = [item['metadata'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'filename': filenames,
        'metadata': metadata_list
    }


def cache_latents(
    metadata_path: Path,
    images_dir: Path,
    output_dir: Path,
    teacher_model: str,
    batch_size: int = 4,
    device: str = 'cuda',
    verify_samples: int = 0
):
    """
    Cache latents for all images in dataset.
    
    Args:
        metadata_path: Path to preprocessed metadata JSON
        images_dir: Directory with preprocessed images
        output_dir: Directory to save cached latents
        teacher_model: HuggingFace model ID for VAE (e.g., 'martineux/janku6')
        batch_size: Batch size for encoding
        device: Device to use ('cuda' or 'cpu')
        verify_samples: Number of samples to verify by decoding (0 = skip)
    """
    print("\n" + "="*60)
    print("Latent Caching for Training")
    print("="*60)
    print(f"Teacher model: {teacher_model}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = output_dir / "latents"
    latents_dir.mkdir(exist_ok=True)
    
    # Load metadata
    print(f"\nLoading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata)} images to process")
    
    # Load VAE
    print(f"\nLoading VAE from {teacher_model}...")
    vae = AutoencoderKL.from_pretrained(
        teacher_model,
        subfolder="vae",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    
    print(f"✅ VAE loaded: {sum(p.numel() for p in vae.parameters()):,} parameters")
    
    # Create dataset and loader with custom collate function
    dataset = ImageDataset(metadata, images_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == 'cuda'),
        collate_fn=custom_collate_fn  # Use custom collate to handle varying metadata
    )
    
    # Cache latents
    print("\n" + "="*60)
    print("Encoding Images to Latents")
    print("="*60)
    
    latent_metadata = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            pixel_values = batch['pixel_values'].to(device)
            filenames = batch['filename']
            batch_metadata = batch['metadata']  # Now a list of dicts
            
            # Encode to latents
            latent_dist = vae.encode(pixel_values).latent_dist
            latents = latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            
            # Save each latent
            for i, filename in enumerate(filenames):
                latent = latents[i].cpu()
                
                # Create latent filename (same as image but .pt extension)
                latent_filename = Path(filename).stem + '.pt'
                latent_path = latents_dir / latent_filename
                
                # Save latent tensor
                torch.save(latent, latent_path)
                
                # Get metadata for this item (already a dict)
                item_metadata = batch_metadata[i].copy()
                item_metadata['latent_filename'] = latent_filename
                item_metadata['latent_path'] = str(latent_path)
                item_metadata['latent_shape'] = list(latent.shape)
                
                latent_metadata.append(item_metadata)
    
    # Save latent metadata
    latent_metadata_path = output_dir / "latent_metadata.json"
    with open(latent_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(latent_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Cached {len(latent_metadata)} latents")
    print(f"Latents directory: {latents_dir}")
    print(f"Metadata: {latent_metadata_path}")
    
    # Verify reconstruction quality
    if verify_samples > 0:
        print("\n" + "="*60)
        print(f"Verification: Decoding {verify_samples} Random Samples")
        print("="*60)
        
        verify_dir = output_dir / "verification"
        verify_dir.mkdir(exist_ok=True)
        
        # Sample random indices
        indices = np.random.choice(len(latent_metadata), min(verify_samples, len(latent_metadata)), replace=False)
        
        metrics = {
            'psnr': [],
            'mse': []
        }
        
        for idx in indices:
            item = latent_metadata[idx]
            filename = item.get('preprocessed_filename', item.get('filename'))
            latent_path = Path(item['latent_path'])
            
            # Load original image
            orig_img_path = images_dir / filename
            orig_img = Image.open(orig_img_path).convert('RGB')
            orig_tensor = transforms.ToTensor()(orig_img).unsqueeze(0).to(device)
            orig_tensor = orig_tensor * 2 - 1  # [0,1] -> [-1,1]
            
            # Load cached latent
            latent = torch.load(latent_path).unsqueeze(0).to(device)
            latent = latent / vae.config.scaling_factor
            
            # Decode
            with torch.no_grad():
                reconstructed = vae.decode(latent).sample
            
            # Calculate metrics
            mse = torch.mean((orig_tensor - reconstructed) ** 2).item()
            psnr = 10 * np.log10(4.0 / mse)  # Range is [-1, 1] so max squared diff is 4
            
            metrics['mse'].append(mse)
            metrics['psnr'].append(psnr)
            
            # Save comparison
            orig_np = ((orig_tensor[0].cpu() + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            recon_np = ((reconstructed[0].cpu() + 1) / 2 * 255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
            
            # Side-by-side comparison
            comparison = np.concatenate([orig_np, recon_np], axis=1)
            comparison_img = Image.fromarray(comparison)
            comparison_img.save(verify_dir / f"verify_{Path(filename).stem}.png")
        
        # Print metrics
        avg_psnr = np.mean(metrics['psnr'])
        avg_mse = np.mean(metrics['mse'])
        
        print(f"\nReconstruction Quality:")
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
        print(f"  Average MSE:  {avg_mse:.6f}")
        print(f"  Comparisons saved to: {verify_dir}")
        
        if avg_psnr > 25:
            print("  ✅ Excellent quality (PSNR > 25 dB)")
        elif avg_psnr > 20:
            print("  ✅ Good quality (PSNR > 20 dB)")
        else:
            print("  ⚠️  Low quality (PSNR < 20 dB)")
    
    print("\n" + "="*60)
    print("✅ Latent Caching Complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Cache latents by encoding preprocessed images"
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to preprocessed metadata JSON'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        required=True,
        help='Directory with preprocessed images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save cached latents'
    )
    parser.add_argument(
        '--teacher-model',
        type=str,
        default='martineux/janku6',
        help='HuggingFace model ID for VAE (default: martineux/janku6)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for encoding (default: 4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--verify-samples',
        type=int,
        default=5,
        help='Number of samples to verify by decoding (default: 5, 0=skip)'
    )
    
    args = parser.parse_args()
    
    metadata_path = Path(args.metadata)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    cache_latents(
        metadata_path,
        images_dir,
        output_dir,
        args.teacher_model,
        args.batch_size,
        args.device,
        args.verify_samples
    )


if __name__ == "__main__":
    main()
