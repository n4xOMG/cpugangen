#!/usr/bin/env python3
"""
Cache latents for training by pre-encoding all preprocessed images.

Uses the frozen SDXL VAE encoder to encode images to (4, 128, 128) latent tensors.
Saves latents to disk for fast training without re-encoding.
Rescues buckets (does NOT force resize) and batches by bucket to prevent stack errors.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import math
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
import numpy as np


class ImageDataset(Dataset):
    """Dataset that loads images WITHOUT forcing resize (respects buckets)."""
    
    def __init__(self, metadata: List[Dict], images_dir: Path):
        self.metadata = metadata
        self.images_dir = images_dir
        
        # Only ToTensor and Normalize. 
        # RESIZE/CROP IS REMOVED to respect the bucketed preprocessing.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # [-1, 1]
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        filename = item.get('preprocessed_filename', item.get('filename'))
        
        # Check for bucket subfolder
        bucket = item.get('bucket')
        if bucket:
            img_path = self.images_dir / bucket / filename
        else:
            img_path = self.images_dir / filename
            
        if not img_path.exists():
             # Fallback check root
            root_path = self.images_dir / filename
            if root_path.exists():
                img_path = root_path
            else:
                raise FileNotFoundError(f"Image not found at {img_path}")
            
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.transform(image)
        
        return {
            'pixel_values': pixel_values,
            'filename': filename,
            'metadata': item,
            # Store shape for grouping check (C, H, W)
            'shape': pixel_values.shape
        }


class BucketBatchSampler(Sampler):
    """
    Groups indices by their image bucket (shape) to ensure each batch 
    contains images of identical dimensions.
    """
    def __init__(self, dataset_metadata: List[Dict], batch_size: int, drop_last: bool = False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by bucket (width, height)
        self.buckets = {}
        
        for idx, item in enumerate(dataset_metadata):
            # Use 'bucket' field or default
            bucket = item.get('bucket', 'default')
            # If bucket is missing but width/height exist (from preprocess), use that
            if bucket == 'default' and 'width' in item and 'height' in item:
                bucket = f"{item['width']}x{item['height']}"
            
            if bucket not in self.buckets:
                self.buckets[bucket] = []
            self.buckets[bucket].append(idx)
            
        # Create batches
        self.batches = []
        for bucket, indices in self.buckets.items():
            # Shuffle indices within bucket
            random.shuffle(indices)
            
            # Create chunks
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i + batch_size]
                if len(batch) == batch_size or not drop_last:
                    self.batches.append(batch)
        
        # Shuffle batch order
        random.shuffle(self.batches)
        
    def __iter__(self):
        for batch in self.batches:
            yield batch
            
    def __len__(self):
        return len(self.batches)


def custom_collate_fn(batch):
    """Custom collate that handles varying metadata structures."""
    # Stack only the pixel_values tensor
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Keep filenames and metadata as lists
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
    
    # Create dataset
    dataset = ImageDataset(metadata, images_dir)
    
    # Use BucketSampler if batch > 1
    if batch_size > 1:
        print("Using BucketBatchSampler to group same-sized images...")
        batch_sampler = BucketBatchSampler(metadata, batch_size)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=2,
            pin_memory=(device == 'cuda'),
            collate_fn=custom_collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=(device == 'cuda'),
            collate_fn=custom_collate_fn
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
            batch_metadata = batch['metadata']
            
            # Encode to latents
            try:
                latent_dist = vae.encode(pixel_values).latent_dist
                latents = latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n❌ OOM Error with batch size {len(pixel_values)}. Try reducing --batch-size.")
                    torch.cuda.empty_cache()
                    sys.exit(1)
                else:
                    raise e
            
            # Save each latent
            for i, filename in enumerate(filenames):
                latent = latents[i].cpu()
                
                # Create latent filename
                latent_filename = Path(filename).stem + '.pt'
                latent_path = latents_dir / latent_filename
                
                # Save latent tensor
                torch.save(latent, latent_path)
                
                # Get metadata
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


def main():
    parser = argparse.ArgumentParser(description="Cache latents by encoding preprocessed images")
    parser.add_argument('--metadata', type=str, required=True, help='Path to metadata JSON')
    parser.add_argument('--images-dir', type=str, required=True, help='Images directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--teacher-model', type=str, default='martineux/janku6', help='VAE model ID')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (default: 1 for safety)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--verify-samples', type=int, default=0, help='Verify samples count')
    
    args = parser.parse_args()
    
    cache_latents(
        Path(args.metadata),
        Path(args.images_dir),
        Path(args.output_dir),
        args.teacher_model,
        args.batch_size,
        args.device,
        args.verify_samples
    )


if __name__ == "__main__":
    main()
