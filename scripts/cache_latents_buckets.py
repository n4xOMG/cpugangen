#!/usr/bin/env python3
"""
Cache latents for bucket-based training.

Handles multiple resolutions by grouping images by bucket size.
Each bucket gets its own latent cache for consistent batching.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from tqdm.auto import tqdm
import numpy as np


class BucketImageDataset(Dataset):
    """Dataset for a single bucket resolution."""
    
    def __init__(
        self,
        metadata: List[Dict],
        images_dir: Path,
        bucket: Tuple[int, int]
    ):
        self.metadata = metadata
        self.images_dir = images_dir
        self.bucket = bucket
        self.width, self.height = bucket
        
        # Transform for this bucket
        self.transform = transforms.Compose([
            transforms.Resize((self.height, self.width), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        filename = item.get('preprocessed_filename', item.get('filename'))
        
        # Images are in bucket subfolders
        bucket_name = f"{self.width}x{self.height}"
        img_path = self.images_dir / bucket_name / filename
        
        # Fallback to flat structure
        if not img_path.exists():
            img_path = self.images_dir / filename
        
        image = Image.open(img_path).convert('RGB')
        pixel_values = self.transform(image)
        
        return {
            'pixel_values': pixel_values,
            'filename': filename,
            'metadata': item
        }


def custom_collate_fn(batch):
    """Custom collate for varying metadata."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    filenames = [item['filename'] for item in batch]
    metadata_list = [item['metadata'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'filename': filenames,
        'metadata': metadata_list
    }


def cache_bucket_latents(
    metadata_path: Path,
    images_dir: Path,
    output_dir: Path,
    teacher_model: str,
    batch_size: int = 4,
    device: str = 'cuda',
    verify_samples: int = 5
):
    """
    Cache latents for bucket-based images.
    
    Groups images by bucket, processes each bucket separately.
    """
    print("\n" + "="*60)
    print("Bucket Latent Caching")
    print("="*60)
    print(f"Teacher model: {teacher_model}")
    print(f"Device: {device}")
    
    # Load metadata
    print(f"\nLoading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Group by bucket
    buckets = {}
    for item in metadata:
        bucket = item.get('bucket', '1024x1024')
        if bucket not in buckets:
            buckets[bucket] = []
        buckets[bucket].append(item)
    
    print(f"\nFound {len(metadata)} images in {len(buckets)} buckets:")
    for bucket, items in sorted(buckets.items(), key=lambda x: -len(x[1])):
        print(f"  {bucket}: {len(items)} images")
    
    # Load VAE
    print(f"\nLoading VAE from {teacher_model}...")
    vae = AutoencoderKL.from_pretrained(
        teacher_model,
        subfolder="vae",
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    print(f"✅ VAE loaded")
    
    # Create output structure
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_latent_metadata = []
    
    # Process each bucket
    for bucket_name, bucket_items in buckets.items():
        print(f"\n{'='*60}")
        print(f"Processing Bucket: {bucket_name}")
        print(f"{'='*60}")
        
        # Parse bucket dimensions
        w, h = map(int, bucket_name.split('x'))
        bucket = (w, h)
        
        # Create bucket output dirs
        bucket_latents_dir = output_dir / bucket_name / "latents"
        bucket_latents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset and loader
        dataset = BucketImageDataset(bucket_items, images_dir, bucket)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device == 'cuda'),
            collate_fn=custom_collate_fn
        )
        
        # Expected latent size for this bucket
        latent_h = h // 8
        latent_w = w // 8
        print(f"Image size: {w}×{h} → Latent size: {latent_w}×{latent_h}")
        
        # Encode
        bucket_metadata = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Encoding {bucket_name}"):
                pixel_values = batch['pixel_values'].to(device)
                filenames = batch['filename']
                batch_meta = batch['metadata']
                
                # Encode
                latent_dist = vae.encode(pixel_values).latent_dist
                latents = latent_dist.sample() * vae.config.scaling_factor
                
                # Save each latent
                for i, filename in enumerate(filenames):
                    latent = latents[i].cpu()
                    
                    latent_filename = Path(filename).stem + '.pt'
                    latent_path = bucket_latents_dir / latent_filename
                    torch.save(latent, latent_path)
                    
                    item_meta = batch_meta[i].copy()
                    item_meta['latent_filename'] = latent_filename
                    item_meta['latent_path'] = str(latent_path)
                    item_meta['latent_shape'] = list(latent.shape)
                    item_meta['bucket'] = bucket_name
                    
                    bucket_metadata.append(item_meta)
        
        # Save bucket metadata
        bucket_meta_path = output_dir / bucket_name / "latent_metadata.json"
        with open(bucket_meta_path, 'w', encoding='utf-8') as f:
            json.dump(bucket_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Cached {len(bucket_metadata)} latents for {bucket_name}")
        all_latent_metadata.extend(bucket_metadata)
    
    # Save combined metadata
    combined_meta_path = output_dir / "all_latent_metadata.json"
    with open(combined_meta_path, 'w', encoding='utf-8') as f:
        json.dump(all_latent_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("✅ Bucket Latent Caching Complete!")
    print(f"{'='*60}")
    print(f"Total latents: {len(all_latent_metadata)}")
    print(f"Output: {output_dir}")
    print(f"Combined metadata: {combined_meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Cache latents for bucket-based training")
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--teacher-model', type=str, default='martineux/janku6')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--verify-samples', type=int, default=5)
    
    args = parser.parse_args()
    
    cache_bucket_latents(
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
