#!/usr/bin/env python3
"""
Dataset loader for knowledge distillation training.
Handles JSON metadata with tags and scores.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class AnimeDistillationDataset(Dataset):
    """
    Dataset for knowledge distillation training on anime images.
    
    Loads images and their Danbooru tags from JSON metadata.
    """
    
    def __init__(
        self,
        metadata_path: str,
        images_dir: str,
        resolution: int = 1024,
        max_tags: int = 75,
        min_tag_score: float = 0.35,
        use_caption: bool = True,
        use_character_tags: bool = True
    ):
        """
        Args:
            metadata_path: Path to JSON file with metadata
            images_dir: Directory containing images
            resolution: Target image resolution (SDXL uses 1024)
            max_tags: Maximum number of tags to use
            min_tag_score: Minimum score threshold for tags
            use_caption: Whether to use caption field
            use_character_tags: Whether to prepend character tags
        """
        self.images_dir = Path(images_dir)
        self.resolution = resolution
        self.max_tags = max_tags
        self.min_tag_score = min_tag_score
        self.use_caption = use_caption
        self.use_character_tags = use_character_tags
        
        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Filter out missing images
        self.valid_samples = []
        for item in self.metadata:
            img_path = self.images_dir / item['filename']
            if img_path.exists():
                self.valid_samples.append(item)
            else:
                print(f"Warning: Image not found: {img_path}")
        
        print(f"Loaded {len(self.valid_samples)} valid samples")
        
        # Image transforms for SDXL (1024x1024)
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self) -> int:
        return len(self.valid_samples)
    
    def _build_prompt(self, item: Dict) -> str:
        """
        Build prompt from metadata using tags and scores.
        
        Strategy:
        1. Prioritize character tags (if enabled)
        2. Use caption if use_caption=True
        3. Otherwise, use general_tags filtered by score
        """
        tags = []
        
        # Add character tags first (usually important)
        if self.use_character_tags and item.get('character_tags'):
            tags.extend(item['character_tags'])
        
        # Use caption or filtered general tags
        if self.use_caption and item.get('caption'):
            # Caption is already comma-separated string
            caption_tags = [t.strip() for t in item['caption'].split(',')]
            tags.extend(caption_tags)
        elif item.get('general_scores'):
            # Filter by score threshold
            high_score_tags = [
                score_item['tag'] 
                for score_item in item['general_scores']
                if score_item['score'] >= self.min_tag_score
            ]
            tags.extend(high_score_tags)
        elif item.get('general_tags'):
            # Fallback to all general tags
            tags.extend(item['general_tags'])
        
        # Limit to max_tags
        tags = tags[:self.max_tags]
        
        # Join with commas
        prompt = ', '.join(tags)
        
        return prompt
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
                - pixel_values: Image tensor (C, H, W)
                - prompt: Text prompt string
                - filename: Original filename
        """
        item = self.valid_samples[idx]
        
        # Load image
        img_path = self.images_dir / item['filename']
        image = Image.open(img_path).convert('RGB')
        
        # Transform image
        pixel_values = self.transform(image)
        
        # Build prompt
        prompt = self._build_prompt(item)
        
        return {
            'pixel_values': pixel_values,
            'prompt': prompt,
            'filename': item['filename']
        }


def collate_fn(batch: List[Dict]) -> Dict[str, any]:
    """
    Custom collate function for DataLoader.
    
    Handles batching of images and prompts.
    """
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    filenames = [item['filename'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'prompts': prompts,
        'filenames': filenames
    }


def create_train_val_split(
    metadata_path: str,
    output_dir: str,
    val_ratio: float = 0.05,
    seed: int = 42
):
    """
    Split metadata JSON into train and validation sets.
    
    Args:
        metadata_path: Path to full metadata JSON
        output_dir: Directory to save train/val splits
        val_ratio: Fraction of data for validation
        seed: Random seed for reproducibility
    """
    import random
    
    random.seed(seed)
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Shuffle
    random.shuffle(metadata)
    
    # Split
    val_size = int(len(metadata) * val_ratio)
    val_data = metadata[:val_size]
    train_data = metadata[val_size:]
    
    # Save splits
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train_metadata.json')
    val_path = os.path.join(output_dir, 'val_metadata.json')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created splits:")
    print(f"  Train: {len(train_data)} samples → {train_path}")
    print(f"  Val:   {len(val_data)} samples → {val_path}")


if __name__ == "__main__":
    """Test dataset loading"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loader")
    parser.add_argument("--metadata", required=True, help="Path to metadata JSON")
    parser.add_argument("--images-dir", required=True, help="Directory with images")
    parser.add_argument("--create-split", action="store_true", help="Create train/val split")
    parser.add_argument("--split-output", default="data/splits", help="Output dir for splits")
    args = parser.parse_args()
    
    if args.create_split:
        # Create train/val split
        create_train_val_split(
            args.metadata,
            args.split_output,
            val_ratio=0.05
        )
    else:
        # Test dataset loading
        dataset = AnimeDistillationDataset(
            metadata_path=args.metadata,
            images_dir=args.images_dir,
            resolution=1024,
            max_tags=75,
            min_tag_score=0.35
        )
        
        print(f"\nDataset size: {len(dataset)}")
        
        # Test first sample
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Filename: {sample['filename']}")
        print(f"  Image shape: {sample['pixel_values'].shape}")
        print(f"  Prompt: {sample['prompt'][:100]}...")
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        batch = next(iter(loader))
        print(f"\nBatch test:")
        print(f"  Pixel values: {batch['pixel_values'].shape}")
        print(f"  Prompts: {len(batch['prompts'])}")
        print(f"  First prompt: {batch['prompts'][0][:100]}...")
