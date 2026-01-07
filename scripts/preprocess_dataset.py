#!/usr/bin/env python3
"""
Unified dataset preprocessing script with intelligent cropping strategies.

Implements 4 strategies:
1. Selective filtering (only near-square images)
2. Smart cropping with anime face detection
3. Resize & pad (composition-preserving)
4. Multi-tile training (for extreme aspect ratios)

Optimized with parallel processing for CPU efficiency.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
from tqdm.auto import tqdm
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from PIL import Image
import torch

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.utils.image_processing import (
    calculate_aspect_ratio,
    classify_aspect_bucket,
    smart_crop_with_face,
    center_crop,
    resize_and_pad,
    create_tiles
)


# Default number of workers (use 75% of CPU cores for efficiency)
DEFAULT_WORKERS = max(1, int(mp.cpu_count() * 0.75))


def process_single_smart_crop(args: Tuple) -> Optional[Dict]:
    """Worker function for parallel smart crop processing."""
    item, images_dir, output_dir, target_size, cascade_path = args
    
    try:
        filename = item.get('filename')
        img_path = images_dir / filename
        
        if not img_path.exists():
            return None
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        ratio = calculate_aspect_ratio(img_path)
        
        # Smart crop
        cropped, crop_metadata = smart_crop_with_face(
            image,
            target_size,
            cascade_path
        )
        
        # Save
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, quality=95)
        
        # Update metadata
        new_item = item.copy()
        new_item.update({
            'preprocessed_filename': filename,
            'strategy': 'smart_crop',
            'original_ratio': ratio,
            'crop_metadata': crop_metadata
        })
        return new_item
        
    except Exception as e:
        print(f"\nError processing {item.get('filename')}: {e}")
        return None


def process_single_selective(args: Tuple) -> Optional[Dict]:
    """Worker function for parallel selective processing."""
    item, images_dir, output_dir, target_size, min_ratio, max_ratio = args
    
    try:
        filename = item.get('filename')
        img_path = images_dir / filename
        
        if not img_path.exists():
            return None
        
        ratio = calculate_aspect_ratio(img_path)
        
        # Filter by ratio
        if not (min_ratio <= ratio <= max_ratio):
            return None
        
        # Load and crop
        image = Image.open(img_path).convert('RGB')
        cropped = center_crop(image, target_size)
        
        # Save
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, quality=95)
        
        new_item = item.copy()
        new_item.update({
            'preprocessed_filename': filename,
            'strategy': 'selective',
            'original_ratio': ratio,
            'crop_type': 'center'
        })
        return new_item
        
    except Exception as e:
        print(f"\nError processing {item.get('filename')}: {e}")
        return None


def preprocess_selective(
    metadata:List[Dict],
    images_dir: Path,
    output_dir: Path,
    target_size: int,
    min_ratio: float,
    max_ratio: float
) -> List[Dict]:
    """
    Strategy 1: Selective filtering with center crop.
    
    Only process images within aspect ratio range.
    """
    print("\n" + "="*60)
    print("Strategy 1: Selective Filtering")
    print("="*60)
    print(f"Aspect ratio range: {min_ratio:.2f} - {max_ratio:.2f}")
    
    processed_metadata = []
    skipped = 0
    
    for item in tqdm(metadata, desc="Processing"):
        filename = item.get('filename')
        img_path = images_dir / filename
        
        if not img_path.exists():
            skipped += 1
            continue
        
        # Calculate aspect ratio
        ratio = calculate_aspect_ratio(img_path)
        
        # Filter by ratio
        if not (min_ratio <= ratio <= max_ratio):
            skipped += 1
            continue
        
        # Load and crop
        image = Image.open(img_path).convert('RGB')
        cropped = center_crop(image, target_size)
        
        # Save processed image
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path, quality=95)
        
        # Update metadata
        new_item = item.copy()
        new_item.update({
            'preprocessed_filename': filename,
            'strategy': 'selective',
            'original_ratio': ratio,
            'crop_type': 'center'
        })
        processed_metadata.append(new_item)
    
    print(f"\nProcessed: {len(processed_metadata)}")
    print(f"Skipped: {skipped}")
    
    return processed_metadata


def preprocess_smart_crop(
    metadata: List[Dict],
    images_dir: Path,
    output_dir: Path,
    target_size: int,
    cascade_path: Path,
    num_workers: int = DEFAULT_WORKERS
) -> List[Dict]:
    """
    Strategy 2: Smart cropping with anime face detection.
    Parallelized for CPU efficiency.
    """
    print("\n" + "="*60)
    print("Strategy 2: Smart Cropping (Face Detection) - PARALLEL")
    print("="*60)
    print(f"Cascade: {cascade_path}")
    print(f"Workers: {num_workers} (CPU cores: {mp.cpu_count()})")
    
    if not cascade_path.exists():
        print(f"\n❌ Cascade not found: {cascade_path}")
        print("Run: python scripts/download_animeface_detector.py")
        return []
    
    # Prepare args for parallel processing
    work_items = [
        (item, images_dir, output_dir, target_size, cascade_path)
        for item in metadata
    ]
    
    processed_metadata = []
    face_detected_count = 0
    fallback_count = 0
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_smart_crop, args): args for args in work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                processed_metadata.append(result)
                if result.get('crop_metadata', {}).get('face_detected'):
                    face_detected_count += 1
                else:
                    fallback_count += 1
    
    print(f"\nProcessed: {len(processed_metadata)}")
    if len(processed_metadata) > 0:
        print(f"Face detected: {face_detected_count} ({face_detected_count/len(processed_metadata)*100:.1f}%)")
        print(f"Fallback (center crop): {fallback_count} ({fallback_count/len(processed_metadata)*100:.1f}%)")
    
    return processed_metadata


def preprocess_pad(
    metadata: List[Dict],
    images_dir: Path,
    output_dir: Path,
    masks_dir: Path,
    target_size: int,
    pad_color: int
) -> List[Dict]:
    """
    Strategy 3: Resize & pad (composition-preserving).
    """
    print("\n" + "="*60)
    print("Strategy 3: Resize & Pad")
    print("="*60)
    print(f"Pad color: {pad_color}")
    
    processed_metadata = []
    
    for item in tqdm(metadata, desc="Processing"):
        filename = item.get('filename')
        img_path = images_dir / filename
        
        if not img_path.exists():
            continue
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        ratio = calculate_aspect_ratio(img_path)
        
        # Resize and pad
        padded, mask = resize_and_pad(image, target_size, pad_color)
        
        # Save image
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        padded.save(output_path, quality=95)
        
        # Save mask
        mask_filename = Path(filename).stem + '_mask.pt'
        mask_path = masks_dir / mask_filename
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mask, mask_path)
        
        # Update metadata
        new_item = item.copy()
        new_item.update({
            'preprocessed_filename': filename,
            'strategy': 'pad',
            'original_ratio': ratio,
            'mask_filename': mask_filename,
            'pad_color': pad_color
        })
        processed_metadata.append(new_item)
    
    print(f"\nProcessed: {len(processed_metadata)}")
    print(f"Masks saved to: {masks_dir}")
    
    return processed_metadata


def preprocess_multitile(
    metadata: List[Dict],
    images_dir: Path,
    output_dir: Path,
    target_size: int,
    overlap: int,
    min_ratio: Optional[float] = None,
    max_ratio: Optional[float] = None
) -> List[Dict]:
    """
    Strategy 4: Multi-tile training (for large/extreme images).
    """
    print("\n" + "="*60)
    print("Strategy 4: Multi-Tile")
    print("="*60)
    print(f"Tile size: {target_size}, Overlap: {overlap}")
    
    processed_metadata = []
    total_tiles = 0
    
    for item in tqdm(metadata, desc="Processing"):
        filename = item.get('filename')
        img_path = images_dir / filename
        
        if not img_path.exists():
            continue
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        ratio = calculate_aspect_ratio(img_path)
        width, height = image.size
        
        # Filter by ratio if specified
        if min_ratio and ratio < min_ratio:
            continue
        if max_ratio and ratio > max_ratio:
            continue
        
        # Skip if image is too small
        if width < target_size or height < target_size:
            continue
        
        # Create tiles
        tiles = create_tiles(image, target_size, overlap)
        
        # Save each tile
        base_stem = Path(filename).stem
        base_ext = Path(filename).suffix
        
        for idx, tile_data in enumerate(tiles):
            tile = tile_data['tile']
            position = tile_data['position']
            
            # Create unique filename for tile
            tile_filename = f"{base_stem}_tile{idx:03d}{base_ext}"
            output_path = output_dir / tile_filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            tile.save(output_path, quality=95)
            
            # Create metadata for this tile
            new_item = item.copy()
            new_item.update({
                'preprocessed_filename': tile_filename,
                'strategy': 'multitile',
                'original_filename': filename,
                'original_ratio': ratio,
                'tile_index': idx,
                'tile_position': position,
                'total_tiles': len(tiles)
            })
            processed_metadata.append(new_item)
            total_tiles += 1
    
    print(f"\nProcessed: {len(metadata)} source images")
    print(f"Generated: {total_tiles} tiles")
    
    return processed_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess anime dataset with intelligent cropping strategies"
    )
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata JSON file'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        required=True,
        help='Directory containing source images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save processed images'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        choices=['selective', 'smart_crop', 'pad', 'multitile'],
        help='Preprocessing strategy to use'
    )
    parser.add_argument(
        '--target-size',
        type=int,
        default=1024,
        help='Target image size (default: 1024)'
    )
    
    # Strategy-specific arguments
    parser.add_argument(
        '--aspect-range',
        type=float,
        nargs=2,
        default=[0.85, 1.15],
        metavar=('MIN', 'MAX'),
        help='Aspect ratio range for selective filtering (default: 0.85 1.15)'
    )
    parser.add_argument(
        '--cascade-path',
        type=str,
        help='Path to anime face cascade (auto-detected if not specified)'
    )
    parser.add_argument(
        '--pad-color',
        type=int,
        default=0,
        help='Padding color for pad strategy (0-255, default: 0 = black)'
    )
    parser.add_argument(
        '--tile-overlap',
        type=int,
        default=128,
        help='Tile overlap for multitile strategy (default: 128)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_WORKERS}, 75%% of CPU cores)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    metadata_path = Path(args.metadata)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    masks_dir = output_dir.parent / (output_dir.name + "_masks")
    
    # Validate inputs
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Create output dirs
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    print(f"\nLoading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} images")
    
    # Execute strategy
    processed_metadata = []
    
    if args.strategy == 'selective':
        processed_metadata = preprocess_selective(
            metadata,
            images_dir,
            output_dir,
            args.target_size,
            args.aspect_range[0],
            args.aspect_range[1]
        )
    
    elif args.strategy == 'smart_crop':
        # Auto-detect cascade path if not specified
        if args.cascade_path:
            cascade_path = Path(args.cascade_path)
        else:
            script_dir = Path(__file__).parent.parent
            cascade_path = script_dir / "hqpd" / "models" / "cascades" / "lbpcascade_animeface.xml"
        
        processed_metadata = preprocess_smart_crop(
            metadata,
            images_dir,
            output_dir,
            args.target_size,
            cascade_path,
            args.workers
        )
    
    elif args.strategy == 'pad':
        masks_dir.mkdir(parents=True, exist_ok=True)
        processed_metadata = preprocess_pad(
            metadata,
            images_dir,
            output_dir,
            masks_dir,
            args.target_size,
            args.pad_color
        )
    
    elif args.strategy == 'multitile':
        processed_metadata = preprocess_multitile(
            metadata,
            images_dir,
            output_dir,
            args.target_size,
            args.tile_overlap
        )
    
    # Save processed metadata
    output_metadata_path = output_dir / "metadata.json"
    
    # Custom encoder for numpy types (OpenCV returns int32/int64)
    import numpy as np
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(processed_metadata, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print("\n" + "="*60)
    print("✅ Preprocessing Complete!")
    print("="*60)
    print(f"Processed images: {output_dir}")
    print(f"Metadata: {output_metadata_path}")
    if args.strategy == 'pad':
        print(f"Masks: {masks_dir}")
    print()


if __name__ == "__main__":
    main()
