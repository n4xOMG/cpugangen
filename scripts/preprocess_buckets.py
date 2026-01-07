#!/usr/bin/env python3
"""
Bucket Resolution Preprocessing with Face Detection.

Implements SDXL-native bucket resolutions with anime face detection:
- Assigns each image to the best-matching SDXL bucket
- Uses face detection to smart-crop to bucket size
- Preserves more content than forcing 1024x1024

SDXL Native Buckets (1 megapixel total):
- 1024×1024 (1:1)
- 1152×896  (9:7)
- 896×1152  (7:9)
- 1216×832  (19:13)
- 832×1216  (13:19)
- 1344×768  (7:4)
- 768×1344  (4:7)
- 1536×640  (12:5)
- 640×1536  (5:12)
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from PIL import Image
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.utils.image_processing import smart_crop_with_face

# SDXL Native Bucket Resolutions (width, height)
SDXL_BUCKETS = [
    (1024, 1024),  # 1:1 square
    (1152, 896),   # 9:7 wide
    (896, 1152),   # 7:9 tall
    (1216, 832),   # 19:13 wide
    (832, 1216),   # 13:19 tall
    (1344, 768),   # 7:4 wide
    (768, 1344),   # 4:7 tall
    (1536, 640),   # 12:5 ultrawide
    (640, 1536),   # 5:12 ultratall
]

# Default workers
DEFAULT_WORKERS = max(1, int(mp.cpu_count() * 0.75))


def get_bucket_aspect_ratios() -> Dict[Tuple[int, int], float]:
    """Get aspect ratios for each bucket."""
    return {bucket: bucket[0] / bucket[1] for bucket in SDXL_BUCKETS}


def find_best_bucket(width: int, height: int) -> Tuple[int, int]:
    """
    Find the best SDXL bucket for an image based on aspect ratio.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        Best matching bucket (width, height)
    """
    image_ratio = width / height
    bucket_ratios = get_bucket_aspect_ratios()
    
    # Find bucket with closest aspect ratio
    best_bucket = min(
        SDXL_BUCKETS,
        key=lambda b: abs(bucket_ratios[b] - image_ratio)
    )
    
    return best_bucket


def smart_crop_to_bucket(
    image: Image.Image,
    bucket: Tuple[int, int],
    cascade_path: Path
) -> Tuple[Image.Image, Dict]:
    """
    Smart crop image to bucket size using face detection.
    
    Args:
        image: PIL Image
        bucket: Target (width, height)
        cascade_path: Path to anime face cascade
        
    Returns:
        (cropped_image, metadata)
    """
    target_w, target_h = bucket
    orig_w, orig_h = image.size
    target_ratio = target_w / target_h
    orig_ratio = orig_w / orig_h
    
    # First, resize to ensure we can crop to target
    if orig_ratio > target_ratio:
        # Image is wider than target - resize by height
        new_h = target_h
        new_w = int(orig_w * (target_h / orig_h))
    else:
        # Image is taller than target - resize by width
        new_w = target_w
        new_h = int(orig_h * (target_w / orig_w))
    
    # Resize
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Now smart crop to exact bucket size
    # Use our existing smart_crop_with_face but adapted for non-square
    cropped, crop_meta = smart_crop_bucket_with_face(
        resized, target_w, target_h, cascade_path
    )
    
    metadata = {
        'bucket': bucket,
        'original_size': (orig_w, orig_h),
        'resized_size': (new_w, new_h),
        'crop_metadata': crop_meta
    }
    
    return cropped, metadata


def smart_crop_bucket_with_face(
    image: Image.Image,
    target_w: int,
    target_h: int,
    cascade_path: Path
) -> Tuple[Image.Image, Dict]:
    """
    Smart crop to arbitrary rectangle with face detection.
    
    Args:
        image: Resized PIL Image (larger than target in at least one dimension)
        target_w: Target width
        target_h: Target height
        cascade_path: Path to anime face cascade
        
    Returns:
        (cropped_image, metadata)
    """
    import cv2
    
    img_w, img_h = image.size
    metadata = {'face_detected': False, 'crop_type': 'center'}
    
    # If already exact size, return as-is
    if img_w == target_w and img_h == target_h:
        return image, metadata
    
    # Try face detection
    try:
        cascade = cv2.CascadeClassifier(str(cascade_path))
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Find face center (use largest face)
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces[0]
            face_cx = x + w // 2
            face_cy = y + h // 2
            
            metadata['face_detected'] = True
            metadata['face_bbox'] = [int(x), int(y), int(w), int(h)]
            metadata['crop_type'] = 'face_centered'
            
            # Calculate crop region centered on face
            crop_x = face_cx - target_w // 2
            crop_y = face_cy - target_h // 2
            
            # Clamp to image bounds
            crop_x = max(0, min(crop_x, img_w - target_w))
            crop_y = max(0, min(crop_y, img_h - target_h))
            
            cropped = image.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
            return cropped, metadata
            
    except Exception as e:
        metadata['error'] = str(e)
    
    # Fallback: center crop
    crop_x = (img_w - target_w) // 2
    crop_y = (img_h - target_h) // 2
    cropped = image.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
    
    return cropped, metadata


def process_single_image(args: Tuple) -> Optional[Dict]:
    """Worker function for parallel processing."""
    item, images_dir, output_dir, cascade_path = args
    
    try:
        filename = item.get('filename')
        img_path = images_dir / filename
        
        if not img_path.exists():
            return None
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # Find best bucket
        bucket = find_best_bucket(orig_w, orig_h)
        
        # Smart crop to bucket
        cropped, crop_metadata = smart_crop_to_bucket(image, bucket, cascade_path)
        
        # Save with bucket subfolder
        bucket_name = f"{bucket[0]}x{bucket[1]}"
        bucket_dir = output_dir / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = bucket_dir / filename
        cropped.save(output_path, quality=95)
        
        # Update metadata
        new_item = item.copy()
        new_item.update({
            'preprocessed_filename': filename,
            'bucket': bucket_name,
            'bucket_width': bucket[0],
            'bucket_height': bucket[1],
            'strategy': 'bucket_smart_crop',
            'original_ratio': orig_w / orig_h,
            'crop_metadata': crop_metadata
        })
        return new_item
        
    except Exception as e:
        print(f"\nError processing {item.get('filename')}: {e}")
        return None


def preprocess_buckets(
    metadata: List[Dict],
    images_dir: Path,
    output_dir: Path,
    cascade_path: Path,
    num_workers: int = DEFAULT_WORKERS
) -> List[Dict]:
    """
    Preprocess images to SDXL bucket resolutions with face detection.
    
    Args:
        metadata: List of image metadata dicts
        images_dir: Source images directory
        output_dir: Output directory (will have bucket subfolders)
        cascade_path: Path to anime face cascade
        num_workers: Parallel workers
        
    Returns:
        Processed metadata list
    """
    print("\n" + "="*60)
    print("Bucket Resolution Preprocessing")
    print("="*60)
    print(f"SDXL Buckets: {len(SDXL_BUCKETS)}")
    for w, h in SDXL_BUCKETS:
        print(f"  {w}×{h} ({w/h:.2f})")
    print(f"\nWorkers: {num_workers}")
    
    if not cascade_path.exists():
        print(f"\n❌ Cascade not found: {cascade_path}")
        print("Run: python scripts/download_animeface_detector.py")
        return []
    
    # Prepare work items
    work_items = [
        (item, images_dir, output_dir, cascade_path)
        for item in metadata
    ]
    
    processed_metadata = []
    bucket_counts = {f"{w}x{h}": 0 for w, h in SDXL_BUCKETS}
    face_detected_count = 0
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, args): args for args in work_items}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            if result is not None:
                processed_metadata.append(result)
                bucket_counts[result['bucket']] += 1
                if result.get('crop_metadata', {}).get('crop_metadata', {}).get('face_detected'):
                    face_detected_count += 1
    
    # Print statistics
    print(f"\nProcessed: {len(processed_metadata)}")
    print(f"\nBucket Distribution:")
    for bucket, count in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / len(processed_metadata) * 100
            print(f"  {bucket}: {count} ({pct:.1f}%)")
    
    return processed_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess images to SDXL bucket resolutions with face detection"
    )
    parser.add_argument('--metadata', type=str, required=True,
                        help='Path to metadata JSON file')
    parser.add_argument('--images-dir', type=str, required=True,
                        help='Directory containing source images')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save processed images')
    parser.add_argument('--cascade-path', type=str,
                        help='Path to anime face cascade (auto-detected if not specified)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Number of parallel workers (default: {DEFAULT_WORKERS})')
    
    args = parser.parse_args()
    
    metadata_path = Path(args.metadata)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    
    # Auto-detect cascade
    if args.cascade_path:
        cascade_path = Path(args.cascade_path)
    else:
        script_dir = Path(__file__).parent.parent
        cascade_path = script_dir / "hqpd" / "models" / "cascades" / "lbpcascade_animeface.xml"
    
    # Validate inputs
    if not metadata_path.exists():
        print(f"❌ Metadata not found: {metadata_path}")
        return
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    # Load metadata
    print(f"\nLoading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} images")
    
    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process
    processed_metadata = preprocess_buckets(
        metadata, images_dir, output_dir, cascade_path, args.workers
    )
    
    # Save metadata (with numpy encoder for any int32 values)
    import numpy as np
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)
    
    output_metadata_path = output_dir / "metadata.json"
    with open(output_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(processed_metadata, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print("\n" + "="*60)
    print("✅ Bucket Preprocessing Complete!")
    print("="*60)
    print(f"Output: {output_dir}")
    print(f"Metadata: {output_metadata_path}")
    print(f"\nBucket folders created:")
    for w, h in SDXL_BUCKETS:
        bucket_dir = output_dir / f"{w}x{h}"
        if bucket_dir.exists():
            count = len(list(bucket_dir.glob("*")))
            if count > 0:
                print(f"  {w}x{h}/: {count} images")


if __name__ == "__main__":
    main()
