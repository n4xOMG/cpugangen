#!/usr/bin/env python3
"""
Analyze aspect ratio distribution of anime dataset.

Generates statistics and bucket classification to inform preprocessing strategy.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import sys

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.utils.image_processing import calculate_aspect_ratio, classify_aspect_bucket


def analyze_dataset(
    metadata_path: Path,
    images_dir: Path,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Analyze aspect ratios of all images in dataset.
    
    Args:
        metadata_path: Path to JSON metadata file
        images_dir: Directory containing images
        output_path: Optional path to save analysis results
        
    Returns:
        Dictionary with analysis results
    """
    print("\n" + "="*60)
    print("Anime Dataset Aspect Ratio Analysis")
    print("="*60)
    
    # Load metadata
    print(f"\nLoading metadata from {metadata_path}...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    print(f"Found {len(metadata)} images in metadata")
    
    # Analyze each image
    print("\nAnalyzing aspect ratios...")
    results = {
        'total_images': len(metadata),
        'analyzed': 0,
        'failed': 0,
        'buckets': defaultdict(list),
        'bucket_counts': {},
        'statistics': {},
        'aspect_ratios': []
    }
    
    for i, item in enumerate(metadata):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(metadata)}...", end='\r')
        
        filename = item.get('filename')
        if not filename:
            results['failed'] += 1
            continue
        
        img_path = images_dir / filename
        if not img_path.exists():
            results['failed'] += 1
            continue
        
        # Calculate aspect ratio
        try:
            ratio = calculate_aspect_ratio(img_path)
            bucket = classify_aspect_bucket(ratio)
            
            results['aspect_ratios'].append(ratio)
            results['buckets'][bucket].append({
                'filename': filename,
                'ratio': ratio
            })
            results['analyzed'] += 1
            
        except Exception as e:
            print(f"\nError analyzing {filename}: {e}")
            results['failed'] += 1
    
    print(f"  Processed {len(metadata)}/{len(metadata)}")
    
    # Calculate statistics
    ratios = results['aspect_ratios']
    if ratios:
        results['statistics'] = {
            'min': min(ratios),
            'max': max(ratios),
            'mean': sum(ratios) / len(ratios),
            'median': sorted(ratios)[len(ratios) // 2]
        }
    
    # Bucket counts
    for bucket, items in results['buckets'].items():
        results['bucket_counts'][bucket] = len(items)
    
    # Print summary
    print("\n" + "="*60)
    print("Analysis Results")
    print("="*60)
    
    print(f"\nTotal images: {results['total_images']}")
    print(f"Successfully analyzed: {results['analyzed']}")
    print(f"Failed: {results['failed']}")
    
    if results['statistics']:
        print(f"\nAspect Ratio Statistics:")
        print(f"  Min:    {results['statistics']['min']:.3f}")
        print(f"  Max:    {results['statistics']['max']:.3f}")
        print(f"  Mean:   {results['statistics']['mean']:.3f}")
        print(f"  Median: {results['statistics']['median']:.3f}")
    
    print(f"\nBucket Distribution:")
    print(f"  {'Bucket':<15} {'Count':<10} {'Percentage':<12} {'Ratio Range'}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*20}")
    
    bucket_info = {
        'square': '0.85 - 1.15',
        'moderate': '0.7-0.85, 1.15-1.5',
        'extreme': '<0.7, >1.5'
    }
    
    for bucket in ['square', 'moderate', 'extreme']:
        count = results['bucket_counts'].get(bucket, 0)
        percentage = (count / results['analyzed'] * 100) if results['analyzed'] > 0 else 0
        range_str = bucket_info[bucket]
        print(f"  {bucket:<15} {count:<10} {percentage:>6.1f}%      {range_str}")
    
    # Recommendations
    print("\n" + "="*60)
    print("Preprocessing Strategy Recommendations")
    print("="*60)
    
    square_pct = (results['bucket_counts'].get('square', 0) / results['analyzed'] * 100) if results['analyzed'] > 0 else 0
    moderate_pct = (results['bucket_counts'].get('moderate', 0) / results['analyzed'] * 100) if results['analyzed'] > 0 else 0
    extreme_pct = (results['bucket_counts'].get('extreme', 0) / results['analyzed'] * 100) if results['analyzed'] > 0 else 0
    
    print()
    if square_pct > 70:
        print("✅ RECOMMENDED: Strategy 1 (Selective Filtering)")
        print("   > 70% of images are near-square")
        print("   → Use simple center crop, discard non-square images")
        print("   → Clean dataset, maximum SDXL compatibility")
    elif square_pct > 50:
        print("✅ RECOMMENDED: Hybrid Approach")
        print("   50-70% images are near-square")
        print("   → Use Strategy 1 (center crop) for square images")
        print("   → Use Strategy 2 (smart crop) for moderate images")
        print("   → Discard extreme aspect ratios")
    else:
        print("✅ RECOMMENDED: Strategy 2 (Smart Cropping)")
        print("   < 50% of images are near-square")
        print("   → Use anime face detection for all images")
        print("   → Preserves character details in vertical/horizontal art")
    
    if moderate_pct > 25:
        print(f"\n⚠️  {moderate_pct:.1f}% of images have moderate aspect ratios")
        print("   → Consider Strategy 2 (smart crop) or Strategy 3 (pad)")
    
    if extreme_pct > 10:
        print(f"\n⚠️  {extreme_pct:.1f}% of images have extreme aspect ratios")
        print("   → Consider Strategy 4 (multi-tile) or exclude these images")
        print("   → Extreme crops will lose significant content")
    
    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare JSON-serializable results
        save_results = {
            'total_images': results['total_images'],
            'analyzed': results['analyzed'],
            'failed': results['failed'],
            'bucket_counts': dict(results['bucket_counts']),
            'statistics': results['statistics'],
            'buckets': {k: v for k, v in results['buckets'].items()}  # Convert defaultdict
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to {output_path}")
    
    print("\n" + "="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze aspect ratio distribution of anime dataset"
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
        help='Directory containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save analysis results (JSON)'
    )
    
    args = parser.parse_args()
    
    metadata_path = Path(args.metadata)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output) if args.output else None
    
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return
    
    if not images_dir.exists():
        print(f"❌ Images directory not found: {images_dir}")
        return
    
    analyze_dataset(metadata_path, images_dir, output_path)


if __name__ == "__main__":
    main()
