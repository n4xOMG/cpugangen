"""
Debug script to check CLIP embedding statistics and dimensions.
Run this to diagnose the high loss issue.
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_embeddings():
    """Check pre-computed CLIP embeddings statistics."""
    
    # Load embeddings
    embeddings_path = Path(__file__).parent.parent / "data" / "clip_embeddings" / "clip_embeddings.npz"
    metadata_path = Path(__file__).parent.parent / "data" / "clip_embeddings" / "metadata.json"
    
    if not embeddings_path.exists():
        print(f"ERROR: Embeddings not found at {embeddings_path}")
        return
    
    print("Loading embeddings...")
    data = np.load(embeddings_path)
    embeddings = data['embeddings']
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"CLIP Embeddings Statistics")
    print(f"{'='*60}")
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")
    print(f"Samples: {metadata['num_samples']}")
    
    # Statistics
    print(f"\nValue Statistics:")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std:  {embeddings.std():.6f}")
    print(f"  Min:  {embeddings.min():.6f}")
    print(f"  Max:  {embeddings.max():.6f}")
    
    # Per-sample statistics
    sample_norms = np.linalg.norm(embeddings.reshape(len(embeddings), -1), axis=1)
    print(f"\nPer-sample L2 norms:")
    print(f"  Mean: {sample_norms.mean():.6f}")
    print(f"  Std:  {sample_norms.std():.6f}")
    print(f"  Min:  {sample_norms.min():.6f}")
    print(f"  Max:  {sample_norms.max():.6f}")
    
    # Check for NaN/Inf
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    print(f"\nData Quality:")
    print(f"  Contains NaN: {has_nan}")
    print(f"  Contains Inf: {has_inf}")
    
    # Sample a few embeddings
    print(f"\nSample embeddings (first 5 values of first sample):")
    print(f"  {embeddings[0, 0, :5]}")
    
    # Compare to random embeddings
    print(f"\n{'='*60}")
    print(f"Comparison to Random Embeddings (torch.randn)")
    print(f"{'='*60}")
    random_emb = np.random.randn(100, 77, 2048).astype(np.float32)
    print(f"Random Mean: {random_emb.mean():.6f}")
    print(f"Random Std:  {random_emb.std():.6f}")
    random_norms = np.linalg.norm(random_emb.reshape(len(random_emb), -1), axis=1)
    print(f"Random L2 norm mean: {random_norms.mean():.6f}")
    
    # MSE between random samples
    print(f"\n{'='*60}")
    print(f"Expected Loss Magnitudes")
    print(f"{'='*60}")
    
    # MSE between random embeddings (what dummy training had)
    mse_random = ((random_emb[0] - random_emb[1]) ** 2).mean()
    print(f"MSE between random samples: {mse_random:.6f}")
    
    # MSE between real embeddings
    if len(embeddings) >= 2:
        mse_real = ((embeddings[0] - embeddings[1]) ** 2).mean()
        print(f"MSE between real CLIP samples: {mse_real:.6f}")
    
    # MSE between real and random
    mse_real_vs_random = ((embeddings[0] - random_emb[0]) ** 2).mean()
    print(f"MSE between real and random: {mse_real_vs_random:.6f}")
    
    print(f"\n{'='*60}")
    print(f"Diagnosis")
    print(f"{'='*60}")
    
    if sample_norms.mean() > 100:
        print("⚠️  WARNING: CLIP embeddings have very large norms!")
        print("   This will cause high loss values.")
        print("   Solution: Normalize embeddings or use smaller learning rate")
    
    if embeddings.std() > 10:
        print("⚠️  WARNING: CLIP embeddings have high variance!")
        print("   This might cause training instability.")
    
    if has_nan or has_inf:
        print("❌ ERROR: Embeddings contain NaN or Inf values!")
        print("   Re-generate embeddings.")
    else:
        print("✓ No NaN/Inf values detected")
    
    return embeddings


if __name__ == "__main__":
    check_embeddings()
