"""
Debug script to diagnose the embedding mismatch issue.
Checks normalization and scale of embeddings.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.toe import TagOptimizedEncoder
from hqpd.utils.danbooru import DanbooruTagProcessor

def check_embedding_norms():
    """Check the norm/scale of TOE vs CLIP vs Training embeddings."""
    
    print("=" * 70)
    print("Debugging Embedding Mismatch")
    print("=" * 70)
    
    # Load vocabulary
    vocab_path = Path("data/vocabulary.json")
    tag_processor = DanbooruTagProcessor(vocab_size=15000)
    tag_processor.load_vocabulary(str(vocab_path))
    
    # Load TOE model
    checkpoint_path = Path("checkpoints/toe/toe_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TagOptimizedEncoder(
        vocab_size=15000, embed_dim=2048, num_layers=4,
        num_heads=8, mlp_ratio=2, max_length=77, quantize_embeddings=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load training embeddings
    train_embeddings = np.load("data/clip_embeddings/clip_embeddings.npz")['embeddings']
    
    # Test case
    test_tags = ['1girl', 'solo', 'long_hair', 'blue_eyes']
    print(f"\nTest tags: {test_tags}")
    
    # Encode with TOE
    tag_ids, tag_weights = tag_processor.encode(test_tags, max_length=77)
    tag_ids = torch.tensor([tag_ids], dtype=torch.long).to(device)
    tag_weights = torch.tensor([tag_weights], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        toe_output = model(tag_ids, tag_weights)
    
    # Statistics
    print("\n" + "=" * 70)
    print("Embedding Statistics")
    print("=" * 70)
    
    # Training embeddings (from pre-computed)
    train_sample = train_embeddings[0]
    train_norm = np.linalg.norm(train_sample.flatten())
    train_mean = train_sample.mean()
    train_std = train_sample.std()
    
    print(f"\n📊 Training CLIP Embeddings (pre-computed):")
    print(f"  Shape: {train_sample.shape}")
    print(f"  L2 Norm: {train_norm:.4f}")
    print(f"  Mean: {train_mean:.6f}")
    print(f"  Std: {train_std:.6f}")
    print(f"  Min: {train_sample.min():.6f}")
    print(f"  Max: {train_sample.max():.6f}")
    
    # TOE output
    toe_np = toe_output.cpu().numpy()
    toe_norm = np.linalg.norm(toe_np.flatten())
    toe_mean = toe_np.mean()
    toe_std = toe_np.std()
    
    print(f"\n🤖 TOE Model Output:")
    print(f"  Shape: {toe_np.shape}")
    print(f"  L2 Norm: {toe_norm:.4f}")
    print(f"  Mean: {toe_mean:.6f}")
    print(f"  Std: {toe_std:.6f}")
    print(f"  Min: {toe_np.min():.6f}")
    print(f"  Max: {toe_np.max():.6f}")
    
    # Compute similarity
    train_flat = train_sample.flatten()
    toe_flat = toe_np.flatten()
    
    cos_sim = np.dot(train_flat, toe_flat) / (np.linalg.norm(train_flat) * np.linalg.norm(toe_flat))
    l2_dist = np.linalg.norm(train_flat - toe_flat)
    
    print(f"\n📏 Similarity Metrics:")
    print(f"  Cosine Similarity: {cos_sim:.4f}")
    print(f"  L2 Distance: {l2_dist:.4f}")
    
    # Diagnosis
    print("\n" + "=" * 70)
    print("Diagnosis")
    print("=" * 70)
    
    norm_ratio = toe_norm / train_norm
    if abs(norm_ratio - 1.0) > 0.5:
        print(f"\n⚠️  SCALE MISMATCH DETECTED!")
        print(f"  TOE norm: {toe_norm:.2f}")
        print(f"  Training norm: {train_norm:.2f}")
        print(f"  Ratio: {norm_ratio:.2f}x")
        print(f"\n  💡 Solution: Normalize embeddings in evaluation OR model output")
    
    if train_norm > 100:
        print(f"\n⚠️  Training embeddings have VERY LARGE norms!")
        print(f"  This suggests they were NOT normalized during pre-computation")
        print(f"\n  💡 Solution: Regenerate embeddings with normalization")
    elif train_norm < 2:
        print(f"\n✓ Training embeddings appear normalized (norm ~ {train_norm:.2f})")
    
    if cos_sim < 0.5:
        print(f"\n❌ Poor cosine similarity ({cos_sim:.4f})")
        print(f"  Possible causes:")
        print(f"  1. Normalization mismatch between training and evaluation")
        print(f"  2. Model didn't converge properly")
        print(f"  3. Evaluation using wrong CLIP extraction method")
    
    print("\n" + "=" * 70)
    
    # Try normalizing
    print("\nTesting with normalized embeddings:")
    print("=" * 70)
    
    # Normalize both
    train_normalized = train_flat / (np.linalg.norm(train_flat) + 1e-8)
    toe_normalized = toe_flat / (np.linalg.norm(toe_flat) + 1e-8)
    
    cos_sim_norm = np.dot(train_normalized, toe_normalized)
    l2_dist_norm = np.linalg.norm(train_normalized - toe_normalized)
    
    print(f"After normalization:")
    print(f"  Cosine Similarity: {cos_sim_norm:.4f}")
    print(f"  L2 Distance: {l2_dist_norm:.4f}")
    
    if cos_sim_norm > 0.7 and cos_sim < 0.5:
        print(f"\n✅ SOLUTION FOUND!")
        print(f"  Normalization fixes the issue!")
        print(f"  Apply normalization in evaluation or model output")


if __name__ == "__main__":
    check_embedding_norms()
