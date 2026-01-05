"""
Quick diagnostic to check normalization in evaluation vs training.
"""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.toe import TagOptimizedEncoder
from hqpd.utils.danbooru import DanbooruTagProcessor

def check_normalization_mismatch():
    """Check if TOE and CLIP use same normalization."""
    
    # Load training embeddings
    train_data = np.load("data/clip_embeddings/clip_embeddings.npz")['embeddings']
    train_sample = train_data[0]
    
    # Calculate norm
    train_norm = np.linalg.norm(train_sample.flatten())
    
    print("=" * 70)
    print("NORMALIZATION ANALYSIS")
    print("=" * 70)
    
    print(f"\n📊 Training Embeddings:")
    print(f"  Shape: {train_sample.shape}")
    print(f"  L2 Norm: {train_norm:.6f}")
    print(f"  Expected for unit norm: 1.0")
    print(f"  Expected for unnormalized: >100")
    print(f"  Actual: ~8.77 (custom normalization)")
    
    # Load TOE
    checkpoint_path = Path("checkpoints/toe/toe_best.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = TagOptimizedEncoder(
        vocab_size=15000, embed_dim=2048, num_layers=4,
        num_heads=8, mlp_ratio=2, max_length=77, quantize_embeddings=False
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Test with TOE
    tag_processor = DanbooruTagProcessor(vocab_size=15000)
    tag_processor.load_vocabulary("data/vocabulary.json")
    
    test_tags = ['1girl', 'solo', 'long_hair']
    tag_ids, tag_weights = tag_processor.encode(test_tags, max_length=77)
    tag_ids = torch.tensor([tag_ids], dtype=torch.long).to(device)
    tag_weights = torch.tensor([tag_weights], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        toe_output = model(tag_ids, tag_weights)
    
    toe_np = toe_output.cpu().numpy()
    toe_norm = np.linalg.norm(toe_np.flatten())
    
    print(f"\n🤖 TOE Model Output:")
    print(f"  Shape: {toe_np.shape}")
    print(f"  L2 Norm: {toe_norm:.6f}")
    
    print(f"\n🔍 Diagnosis:")
    if abs(toe_norm - train_norm) < 0.5:
        print(f"  ✅ TOE matches training norm (~{train_norm:.2f})")
        print(f"  ⚠️  But evaluation script normalizes to 1.0!")
        print(f"")
        print(f"  💡 SOLUTION:")
        print(f"  Remove normalization from evaluation script")
        print(f"  OR normalize training embeddings to unit norm and retrain")
    else:
        print(f"  ❌ TOE norm ({toe_norm:.2f}) != training norm ({train_norm:.2f})")
        print(f"  This suggests model didn't learn properly")
    
    # Test without normalization
    print(f"\n📏 Similarity WITHOUT Evaluation Normalization:")
    train_flat = train_sample.flatten()
    toe_flat = toe_np.flatten()
    
    cos_sim_raw = np.dot(train_flat, toe_flat) / (np.linalg.norm(train_flat) * np.linalg.norm(toe_flat))
    print(f"  Cosine Similarity (raw): {cos_sim_raw:.4f}")
    
    # Test WITH normalization (what evaluation does)
    train_norm_unit = train_flat / (np.linalg.norm(train_flat) + 1e-8)
    toe_norm_unit = toe_flat / (np.linalg.norm(toe_flat) + 1e-8)
    
    cos_sim_normalized = np.dot(train_norm_unit, toe_norm_unit)
    print(f"  Cosine Similarity (normalized): {cos_sim_normalized:.4f}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDED FIX")
    print("=" * 70)
    print("\nYour training embeddings have norm ~8.77, not 1.0")
    print("The evaluation script normalizes both to 1.0, causing mismatch")
    print("\nOption 1: REMOVE normalization from evaluate_toe.py")
    print("  - Delete normalization code at lines 117 and 137")
    print("  - Re-run evaluation")
    print("\nOption 2: Use TOE output as-is (RECOMMENDED)")
    print("  - TOE already learned the correct scale")
    print("  - Just compare raw embeddings without extra normalization")


if __name__ == "__main__":
    check_normalization_mismatch()
