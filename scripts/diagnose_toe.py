#!/usr/bin/env python3
"""
Diagnose TOE checkpoint issues.
"""

import torch
from pathlib import Path

# Check checkpoint
ckpt_path = "checkpoints/toe/toe_with_pooling_best.pt"
checkpoint = torch.load(ckpt_path, map_location="cpu")

print("=" * 70)
print("TOE Checkpoint Diagnosis")
print("=" * 70)

print(f"\nCheckpoint file: {ckpt_path}")
print(f"File size: {Path(ckpt_path).stat().st_size / (1024**2):.2f} MB")

print(f"\nCheckpoint keys: {list(checkpoint.keys())}")

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print(f"\nModel state dict keys (first 10):")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {key}: {state_dict[key].shape}")
    
    # Check if weights look initialized
    print(f"\nWeight statistics:")
    for key in list(state_dict.keys())[:5]:
        tensor = state_dict[key]
        print(f"\n  {key}:")
        print(f"    Mean: {tensor.mean():.6f}")
        print(f"    Std: {tensor.std():.6f}")
        print(f"    Min: {tensor.min():.6f}")
        print(f"    Max: {tensor.max():.6f}")
        
        # Check if it looks like random initialization
        if abs(tensor.mean()) < 0.01 and abs(tensor.std() - 0.02) < 0.01:
            print(f"    ⚠️  WARNING: Looks like random initialization!")
    
    # Check training metadata
    if 'epoch' in checkpoint:
        print(f"\nTraining metadata:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Loss: {checkpoint.get('loss', 'N/A')}")
        print(f"  Best loss: {checkpoint.get('best_loss', 'N/A')}")
else:
    print("\n⚠️  No 'model_state_dict' key found!")
    print("Checkpoint structure might be different.")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if 'epoch' in checkpoint and checkpoint.get('epoch', 0) < 5:
    print("❌ Model appears to be from early training (< 5 epochs)")
    print("   Recommendation: Train TOE properly before using")
elif not any('projection' in k for k in state_dict.keys()):
    print("❌ Projection layer might be missing")
else:
    print("✅ Checkpoint structure looks okay")
    print("   Issue might be elsewhere (embedding quality, training data, etc.)")
