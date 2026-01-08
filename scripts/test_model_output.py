#!/usr/bin/env python3
"""
Quick diagnostic: check if model is learning
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.optimization.structured_init import StructuredInitializer, StructuredInitDataset

# Load model
model_path = "checkpoints/struct_init.pt"
model = StructuredInitializer.load(model_path)

print(f"Model loaded from: {model_path}")
print(f"Delta mean: {model.delta_mean.squeeze()}")
print(f"Delta std: {model.delta_std.squeeze()}")

# Load a few samples
dataset = StructuredInitDataset("data/struct_init_training")
print(f"\nDataset: {len(dataset)} samples")

# Test on first 5 samples
print("\nTesting predictions:")
for i in range(min(5, len(dataset))):
    pooled_embed, initial_latent, target_latent = dataset[i]
    
    # Ground truth delta
    target_delta = target_latent - initial_latent
    
    # Predict (with normalization)
    with torch.no_grad():
        predicted_delta = model(pooled_embed.unsqueeze(0), use_normalization=True)
        predicted_delta = predicted_delta.squeeze(0)
    
    # Stats
    print(f"\nSample {i}:")
    print(f"  Target delta:    mean={target_delta.mean():.3f}, std={target_delta.std():.3f}")
    print(f"  Predicted delta: mean={predicted_delta.mean():.3f}, std={predicted_delta.std():.3f}")
    print(f"  MSE: {((predicted_delta - target_delta)**2).mean():.3f}")
