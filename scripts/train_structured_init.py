#!/usr/bin/env python3
"""
Train Structured Initializer

Trains the StructuredInitializer network on collected (embedding, latent) pairs.

Usage:
    python scripts/train_structured_init.py \\
        --data data/struct_init_training \\
        --epochs 50 \\
        --output checkpoints/struct_init.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.optimization.structured_init import (
    StructuredInitializer,
    StructuredInitDataset,
    StructuredInitTrainer,
)


def main():
    parser = argparse.ArgumentParser(description="Train structured initializer")
    
    parser.add_argument("--data", type=str, required=True, help="Training data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="checkpoints/struct_init.pt", help="Output checkpoint")
    parser.add_argument("--blur-sigma", type=float, default=4.0, help="Blur sigma for low-freq loss")
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRAINING STRUCTURED INITIALIZER")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = StructuredInitDataset(args.data)
    print(f"  Found {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    # Create model
    print("\nCreating model...")
    model = StructuredInitializer(
        embed_dim=1280,
        hidden_dim=512,
        latent_channels=4,
        latent_size=64,
        low_res=8,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params * 4 / 1024 / 1024:.1f} MB)")
    
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")
    
    # Create trainer
    trainer = StructuredInitTrainer(
        model=model,
        lr=args.lr,
        blur_sigma=args.blur_sigma,
        device=device,
    )
    
    # Training loop
    print("\nTraining...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(dataloader, verbose=False)
        
        print(f"Epoch {epoch+1}/{args.epochs}: loss={metrics['avg_loss']:.4f}, delta={metrics['avg_loss_delta']:.4f}")
        
        # Save best
        if metrics['avg_loss'] < best_loss:
            best_loss = metrics['avg_loss']
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path))
            print(f"  ✓ Saved checkpoint (loss={best_loss:.4f})")
    
    print(f"\n✓ Training complete. Best loss: {best_loss:.4f}")
    print(f"✓ Model saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
