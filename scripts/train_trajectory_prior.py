#!/usr/bin/env python3
"""
Train Trajectory Predictor

Trains the TrajectoryPredictor to learn universal denoising patterns.

Usage:
    python scripts/train_trajectory_prior.py \\
        --data data/trajectory_training \\
        --epochs 30 \\
        --output checkpoints/trajectory_prior.pt
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.optimization.trajectory_prior import (
    TrajectoryPredictor,
    TrajectoryDataset,
    TrajectoryTrainer,
)


def main():
    parser = argparse.ArgumentParser(description="Train trajectory predictor")
    parser.add_argument("--data", type=str, required=True, help="Training data directory")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output", type=str, default="checkpoints/trajectory_prior.pt")
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRAINING TRAJECTORY PREDICTOR")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = TrajectoryDataset(args.data)
    print(f"  Found {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    # Create model
    print("\nCreating model...")
    model = TrajectoryPredictor(
        in_channels=4,
        base_channels=64,
        t_emb_dim=256,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params * 4 / 1024 / 1024:.1f} MB)")
    
    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")
    
    # Create trainer
    trainer = TrajectoryTrainer(model=model, lr=args.lr, device=device)
    
    # Training loop
    print("\nTraining...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(dataloader, verbose=False)
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"loss={metrics['avg_loss']:.4f}, "
              f"mse={metrics['avg_loss_mse']:.4f}, "
              f"cos={metrics['avg_cos_sim']:.3f}")
        
        if metrics['avg_loss'] < best_loss:
            best_loss = metrics['avg_loss']
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path))
            print(f"  ✓ Saved (loss={best_loss:.4f})")
    
    print(f"\n✓ Training complete. Best loss: {best_loss:.4f}")
    print(f"✓ Model saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
