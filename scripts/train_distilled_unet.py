#!/usr/bin/env python3
"""
Train distilled UNet from SDXL teacher.

This script:
1. Loads full SDXL UNet as teacher
2. Creates distilled UNet as student (Segmind SSD-1B style)
3. Transfers matching weights
4. Distills knowledge from teacher to student

Usage:
    python scripts/train_distilled_unet.py \
        --data_dir data/latents \
        --output_dir checkpoints/distilled_unet
        
For quick start with pre-trained Segmind:
    python scripts/train_distilled_unet.py --use_segmind
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.distillation import (
    create_distilled_unet,
    transfer_weights_from_teacher,
    DistilledUNetWrapper,
    load_segmind_ssd1b,
    DistillationLoss,
    ProgressiveDistillationLoss,
    DistillationTrainer,
    TrainingConfig,
)


class LatentDataset(Dataset):
    """
    Dataset of pre-computed latents for distillation.
    
    Expected format:
    - latent_{idx}.pt containing:
        - latent: (4, H, W) noise latent
        - encoder_hidden_states: (77, 2048) text embeddings
        - text_embeds: (1280,) pooled embeddings
        - time_ids: (6,) SDXL time conditioning
    """
    
    def __init__(self, data_dir: str, max_samples: Optional[int] = None):
        self.data_dir = Path(data_dir)
        
        # Find all latent files
        self.files = sorted(self.data_dir.glob("latent_*.pt"))
        
        if max_samples:
            self.files = self.files[:max_samples]
        
        print(f"Found {len(self.files)} latent files")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        
        # Sample random timestep
        timestep = torch.randint(0, 1000, (1,)).item()
        
        return {
            "latents": data["latent"],
            "timesteps": torch.tensor([timestep]),
            "encoder_hidden_states": data["encoder_hidden_states"],
            "text_embeds": data["text_embeds"],
            "time_ids": data["time_ids"],
        }


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing (no real data needed).
    
    Generates random latents and embeddings.
    """
    
    def __init__(self, num_samples: int = 10000, latent_size: int = 128):
        self.num_samples = num_samples
        self.latent_size = latent_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random latent
        latent = torch.randn(4, self.latent_size, self.latent_size)
        
        # Random timestep
        timestep = torch.randint(0, 1000, (1,))
        
        # Random embeddings
        encoder_hidden_states = torch.randn(77, 2048)
        text_embeds = torch.randn(1280)
        time_ids = torch.randn(6)
        
        return {
            "latents": latent,
            "timesteps": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }


def collate_fn(batch):
    """Collate batch samples."""
    return {
        "latents": torch.stack([b["latents"] for b in batch]),
        "timesteps": torch.cat([b["timesteps"] for b in batch]),
        "encoder_hidden_states": torch.stack([b["encoder_hidden_states"] for b in batch]),
        "text_embeds": torch.stack([b["text_embeds"] for b in batch]),
        "time_ids": torch.stack([b["time_ids"] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser(description="Train distilled UNet")
    
    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/latents",
        help="Directory with latent files",
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    parser.add_argument(
        "--num_synthetic",
        type=int,
        default=10000,
        help="Number of synthetic samples",
    )
    
    # Model
    parser.add_argument(
        "--use_segmind",
        action="store_true",
        help="Use pre-trained Segmind SSD-1B as starting point",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Teacher model (HuggingFace ID or path)",
    )
    
    # Training
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per step",
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/distilled_unet",
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("UNet Distillation Training")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Teacher: {args.teacher_model}")
    print(f"Use Segmind: {args.use_segmind}")
    print(f"Max steps: {args.max_steps}")
    print("=" * 70)
    
    # Load teacher
    print("\n1. Loading teacher UNet...")
    from diffusers import UNet2DConditionModel
    
    teacher = UNet2DConditionModel.from_pretrained(
        args.teacher_model,
        subfolder="unet",
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"   ✓ Teacher: {teacher_params:,} parameters ({teacher_params/1e9:.2f}B)")
    
    # Create or load student
    print("\n2. Creating student UNet...")
    if args.use_segmind:
        print("   Using pre-trained Segmind SSD-1B")
        student_unet = load_segmind_ssd1b(device="cpu")
    else:
        print("   Creating distilled architecture")
        student_unet = create_distilled_unet()
        
        # Transfer weights from teacher
        print("   Transferring weights from teacher...")
        transferred, missing = transfer_weights_from_teacher(teacher, student_unet)
    
    student_params = sum(p.numel() for p in student_unet.parameters())
    print(f"   ✓ Student: {student_params:,} parameters ({student_params/1e9:.2f}B)")
    print(f"   ✓ Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # Wrap student for feature extraction
    student = DistilledUNetWrapper(student_unet, enable_feature_extraction=True)
    
    # Create dataset
    print("\n3. Setting up data...")
    if args.use_synthetic:
        print("   Using synthetic data for testing")
        dataset = SyntheticDataset(num_samples=args.num_synthetic)
    else:
        if not Path(args.data_dir).exists():
            print(f"   ⚠️  Data dir '{args.data_dir}' not found, using synthetic")
            dataset = SyntheticDataset(num_samples=args.num_synthetic)
        else:
            dataset = LatentDataset(args.data_dir)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if args.device == "cuda" else False,
    )
    print(f"   ✓ Dataset: {len(dataset)} samples")
    
    # Create loss function
    print("\n4. Setting up training...")
    loss_fn = ProgressiveDistillationLoss(
        max_steps=args.max_steps,
        output_weight=1.0,
        feature_weight=0.5,
        cosine_weight=0.1,
    )
    
    # Training config
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_steps=args.max_steps,
        enable_gradient_checkpointing=True,
        enable_xformers=True,
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        loss_fn=loss_fn,
        config=config,
        output_dir=args.output_dir,
        device=args.device,
    )
    
    # Train
    print("\n5. Starting training...")
    trainer.train(dataloader, resume_from=args.resume)
    
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)
    print(f"\nCheckpoints saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Evaluate with: python scripts/eval_distilled_unet.py")
    print("  2. Quantize with: python scripts/quantize_unet.py")
    print("  3. Integrate with pipeline")


if __name__ == "__main__":
    main()

