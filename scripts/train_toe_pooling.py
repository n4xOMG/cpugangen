"""
Train pooling head for TOE to enable full TOE mode (no CLIP dependency).

This script:
1. Loads pre-trained TOE model (frozen)
2. Adds pooling head (trainable)
3. Uses CLIP pooled embeddings as supervision
4. Trains only the pooling head for ~30-60 minutes

Usage:
    python scripts/train_toe_pooling.py --checkpoint checkpoints/toe/toe_best.pt
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.toe import TagOptimizedEncoder
from hqpd.utils.danbooru import DanbooruTagProcessor


class PoolingDataset(Dataset):
    """Dataset for training pooling head using CLIP embeddings."""
    
    def __init__(self, embeddings_file):
        """
        Args:
            embeddings_file: Path to pre-computed embeddings
                Format: {
                    'tags': List[List[str]],
                    'clip_pooled': torch.Tensor (N, 1280)
                }
        """
        print(f"Loading embeddings from {embeddings_file}...")
        data = torch.load(embeddings_file)
        
        self.tags_list = data['tags']
        self.clip_pooled = data['clip_pooled']
        
        print(f"Loaded {len(self.tags_list)} samples")
    
    def __len__(self):
        return len(self.tags_list)
    
    def __getitem__(self, idx):
        return {
            'tags': self.tags_list[idx],
            'clip_pooled': self.clip_pooled[idx]
        }


def collate_fn(batch, tag_processor):
    """Collate batch and encode tags."""
    tags_batch = [item['tags'] for item in batch]
    clip_pooled_batch = torch.stack([item['clip_pooled'] for item in batch])
    
    # Encode all tags
    tag_ids_batch = []
    tag_weights_batch = []
    
    for tags in tags_batch:
        tag_ids, tag_weights = tag_processor.encode(tags, max_length=77)
        tag_ids_batch.append(tag_ids)
        tag_weights_batch.append(tag_weights)
    
    return {
        'tag_ids': torch.tensor(tag_ids_batch, dtype=torch.long),
        'tag_weights': torch.tensor(tag_weights_batch, dtype=torch.float32),
        'clip_pooled': clip_pooled_batch
    }


def train_pooling_head(
    checkpoint_path: str,
    vocab_path: str,
    embeddings_path: str,
    output_dir: str = "checkpoints/toe",
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    device: str = "cuda"
):
    """
    Train pooling head for TOE.
    
    Args:
        checkpoint_path: Path to pre-trained TOE checkpoint
        vocab_path: Path to vocabulary
        embeddings_path: Path to CLIP pooled embeddings
        output_dir: Where to save trained model
        batch_size: Training batch size
        learning_rate: Learning rate
        num_epochs: Number of epochs
        device: Device to use
    """
    
    print("=" * 70)
    print("Training TOE Pooling Head")
    print("=" * 70)
    
    # Load tag processor
    print("\n1. Loading vocabulary...")
    tag_processor = DanbooruTagProcessor(vocab_size=15000)
    tag_processor.load_vocabulary(vocab_path)
    print(f"   ✓ Loaded {len(tag_processor.tag_to_id)} tags")
    
    # Load pre-trained TOE and enable pooling
    print("\n2. Loading pre-trained TOE...")
    toe_model = TagOptimizedEncoder(
        vocab_size=15000,
        embed_dim=2048,
        num_layers=4,
        num_heads=8,
        mlp_ratio=2,
        max_length=77,
        quantize_embeddings=False,
        enable_pooling=True  # Enable pooling head
    ).to(device)
    
    # Load weights (without pooling head)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict, ignoring pooling head (new component)
    model_state = checkpoint['model_state_dict']
    missing_keys, unexpected_keys = toe_model.load_state_dict(model_state, strict=False)
    
    print(f"   ✓ TOE loaded: {toe_model.get_num_params():,} parameters")
    print(f"   ✓ New pooling head added")
    if missing_keys:
        print(f"   → New keys (pooling head): {missing_keys}")
    
    # Freeze everything except pooling head
    for name, param in toe_model.named_parameters():
        if 'pooling_head' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in toe_model.parameters() if p.requires_grad)
    print(f"   ✓ Trainable parameters: {trainable_params:,} (pooling head only)")
    
    # Load dataset
    print(f"\n3. Loading training data from {embeddings_path}...")
    dataset = PoolingDataset(embeddings_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tag_processor),
        num_workers=0  # Can increase if needed
    )
    
    # Training setup
    print("\n4. Setting up training...")
    optimizer = torch.optim.AdamW(
        [p for p in toe_model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Combined loss: MSE + Cosine similarity
    def pooling_loss(pred_pooled, target_pooled):
        mse = F.mse_loss(pred_pooled, target_pooled)
        cosine = 1 - F.cosine_similarity(pred_pooled, target_pooled).mean()
        return mse + 0.5 * cosine
    
    # Training loop
    print(f"\n5. Training for {num_epochs} epochs...")
    print("=" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        toe_model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_cosine = 0.0
        
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress:
            tag_ids = batch['tag_ids'].to(device)
            tag_weights = batch['tag_weights'].to(device)
            target_pooled = batch['clip_pooled'].to(device).float()  # Convert to float32
            
            # Forward
            _, pred_pooled = toe_model(tag_ids, tag_weights, return_pooled=True)
            
            # Loss
            loss = pooling_loss(pred_pooled, target_pooled)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                mse = F.mse_loss(pred_pooled, target_pooled).item()
                cosine_sim = F.cosine_similarity(pred_pooled, target_pooled).mean().item()
            
            epoch_loss += loss.item()
            epoch_mse += mse
            epoch_cosine += cosine_sim
            
            progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse:.4f}',
                'cos': f'{cosine_sim:.3f}'
            })
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_mse = epoch_mse / len(dataloader)
        avg_cosine = epoch_cosine / len(dataloader)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  MSE: {avg_mse:.4f}")
        print(f"  Cosine Similarity: {avg_cosine:.3f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            output_path = Path(output_dir) / "toe_with_pooling_best.pt"
            os.makedirs(output_dir, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': toe_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'mse': avg_mse,
                'cosine_similarity': avg_cosine,
            }, output_path)
            
            print(f"  ✓ Best model saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("✓ Training Complete!")
    print("=" * 70)
    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Saved to: {output_dir}/toe_with_pooling_best.pt")
    print("\nTo use full TOE mode:")
    print("  python scripts/integrate_toe_full.py --hybrid=False \\")
    print(f"    --checkpoint {output_dir}/toe_with_pooling_best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train TOE pooling head")
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/toe/toe_best.pt',
        help='Path to pre-trained TOE checkpoint'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default='data/vocabulary.json',
        help='Path to vocabulary'
    )
    parser.add_argument(
        '--embeddings',
        type=str,
        default='data/clip_pooled_embeddings.pt',
        help='Path to CLIP pooled embeddings dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/toe',
        help='Output directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of epochs'
    )
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    train_pooling_head(
        checkpoint_path=args.checkpoint,
        vocab_path=args.vocab,
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        device=device
    )


if __name__ == "__main__":
    main()
