"""
Training script for Tag-Optimized Encoder (TOE)
Knowledge distillation from SDXL CLIP encoders
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.toe import TagOptimizedEncoder
from hqpd.utils.danbooru import DanbooruTagProcessor


class DanbooruDataset(Dataset):
    """
    Dataset for Danbooru images with tags and pre-computed CLIP embeddings.
    
    Updated to support real vocabulary and tag processing.
    Can load pre-computed CLIP embeddings or use dummy data for testing.
    """
    def __init__(self, num_samples=10000, vocab_size=15000, max_tags=77, 
                 vocabulary_path=None, tag_processor=None, clip_embeddings_path=None):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_tags = max_tags
        self.use_real_embeddings = False
        self.embeddings_data = None
        self.tag_combinations = None
        
        # Load vocabulary if provided
        if tag_processor is not None:
            self.tag_processor = tag_processor
            print(f"Using provided tag processor with {len(tag_processor.tag_to_id)} tags")
        elif vocabulary_path is not None:
            from pathlib import Path
            vocab_path = Path(vocabulary_path)
            if vocab_path.exists():
                self.tag_processor = DanbooruTagProcessor(vocab_size=vocab_size)
                self.tag_processor.load_vocabulary(str(vocab_path))
                print(f"Loaded vocabulary from {vocab_path}")
            else:
                print(f"WARNING: Vocabulary not found at {vocabulary_path}")
                print(f"Using dummy tag processor")
                self.tag_processor = None
        else:
            print(f"No vocabulary provided, using dummy data")
            self.tag_processor = None
        
        # Load pre-computed CLIP embeddings if provided
        if clip_embeddings_path is not None:
            from pathlib import Path
            import json
            
            embeddings_dir = Path(clip_embeddings_path)
            embeddings_file = embeddings_dir / "clip_embeddings.npz"
            metadata_file = embeddings_dir / "metadata.json"
            
            if embeddings_file.exists() and metadata_file.exists():
                print(f"Loading pre-computed CLIP embeddings from {embeddings_dir}")
                
                # Load embeddings
                self.embeddings_data = np.load(str(embeddings_file))['embeddings']
                print(f"  Loaded {self.embeddings_data.shape[0]} embeddings")
                
                # Load metadata
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.tag_combinations = metadata['tag_combinations']
                
                self.use_real_embeddings = True
                self.num_samples = min(num_samples, len(self.embeddings_data))
                print(f"  Using {self.num_samples} samples with REAL CLIP embeddings")
            else:
                print(f"WARNING: CLIP embeddings not found at {embeddings_dir}")
                print(f"  Expected: {embeddings_file}")
                print(f"  Run: python scripts/precompute_clip_embeddings.py")
                print(f"  Falling back to DUMMY embeddings")
        
        # In a real implementation, load actual tags and CLIP embeddings here
        print(f"Initialized dataset with {self.num_samples} samples")
        if not self.use_real_embeddings:
            print("  (DUMMY CLIP embeddings)")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load real embeddings if available
        if self.use_real_embeddings:
            # Get pre-computed embedding
            clip_embedding = torch.from_numpy(self.embeddings_data[idx]).float()
            
            # Get corresponding tags
            tags = self.tag_combinations[idx]
            
            # Encode tags using tag processor
            if self.tag_processor is not None:
                tag_ids, tag_weights = self.tag_processor.encode(tags, max_length=self.max_tags)
                tag_ids = torch.tensor(tag_ids, dtype=torch.long)
                tag_weights = torch.tensor(tag_weights, dtype=torch.float32)
            else:
                # Fallback to dummy tag encoding (shouldn't happen)
                tag_ids = torch.randint(3, self.vocab_size, (self.max_tags,))
                tag_weights = torch.ones(self.max_tags)
                tag_weights[len(tags):] = 0.0
        else:
            # Generate or load tags (dummy mode)
            if self.tag_processor is not None:
                # Use common tags from vocabulary for more realistic dummy data
                common_tags = ['1girl', 'solo', 'long_hair', 'smile', 'looking_at_viewer',
                              'blue_eyes', 'blush', 'breasts', 'open_mouth', 'brown_hair']
                
                # Randomly sample some tags
                import random
                num_tags = random.randint(3, 12)
                tags = random.sample(common_tags, min(num_tags, len(common_tags)))
                
                # Encode using tag processor
                tag_ids, tag_weights = self.tag_processor.encode(tags, max_length=self.max_tags)
                tag_ids = torch.tensor(tag_ids, dtype=torch.long)
                tag_weights = torch.tensor(tag_weights, dtype=torch.float32)
            else:
                # Fallback to completely dummy data
                num_tags = torch.randint(5, self.max_tags, (1,)).item()
                tag_ids = torch.randint(3, self.vocab_size, (self.max_tags,))
                tag_weights = torch.ones(self.max_tags)
                tag_weights[num_tags:] = 0.0  # Zero out padding
            
            # Dummy CLIP embedding (random noise)
            clip_embedding = torch.randn(77, 2048)
        
        return {
            'tag_ids': tag_ids,
            'tag_weights': tag_weights,
            'clip_embedding': clip_embedding,
            'idx': idx
        }


class KnowledgeDistillationLoss(nn.Module):
    """Loss function for distilling from CLIP to TOE."""
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, student_output, teacher_output):
        """
        Args:
            student_output: (batch, seq_len, dim) from TOE
            teacher_output: (batch, seq_len, dim) from CLIP
        """
        if self.loss_type == 'mse':
            return F.mse_loss(student_output, teacher_output)
        elif self.loss_type == 'cosine':
            # Cosine similarity loss
            return 1 - F.cosine_similarity(
                student_output.reshape(-1, student_output.size(-1)),
                teacher_output.reshape(-1, teacher_output.size(-1))
            ).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


def train_epoch(model, dataloader, optimizer, criterion, scaler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        tag_ids = batch['tag_ids'].to(device)
        tag_weights = batch['tag_weights'].to(device)
        clip_embedding = batch['clip_embedding'].to(device)
        
        # Forward pass with mixed precision
        with autocast(enabled=config['training']['mixed_precision']):
            toe_output = model(tag_ids, tag_weights)
            loss = criterion(toe_output, clip_embedding)
            
            # Scale loss for gradient accumulation
            loss = loss / config['training']['gradient_accumulation_steps']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights every N steps
        if (batch_idx + 1) % config['training']['gradient_accumulation_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Logging
        total_loss += loss.item() * config['training']['gradient_accumulation_steps']
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, criterion, device, config):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Validating"):
        tag_ids = batch['tag_ids'].to(device)
        tag_weights = batch['tag_weights'].to(device)
        clip_embedding = batch['clip_embedding'].to(device)
        
        toe_output = model(tag_ids, tag_weights)
        loss = criterion(toe_output, clip_embedding)
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    # Initialize model
    model = TagOptimizedEncoder(
        vocab_size=config['model']['vocab_size'],
        embed_dim=config['model']['embed_dim'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        mlp_ratio=config['model']['mlp_ratio'],
        max_length=config['model']['max_length'],
        quantize_embeddings=config['model']['quantize_embeddings']
    ).to(device)
    
    print(f"Model initialized: {model.get_num_params():,} parameters")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    # Load tag processor with vocabulary
    tag_processor = None
    vocab_path = Path(__file__).parent.parent / "data" / "vocabulary.json"
    if vocab_path.exists():
        print(f"\nLoading vocabulary from {vocab_path}")
        tag_processor = DanbooruTagProcessor(vocab_size=config['model']['vocab_size'])
        tag_processor.load_vocabulary(str(vocab_path))
    else:
        print(f"\nWARNING: Vocabulary not found at {vocab_path}")
        print(f"Run: python scripts/build_vocabulary.py")
    
    # Get CLIP embeddings path from config (optional)
    clip_embeddings_path = config['data'].get('clip_embeddings_path', None)
    if clip_embeddings_path:
        clip_embeddings_path = Path(__file__).parent.parent / clip_embeddings_path
    
    # Create datasets
    train_dataset = DanbooruDataset(
        num_samples=int(config['data']['num_samples'] * config['data']['train_split']),
        vocab_size=config['model']['vocab_size'],
        max_tags=config['data']['max_tags_per_image'],
        tag_processor=tag_processor,
        clip_embeddings_path=clip_embeddings_path
    )
    val_dataset = DanbooruDataset(
        num_samples=int(config['data']['num_samples'] * config['data']['val_split']),
        vocab_size=config['model']['vocab_size'],
        max_tags=config['data']['max_tags_per_image'],
        tag_processor=tag_processor,
        clip_embeddings_path=clip_embeddings_path
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Setup training
    criterion = KnowledgeDistillationLoss(loss_type=config['training']['distillation_loss'])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    scaler = GradScaler(enabled=config['training']['mixed_precision'])
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, config)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, config)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save ONLY best model (removed periodic checkpoints)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(
                config['training']['checkpoint_dir'],
                "toe_best.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, best_path)
            print(f"New best model! Val loss: {val_loss:.4f}")
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tag-Optimized Encoder")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/toe_config.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    main(args)
