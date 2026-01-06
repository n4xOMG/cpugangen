#!/usr/bin/env python3
"""
Knowledge Distillation Training Script for Anime-Specialized Student UNet

Trains a smaller student UNet (SSD-1B architecture) using Illustrious as teacher.
Uses layer-level feature matching and output matching for distillation.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from tqdm.auto import tqdm
import wandb

from dataset_anime import AnimeDistillationDataset, collate_fn


class DistillationLoss(nn.Module):
    """
    Multi-level distillation loss combining:
    1. Output matching (MSE between final outputs)
    2. Layer-level feature matching (MSE between intermediate features)
    3. Optional attention map matching
    """
    
    def __init__(
        self,
        output_weight: float = 1.0,
        feature_weight: float = 0.5,
        attention_weight: float = 0.1,
        use_attention: bool = False
    ):
        super().__init__()
        self.output_weight = output_weight
        self.feature_weight = feature_weight
        self.attention_weight = attention_weight
        self.use_attention = use_attention
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_features: Optional[Dict] = None,
        teacher_features: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.
        
        Args:
            student_output: Student UNet output
            teacher_output: Teacher UNet output (detached)
            student_features: Dict of intermediate features from student
            teacher_features: Dict of intermediate features from teacher
            
        Returns:
            Dict with total loss and component losses
        """
        losses = {}
        
        # 1. Output matching loss
        output_loss = F.mse_loss(student_output, teacher_output)
        losses['output'] = output_loss
        
        # 2. Feature matching loss (if features provided)
        if student_features and teacher_features:
            feature_loss = 0.0
            num_features = 0
            
            for key in student_features.keys():
                if key in teacher_features:
                    s_feat = student_features[key]
                    t_feat = teacher_features[key]
                    
                    # Match feature dimensions if needed
                    if s_feat.shape != t_feat.shape:
                        # Interpolate to match spatial dimensions
                        s_feat = F.interpolate(
                            s_feat, 
                            size=t_feat.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    feature_loss += F.mse_loss(s_feat, t_feat)
                    num_features += 1
            
            if num_features > 0:
                feature_loss = feature_loss / num_features
                losses['feature'] = feature_loss
        
        # 3. Total loss
        total_loss = (
            self.output_weight * losses.get('output', 0.0) +
            self.feature_weight * losses.get('feature', 0.0)
        )
        losses['total'] = total_loss
        
        return losses


def encode_prompts(
    prompts: list[str],
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text prompts using SDXL's dual text encoders.
    
    Returns:
        (prompt_embeds, pooled_prompt_embeds)
    """
    # Tokenize
    tokens_1 = tokenizer_1(
        prompts,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    
    tokens_2 = tokenizer_2(
        prompts,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    
    # Encode
    with torch.no_grad():
        encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
        encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)
        
        # Get last hidden states
        prompt_embeds_1 = encoder_output_1.hidden_states[-2]  # penultimate layer
        prompt_embeds_2 = encoder_output_2.hidden_states[-2]
        
        # Concatenate
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        # Get pooled embeddings from second encoder
        pooled_prompt_embeds = encoder_output_2[0]
    
    return prompt_embeds, pooled_prompt_embeds


def train_one_epoch(
    teacher_unet: UNet2DConditionModel,
    student_unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    distillation_loss: DistillationLoss,
    device: torch.device,
    epoch: int,
    config: Dict
) -> float:
    """Train for one epoch."""
    
    student_unet.train()
    teacher_unet.eval()  # Teacher always in eval mode
    
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
        # Get batch
        pixel_values = batch.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        prompts = batch['prompts']
        
        # Encode prompts
        prompt_embeds, pooled_prompt_embeds = encode_prompts(
            prompts,
            text_encoder_1,
            text_encoder_2,
            tokenizer_1,
            tokenizer_2,
            device
        )
        
        # Get latents (either from batch or encode on-the-fly)
        if 'latents' in batch:
            # Using cached latents
            latents = batch['latents'].to(device)
        else:
            # Encode images to latents (original behavior)
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, 
            (batch_size,), 
            device=device
        ).long()
        
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Prepare added_cond_kwargs for SDXL
        add_time_ids = torch.tensor([
            [1024, 1024,  # original_size
             0, 0,        # crops_coords_top_left
             1024, 1024]  # target_size
        ]).repeat(batch_size, 1).to(device)
        
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids
        }
        
        # Forward pass with mixed precision
        with autocast():
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_output = teacher_unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]
            
            # Student forward
            student_output = student_unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
            
            # Compute distillation loss
            losses = distillation_loss(
                student_output,
                teacher_output.detach()
            )
            
            loss = losses['total']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        if config['training']['max_grad_norm'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                student_unet.parameters(), 
                config['training']['max_grad_norm']
            )
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Logging
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'output_loss': f"{losses.get('output', 0):.4f}",
            'feature_loss': f"{losses.get('feature', 0):.4f}"
        })
        
        # Log to wandb
        if step % config['logging']['log_every'] == 0:
            wandb.log({
                'train/loss': loss.item(),
                'train/output_loss': losses.get('output', 0).item(),
                'train/feature_loss': losses.get('feature', 0).item(),
                'train/epoch': epoch,
                'train/step': step
            })
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    teacher_unet: UNet2DConditionModel,
    student_unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    noise_scheduler: DDPMScheduler,
    dataloader: DataLoader,
    distillation_loss: DistillationLoss,
    device: torch.device,
    epoch: int
) -> float:
    """Validate the student model."""
    
    student_unet.eval()
    teacher_unet.eval()
    
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Validating"):
        pixel_values = batch['pixel_values'].to(device)
        prompts = batch['prompts']
        
        # Similar to training but without gradient computation
        prompt_embeds, pooled_prompt_embeds = encode_prompts(
            prompts, text_encoder_1, text_encoder_2,
            tokenizer_1, tokenizer_2, device
        )
        
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        add_time_ids = torch.tensor([
            [1024, 1024, 0, 0, 1024, 1024]
        ]).repeat(batch_size, 1).to(device)
        
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids
        }
        
        teacher_output = teacher_unet(
            noisy_latents, timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]
        
        student_output = student_unet(
            noisy_latents, timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]
        
        losses = distillation_loss(student_output, teacher_output)
        total_loss += losses['total'].item()
    
    avg_loss = total_loss / len(dataloader)
    
    wandb.log({
        'val/loss': avg_loss,
        'val/epoch': epoch
    })
    
    return avg_loss


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['wandb_project'],
            name=config['logging']['run_name'],
            config=config
        )
    
    # Create output directory
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    
    # Load teacher pipeline (Illustrious)
    print(f"Loading teacher model: {config['teacher']['model_id']}")
    teacher_pipe = StableDiffusionXLPipeline.from_pretrained(
        config['teacher']['model_id'],
        torch_dtype=torch.float32,  # Train in FP32, inference can use FP16
    )
    
    teacher_unet = teacher_pipe.unet.to(device)
    teacher_unet.requires_grad_(False)  # Freeze teacher
    
    # Load student UNet (SSD-1B architecture)
    print(f"Loading student model: {config['student']['model_id']}")
    student_pipe = StableDiffusionXLPipeline.from_pretrained(
        config['student']['model_id'],
        torch_dtype=torch.float32,
    )
    student_unet = student_pipe.unet.to(device)
    
    # Share VAE and text encoders from teacher
    vae = teacher_pipe.vae.to(device)
    vae.requires_grad_(False)
    
    text_encoder_1 = teacher_pipe.text_encoder.to(device)
    text_encoder_1.requires_grad_(False)
    
    text_encoder_2 = teacher_pipe.text_encoder_2.to(device)
    text_encoder_2.requires_grad_(False)
    
    tokenizer_1 = teacher_pipe.tokenizer
    tokenizer_2 = teacher_pipe.tokenizer_2
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        config['teacher']['model_id'],
        subfolder="scheduler"
    )
    
    # Print model sizes
    teacher_params = sum(p.numel() for p in teacher_unet.parameters())
    student_params = sum(p.numel() for p in student_unet.parameters())
    print(f"\nTeacher UNet: {teacher_params:,} parameters")
    print(f"Student UNet: {student_params:,} parameters")
    print(f"Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    
    # Load datasets based on config
    use_cached = config['data'].get('use_cached_latents', False)
    
    if use_cached:
        print("Using cached latents (faster training)...")
        from dataset_anime import CachedLatentDataset, collate_fn_cached
        
        train_dataset = CachedLatentDataset(
            latent_metadata_path=config['data']['train_latent_metadata'],
            latents_dir=config['data']['train_latents_dir'],
            max_tags=config['data']['max_tags']
        )
        
        val_dataset = CachedLatentDataset(
            latent_metadata_path=config['data']['val_latent_metadata'],
            latents_dir=config['data']['val_latents_dir'],
            max_tags=config['data']['max_tags']
        )
        
        batch_collate_fn = collate_fn_cached
    else:
        print("Using on-the-fly image encoding...")
        train_dataset = AnimeDistillationDataset(
            metadata_path=config['data']['train_metadata'],
            images_dir=config['data']['images_dir'],
            resolution=config['data']['resolution'],
            max_tags=config['data']['max_tags'],
            min_tag_score=config['data']['min_tag_score']
        )
        
        val_dataset = AnimeDistillationDataset(
            metadata_path=config['data']['val_metadata'],
            images_dir=config['data']['images_dir'],
            resolution=config['data']['resolution'],
            max_tags=config['data']['max_tags'],
            min_tag_score=config['data']['min_tag_score']
        )
        
        batch_collate_fn = collate_fn
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=batch_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=batch_collate_fn,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Setup training
    distillation_loss = DistillationLoss(
        output_weight=config['loss']['output_weight'],
        feature_weight=config['loss']['feature_weight']
    )
    
    optimizer = torch.optim.AdamW(
        student_unet.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scaler = GradScaler()
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_one_epoch(
            teacher_unet, student_unet, vae,
            text_encoder_1, text_encoder_2,
            tokenizer_1, tokenizer_2,
            noise_scheduler, train_loader,
            optimizer, scaler, distillation_loss,
            device, epoch, config
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(
            teacher_unet, student_unet, vae,
            text_encoder_1, text_encoder_2,
            tokenizer_1, tokenizer_2,
            noise_scheduler, val_loader,
            distillation_loss, device, epoch
        )
        
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                config['training']['output_dir'],
                "best_student_unet.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"✅ Saved best model! Val loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['training']['output_dir'],
                f"student_unet_epoch{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    if config['logging']['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train anime-specialized student UNet")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/distillation_config.json',
        help='Path to config file'
    )
    args = parser.parse_args()
    
    main(args)
