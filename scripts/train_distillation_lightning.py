#!/usr/bin/env python3
"""
Knowledge Distillation Training Script with Lightning-Enabled Teacher

Trains a smaller student UNet (SSD-1B architecture) using Illustrious + Lightning as teacher.
This version applies Lightning LoRA to the TEACHER so the student learns Lightning behavior.

Key differences from train_distillation.py:
- Teacher has Lightning LoRA applied BEFORE distillation
- Uses Lightning scheduler (EulerDiscrete) instead of DDPM
- Trains on Lightning-specific timestep distribution (few steps)
- Student learns to work with 4-step inference from the start
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
    DDPMScheduler,
    EulerDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import wandb
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_anime import AnimeDistillationDataset, collate_fn
from feature_extractor import UNetFeatureExtractor, get_default_feature_layers


class DistillationLoss(nn.Module):
    """
    Multi-level distillation loss combining:
    1. Output matching (MSE between final outputs)
    2. Layer-level feature matching (MSE between intermediate features)
    """
    
    def __init__(
        self,
        output_weight: float = 1.0,
        feature_weight: float = 0.5,
        normalize_features: bool = True
    ):
        super().__init__()
        self.output_weight = output_weight
        self.feature_weight = feature_weight
        self.normalize_features = normalize_features
        
        # Feature aligner for dimension mismatches
        from feature_extractor import FeatureAligner
        self.feature_aligner = FeatureAligner()
    
    def _normalize_feature(self, feat: torch.Tensor) -> torch.Tensor:
        """L2-normalize feature maps channel-wise."""
        norm = torch.norm(feat, p=2, dim=1, keepdim=True) + 1e-8
        return feat / norm
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_features: Optional[Dict] = None,
        teacher_features: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss."""
        losses = {}
        
        # 1. Output matching loss
        output_loss = F.mse_loss(student_output, teacher_output)
        losses['output'] = output_loss
        
        # 2. Feature matching loss (if features provided)
        if student_features is not None and teacher_features is not None:
            feature_loss = 0.0
            num_features = 0
            
            for key in student_features.keys():
                if key in teacher_features:
                    s_feat = student_features[key]
                    t_feat = teacher_features[key]
                    
                    # Normalize features if requested
                    if self.normalize_features:
                        s_feat = self._normalize_feature(s_feat)
                        t_feat = self._normalize_feature(t_feat)
                    
                    # Align feature dimensions
                    s_feat_aligned = self.feature_aligner(s_feat, t_feat, key)
                    
                    # Compute MSE loss
                    feature_loss += F.mse_loss(s_feat_aligned, t_feat)
                    num_features += 1
            
            if num_features > 0:
                feature_loss = feature_loss / num_features
                losses['feature'] = feature_loss
            else:
                losses['feature'] = torch.tensor(0.0, device=student_output.device)
        else:
            losses['feature'] = torch.tensor(0.0, device=student_output.device)
        
        # 3. Total loss
        total_loss = (
            self.output_weight * losses['output'] +
            self.feature_weight * losses['feature']
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
    """Encode text prompts using SDXL's dual text encoders."""
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
        prompt_embeds_1 = encoder_output_1.hidden_states[-2]
        prompt_embeds_2 = encoder_output_2.hidden_states[-2]
        
        # Concatenate
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        
        # Get pooled embeddings
        pooled_prompt_embeds = encoder_output_2[0]
    
    return prompt_embeds, pooled_prompt_embeds


def get_lightning_timesteps(batch_size: int, num_steps: int, device: torch.device) -> torch.Tensor:
    """
    Sample timesteps appropriate for Lightning training.
    
    Lightning uses a specific few-step schedule, so we need to sample
    timesteps that match the Lightning inference distribution.
    """
    # Lightning 4-step uses timesteps: [999, 749, 499, 249] approximately
    # We'll sample from a distribution weighted towards these values
    
    if num_steps == 4:
        # Exact Lightning 4-step timesteps
        possible_timesteps = torch.tensor([999, 749, 499, 249], device=device)
    elif num_steps == 8:
        # Lightning 8-step
        possible_timesteps = torch.tensor([999, 874, 749, 624, 499, 374, 249, 124], device=device)
    else:
        # Fallback: uniform sampling for other step counts
        possible_timesteps = torch.linspace(999, 0, num_steps, device=device).long()
    
    # Randomly sample from possible timesteps
    indices = torch.randint(0, len(possible_timesteps), (batch_size,), device=device)
    timesteps = possible_timesteps[indices]
    
    return timesteps


def train_one_epoch(
    teacher_unet: UNet2DConditionModel,
    student_unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    noise_scheduler: EulerDiscreteScheduler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    distillation_loss: DistillationLoss,
    device: torch.device,
    epoch: int,
    config: Dict
) -> float:
    """Train for one epoch with Lightning-enabled teacher."""
    
    student_unet.train()
    teacher_unet.eval()  # Teacher always in eval mode
    
    # Feature extractors
    feature_layers = get_default_feature_layers('sdxl')
    teacher_extractor = UNetFeatureExtractor(teacher_unet, feature_layers, normalize=False)
    student_extractor = UNetFeatureExtractor(student_unet, feature_layers, normalize=False)
    
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    # Lightning uses 4 steps by default
    lightning_steps = config['training'].get('lightning_steps', 4)
    
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
        
        # Get latents
        if 'latents' in batch:
            latents = batch['latents'].to(device)
        else:
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        
        # CRITICAL: Sample timesteps appropriate for Lightning
        timesteps = get_lightning_timesteps(batch_size, lightning_steps, device)
        
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
            # Teacher forward (no grad) with feature extraction
            with torch.no_grad():
                teacher_output = teacher_extractor.extract(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False
                )[0]
                teacher_features = teacher_extractor.get_features()
            
            # Student forward with feature extraction
            student_output = student_extractor.extract(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
            student_features = student_extractor.get_features()
            
            # Compute distillation loss with features
            losses = distillation_loss(
                student_output,
                teacher_output.detach(),
                student_features=student_features,
                teacher_features=teacher_features
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
        optimizer.zero_grad(set_to_none=True)
        
        # Clear feature caches
        teacher_extractor.clear()
        student_extractor.clear()
        
        # Logging
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'output_loss': f"{losses.get('output', 0):.4f}",
            'feature_loss': f"{losses.get('feature', 0):.4f}"
        })
        
        # Log to wandb
        if step % config['logging']['log_every'] == 0:
            def get_val(v):
                return v.item() if hasattr(v, 'item') else v

            wandb.log({
                'train/loss': loss.item(),
                'train/output_loss': get_val(losses.get('output', 0)),
                'train/feature_loss': get_val(losses.get('feature', 0)),
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
    noise_scheduler: EulerDiscreteScheduler,
    dataloader: DataLoader,
    distillation_loss: DistillationLoss,
    device: torch.device,
    epoch: int,
    config: Dict
) -> float:
    """Validate the student model."""
    
    student_unet.eval()
    teacher_unet.eval()
    
    # Feature extractors
    feature_layers = get_default_feature_layers('sdxl')
    teacher_extractor = UNetFeatureExtractor(teacher_unet, feature_layers, normalize=False)
    student_extractor = UNetFeatureExtractor(student_unet, feature_layers, normalize=False)
    
    total_loss = 0.0
    lightning_steps = config['training'].get('lightning_steps', 4)
    
    for batch in tqdm(dataloader, desc="Validating"):
        prompts = batch['prompts']
        
        # Encode prompts
        prompt_embeds, pooled_prompt_embeds = encode_prompts(
            prompts, text_encoder_1, text_encoder_2,
            tokenizer_1, tokenizer_2, device
        )

        # Get latents
        if 'latents' in batch:
            latents = batch['latents'].to(device)
        else:
            pixel_values = batch['pixel_values'].to(device)
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
        
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        
        # Use Lightning timesteps
        timesteps = get_lightning_timesteps(batch_size, lightning_steps, device)
        
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        add_time_ids = torch.tensor([
            [1024, 1024, 0, 0, 1024, 1024]
        ]).repeat(batch_size, 1).to(device)
        
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds,
            "time_ids": add_time_ids
        }
        
        with autocast():
            # Extract features from teacher
            teacher_output = teacher_extractor.extract(
                noisy_latents, timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
            teacher_features = teacher_extractor.get_features()
            
            # Extract features from student
            student_output = student_extractor.extract(
                noisy_latents, timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
            student_features = student_extractor.get_features()
        
        # Compute distillation loss
        losses = distillation_loss(
            student_output, 
            teacher_output,
            student_features=student_features,
            teacher_features=teacher_features
        )
        total_loss += losses['total'].item()
        
        # Clear feature caches
        teacher_extractor.clear()
        student_extractor.clear()
    
    avg_loss = total_loss / len(dataloader)
    
    # Log to wandb
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
            name=config['logging']['run_name'] + "_lightning",
            config=config
        )
    
    # Create output directory
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    print("\n" + "="*60)
    print("Loading Models (Lightning-Enabled Teacher)")
    print("="*60)
    
    # Load teacher pipeline (Illustrious) in FP16
    print(f"Loading teacher model: {config['teacher']['model_id']}")
    teacher_pipe = StableDiffusionXLPipeline.from_pretrained(
        config['teacher']['model_id'],
        torch_dtype=torch.float16,
    )
    
    # CRITICAL: Apply Lightning LoRA to TEACHER
    print("\n⚡ Applying Lightning LoRA to TEACHER...")
    lightning_ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
    teacher_pipe.load_lora_weights(load_file(lightning_ckpt))
    teacher_pipe.fuse_lora()
    print("  ✓ Teacher now has Lightning behavior!")
    
    # Configure Lightning scheduler for teacher
    teacher_pipe.scheduler = EulerDiscreteScheduler.from_config(
        teacher_pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    teacher_unet = teacher_pipe.unet.to(device)
    teacher_unet.requires_grad_(False)
    
    # Load student UNet (SSD-1B architecture)
    print(f"\nLoading student model: {config['student']['model_id']}")
    try:
        student_pipe = StableDiffusionXLPipeline.from_pretrained(
            config['student']['model_id'],
            torch_dtype=torch.float32,
        )
        student_unet = student_pipe.unet.to(device)
    except OSError:
        print("Pipeline load failed, trying generic UNet loading...")
        student_unet = UNet2DConditionModel.from_pretrained(
            config['student']['model_id'],
            torch_dtype=torch.float32
        ).to(device)
    
    # Enable optimizations
    student_unet.enable_gradient_checkpointing()
    print("Enabled gradient checkpointing for Student UNet")

    if torch.cuda.is_available():
        try:
            student_unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except Exception:
            print("xformers not found, using standard attention")

    # Load other components
    use_cached_latents = config['data'].get('use_cached_latents', False)
    
    if not use_cached_latents:
        vae = teacher_pipe.vae.to(device)
    else:
        vae = teacher_pipe.vae
        print("Using cached latents, VAE kept on CPU to save VRAM")

    vae.requires_grad_(False)
    
    text_encoder_1 = teacher_pipe.text_encoder.to(device)
    text_encoder_1.requires_grad_(False)
    
    text_encoder_2 = teacher_pipe.text_encoder_2.to(device)
    text_encoder_2.requires_grad_(False)
    
    tokenizer_1 = teacher_pipe.tokenizer
    tokenizer_2 = teacher_pipe.tokenizer_2
    
    # Use Lightning scheduler (NOT DDPM!)
    noise_scheduler = teacher_pipe.scheduler
    
    # Print model info
    teacher_params = sum(p.numel() for p in teacher_unet.parameters())
    student_params = sum(p.numel() for p in student_unet.parameters())
    print(f"\nTeacher UNet (+ Lightning): {teacher_params:,} parameters")
    print(f"Student UNet: {student_params:,} parameters")
    print(f"Reduction: {(1 - student_params/teacher_params)*100:.1f}%")
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    use_cached = config['data'].get('use_cached_latents', False)
    
    if use_cached:
        print("Using cached latents...")
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
    
    scaler = GradScaler()
    
    # Optimizer
    try:
        import bitsandbytes as bnb
        print("Using 8-bit AdamW optimizer")
        optimizer = bnb.optim.AdamW8bit(
            student_unet.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    except ImportError:
        print("Using standard AdamW")
        optimizer = torch.optim.AdamW(
            student_unet.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Lightning-Enabled Training")
    print("="*60)
    print("⚡ Student will learn 4-step Lightning behavior!")
    
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
            distillation_loss, device, epoch, config
        )
        
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                config['training']['output_dir'],
                "best_student_unet_lightning.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'lightning_enabled': True  # Marker for Lightning training
            }, checkpoint_path)
            print(f"✅ Saved best Lightning model! Val loss: {val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['training']['output_dir'],
                f"student_unet_lightning_epoch{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'lightning_enabled': True
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("✅ Lightning Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("\n⚡ Student is now compatible with Lightning LoRA!")
    print("   Use with 4-step inference + Lightning scheduler")
    
    if config['logging']['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train student UNet with Lightning-enabled teacher")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/distillation_config.json',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    main(args)
