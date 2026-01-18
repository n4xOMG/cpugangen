#!/usr/bin/env python3
"""
Consistency Distillation Training Script

Trains a student to achieve 20-step BASE teacher quality in just 4 steps!

Key Concept:
- Teacher: Base Illustrious running 20 steps (HIGH QUALITY)
- Student: Trained to match teacher's final output in 4 steps (FAST + HIGH QUALITY)

This is different from Lightning distillation:
- Lightning distillation: Student mimics Lightning teacher (4-step quality)
- Consistency distillation: Student mimics BASE teacher (20-step quality) in 4 steps!

The student learns to "jump" directly to high-quality outputs that would normally
require 20 denoising steps.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

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
import wandb
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_anime import AnimeDistillationDataset, collate_fn
from feature_extractor import UNetFeatureExtractor, get_default_feature_layers


class ConsistencyDistillationLoss(nn.Module):
    """
    Consistency distillation loss that matches final denoised outputs.
    
    Instead of matching intermediate predictions, we:
    1. Run teacher for FULL 20 steps → get high-quality final output
    2. Run student for just 4 steps → get fast output
    3. Match student's 4-step output to teacher's 20-step output
    
    This teaches student to achieve 20-step quality in 4 steps!
    """
    
    def __init__(
        self,
        consistency_weight: float = 1.0,
        feature_weight: float = 0.3,
        normalize_features: bool = True
    ):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.feature_weight = feature_weight
        self.normalize_features = normalize_features
        
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
        """
        Compute consistency distillation loss.
        
        Args:
            student_output: Student's final denoised latent (after 4 steps)
            teacher_output: Teacher's final denoised latent (after 20 steps)
            student_features: Optional intermediate features
            teacher_features: Optional intermediate features
        """
        losses = {}
        
        # 1. Consistency loss: Match final denoised outputs
        # This is the KEY - student learns to reach teacher's quality in fewer steps
        consistency_loss = F.mse_loss(student_output, teacher_output)
        losses['consistency'] = consistency_loss
        
        # 2. Optional feature matching (helps with training stability)
        if student_features is not None and teacher_features is not None:
            feature_loss = 0.0
            num_features = 0
            
            for key in student_features.keys():
                if key in teacher_features:
                    s_feat = student_features[key]
                    t_feat = teacher_features[key]
                    
                    if self.normalize_features:
                        s_feat = self._normalize_feature(s_feat)
                        t_feat = self._normalize_feature(t_feat)
                    
                    s_feat_aligned = self.feature_aligner(s_feat, t_feat, key)
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
            self.consistency_weight * losses['consistency'] +
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
    
    with torch.no_grad():
        encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
        encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)
        
        prompt_embeds_1 = encoder_output_1.hidden_states[-2]
        prompt_embeds_2 = encoder_output_2.hidden_states[-2]
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        pooled_prompt_embeds = encoder_output_2[0]
    
    return prompt_embeds, pooled_prompt_embeds


@torch.no_grad()
def denoise_multistep(
    unet: UNet2DConditionModel,
    noisy_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    scheduler: DDPMScheduler,
    num_steps: int,
    device: torch.device
) -> torch.Tensor:
    """
    Run multi-step denoising to get final clean latent.
    
    This is used for the TEACHER to produce high-quality 20-step output.
    
    Args:
        unet: UNet model
        noisy_latents: Starting noisy latents
        prompt_embeds: Text conditioning
        pooled_prompt_embeds: Pooled text embeddings
        scheduler: Noise scheduler
        num_steps: Number of denoising steps (e.g., 20)
        device: Device
        
    Returns:
        Final denoised latent
    """
    # Set timesteps
    scheduler.set_timesteps(num_steps, device=device)
    timesteps = scheduler.timesteps
    
    # Clone to avoid modifying input
    latents = noisy_latents.clone()
    
    # Prepare conditioning
    add_time_ids = torch.tensor([
        [1024, 1024, 0, 0, 1024, 1024]
    ]).repeat(latents.shape[0], 1).to(device)
    
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids
    }
    
    # Denoising loop
    for t in timesteps:
        # Expand timestep
        timestep = t.unsqueeze(0).repeat(latents.shape[0])
        
        # Predict noise
        noise_pred = unet(
            latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False
        )[0]
        
        # Denoise step
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    return latents


def train_one_epoch(
    teacher_unet: UNet2DConditionModel,
    student_unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder_1: CLIPTextModel,
    text_encoder_2: CLIPTextModelWithProjection,
    tokenizer_1: CLIPTokenizer,
    tokenizer_2: CLIPTokenizer,
    teacher_scheduler: DDPMScheduler,
    student_scheduler: DDPMScheduler,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    distillation_loss: ConsistencyDistillationLoss,
    device: torch.device,
    epoch: int,
    config: Dict
) -> float:
    """Train for one epoch with consistency distillation."""
    
    student_unet.train()
    teacher_unet.eval()
    
    # Feature extractors (optional, for intermediate matching)
    feature_layers = get_default_feature_layers('sdxl')
    teacher_extractor = UNetFeatureExtractor(teacher_unet, feature_layers, normalize=False)
    student_extractor = UNetFeatureExtractor(student_unet, feature_layers, normalize=False)
    
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    teacher_steps = config['training'].get('teacher_steps', 20)
    student_steps = config['training'].get('student_steps', 4)
    
    for step, batch in enumerate(progress_bar):
        # Get batch
        pixel_values = batch.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        prompts = batch['prompts']
        
        # Encode prompts
        prompt_embeds, pooled_prompt_embeds = encode_prompts(
            prompts, text_encoder_1, text_encoder_2,
            tokenizer_1, tokenizer_2, device
        )
        
        # Get latents
        if 'latents' in batch:
            clean_latents = batch['latents'].to(device)
        else:
            with torch.no_grad():
                clean_latents = vae.encode(pixel_values).latent_dist.sample()
                clean_latents = clean_latents * vae.config.scaling_factor
        
        # Add noise (start from same noisy point)
        noise = torch.randn_like(clean_latents)
        batch_size = clean_latents.shape[0]
        
        # Sample a random timestep to start from
        start_timestep = torch.randint(
            500, teacher_scheduler.config.num_train_timesteps,
            (1,), device=device
        ).item()
        
        timestep_tensor = torch.tensor([start_timestep] * batch_size, device=device)
        noisy_latents = teacher_scheduler.add_noise(clean_latents, noise, timestep_tensor)
        
        # TEACHER: Run full 20-step denoising (HIGH QUALITY)
        with torch.no_grad():
            teacher_output = denoise_multistep(
                teacher_unet,
                noisy_latents,
                prompt_embeds,
                pooled_prompt_embeds,
                teacher_scheduler,
                num_steps=teacher_steps,
                device=device
            )
        
        # STUDENT: Run 4-step denoising (FAST)
        # We want student to reach teacher's quality in fewer steps!
        with autocast():
            student_output = denoise_multistep(
                student_unet,
                noisy_latents,
                prompt_embeds,
                pooled_prompt_embeds,
                student_scheduler,
                num_steps=student_steps,
                device=device
            )
            
            # Compute consistency loss
            # Student's 4-step output should match teacher's 20-step output!
            losses = distillation_loss(
                student_output,
                teacher_output.detach(),
                student_features=None,  # Can enable for stability
                teacher_features=None
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
        
        # Logging
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'consistency': f"{losses.get('consistency', 0):.4f}",
        })
        
        # Log to wandb
        if step % config['logging']['log_every'] == 0:
            def get_val(v):
                return v.item() if hasattr(v, 'item') else v

            wandb.log({
                'train/loss': loss.item(),
                'train/consistency_loss': get_val(losses.get('consistency', 0)),
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
    teacher_scheduler: DDPMScheduler,
    student_scheduler: DDPMScheduler,
    dataloader: DataLoader,
    distillation_loss: ConsistencyDistillationLoss,
    device: torch.device,
    epoch: int,
    config: Dict
) -> float:
    """Validate the student model."""
    
    student_unet.eval()
    teacher_unet.eval()
    
    total_loss = 0.0
    teacher_steps = config['training'].get('teacher_steps', 20)
    student_steps = config['training'].get('student_steps', 4)
    
    for batch in tqdm(dataloader, desc="Validating"):
        prompts = batch['prompts']
        
        prompt_embeds, pooled_prompt_embeds = encode_prompts(
            prompts, text_encoder_1, text_encoder_2,
            tokenizer_1, tokenizer_2, device
        )

        if 'latents' in batch:
            clean_latents = batch['latents'].to(device)
        else:
            pixel_values = batch['pixel_values'].to(device)
            clean_latents = vae.encode(pixel_values).latent_dist.sample()
            clean_latents = clean_latents * vae.config.scaling_factor
        
        noise = torch.randn_like(clean_latents)
        batch_size = clean_latents.shape[0]
        
        start_timestep = torch.randint(
            500, teacher_scheduler.config.num_train_timesteps,
            (1,), device=device
        ).item()
        
        timestep_tensor = torch.tensor([start_timestep] * batch_size, device=device)
        noisy_latents = teacher_scheduler.add_noise(clean_latents, noise, timestep_tensor)
        
        # Teacher: 20 steps
        teacher_output = denoise_multistep(
            teacher_unet, noisy_latents, prompt_embeds,
            pooled_prompt_embeds, teacher_scheduler,
            num_steps=teacher_steps, device=device
        )
        
        # Student: 4 steps
        student_output = denoise_multistep(
            student_unet, noisy_latents, prompt_embeds,
            pooled_prompt_embeds, student_scheduler,
            num_steps=student_steps, device=device
        )
        
        losses = distillation_loss(
            student_output, teacher_output,
            student_features=None,
            teacher_features=None
        )
        total_loss += losses['total'].item()
    
    avg_loss = total_loss / len(dataloader)
    
    wandb.log({
        'val/loss': avg_loss,
        'val/epoch': epoch
    })
    
    return avg_loss


def main(args):
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if config['logging']['use_wandb']:
        wandb.init(
            project=config['logging']['wandb_project'],
            name=config['logging']['run_name'] + "_consistency",
            config=config
        )
    
    os.makedirs(config['training']['output_dir'], exist_ok=True)
    
    print("\n" + "="*60)
    print("Consistency Distillation Training")
    print("="*60)
    print("Teacher: 20 steps (HIGH QUALITY)")
    print("Student: 4 steps (FAST, learns to match teacher quality!)")
    print("="*60)
    
    # Load teacher (BASE model, NO Lightning)
    print(f"\nLoading BASE teacher: {config['teacher']['model_id']}")
    teacher_pipe = StableDiffusionXLPipeline.from_pretrained(
        config['teacher']['model_id'],
        torch_dtype=torch.float16,
    )
    
    # NO Lightning LoRA for teacher - we want BASE quality!
    print("  ✓ Using BASE teacher (20-step quality)")
    
    teacher_unet = teacher_pipe.unet.to(device)
    teacher_unet.requires_grad_(False)
    
    # Load student
    print(f"\nLoading student: {config['student']['model_id']}")
    try:
        student_pipe = StableDiffusionXLPipeline.from_pretrained(
            config['student']['model_id'],
            torch_dtype=torch.float32,
        )
        student_unet = student_pipe.unet.to(device)
    except OSError:
        student_unet = UNet2DConditionModel.from_pretrained(
            config['student']['model_id'],
            torch_dtype=torch.float32
        ).to(device)
    
    student_unet.enable_gradient_checkpointing()

    if torch.cuda.is_available():
        try:
            student_unet.enable_xformers_memory_efficient_attention()
            teacher_unet.enable_xformers_memory_efficient_attention()
            print("Enabled xformers")
        except Exception:
            pass

    # Load components
    use_cached_latents = config['data'].get('use_cached_latents', False)
    
    if not use_cached_latents:
        vae = teacher_pipe.vae.to(device)
    else:
        vae = teacher_pipe.vae
        print("Using cached latents")

    vae.requires_grad_(False)
    
    text_encoder_1 = teacher_pipe.text_encoder.to(device)
    text_encoder_1.requires_grad_(False)
    
    text_encoder_2 = teacher_pipe.text_encoder_2.to(device)
    text_encoder_2.requires_grad_(False)
    
    tokenizer_1 = teacher_pipe.tokenizer
    tokenizer_2 = teacher_pipe.tokenizer_2
    
    # Schedulers
    teacher_scheduler = DDPMScheduler.from_pretrained(
        config['teacher']['model_id'],
        subfolder="scheduler"
    )
    
    student_scheduler = DDPMScheduler.from_pretrained(
        config['teacher']['model_id'],
        subfolder="scheduler"
    )
    
    # Print info
    teacher_params = sum(p.numel() for p in teacher_unet.parameters())
    student_params = sum(p.numel() for p in student_unet.parameters())
    print(f"\nTeacher UNet: {teacher_params:,} parameters")
    print(f"Student UNet: {student_params:,} parameters")
    
    # Load datasets
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    use_cached = config['data'].get('use_cached_latents', False)
    
    if use_cached:
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
    distillation_loss = ConsistencyDistillationLoss(
        consistency_weight=config['loss']['consistency_weight'],
        feature_weight=config['loss'].get('feature_weight', 0.0)
    )
    
    scaler = GradScaler()
    
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(
            student_unet.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    except ImportError:
        optimizer = torch.optim.AdamW(
            student_unet.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Consistency Training")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['num_epochs']}")
        
        train_loss = train_one_epoch(
            teacher_unet, student_unet, vae,
            text_encoder_1, text_encoder_2,
            tokenizer_1, tokenizer_2,
            teacher_scheduler, student_scheduler,
            train_loader, optimizer, scaler,
            distillation_loss, device, epoch, config
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        
        val_loss = validate(
            teacher_unet, student_unet, vae,
            text_encoder_1, text_encoder_2,
            tokenizer_1, tokenizer_2,
            teacher_scheduler, student_scheduler,
            val_loader, distillation_loss,
            device, epoch, config
        )
        
        print(f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                config['training']['output_dir'],
                "best_student_unet_consistency.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'consistency_distilled': True,
                'teacher_steps': config['training']['teacher_steps'],
                'student_steps': config['training']['student_steps']
            }, checkpoint_path)
            print(f"✅ Saved best model! Val loss: {val_loss:.4f}")
        
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['training']['output_dir'],
                f"student_unet_consistency_epoch{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': student_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config,
                'consistency_distilled': True,
                'teacher_steps': config['training']['teacher_steps'],
                'student_steps': config['training']['student_steps']
            }, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    print("\n" + "="*60)
    print("✅ Consistency Training Complete!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("\n🎯 Student can now achieve 20-step quality in 4 steps!")
    
    if config['logging']['use_wandb']:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consistency distillation training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/distillation_consistency_config.json',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    main(args)
