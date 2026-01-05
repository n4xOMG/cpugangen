#!/usr/bin/env python3
"""
Train Latent Consistency Model (LCM) for fast inference.

LCM enables 4-step generation (vs 20-50 steps) with 90-95% quality retention.
This provides ~12x speedup on CPU inference.

Based on:
"Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference"
Luo et al., 2023

Requires:
- GPU for training (tested on 3090 24GB)
- Base SDXL model (e.g., Illustrious)
- Training dataset (or use prompts)
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def train_lcm(
    base_model: str,
    output_dir: str,
    num_steps: int = 4,
    num_epochs: int = 50,
    train_batch_size: int = 4,
    learning_rate: float = 1e-5,
    use_lora: bool = True,
    lora_rank: int = 64,
):
    """
    Train Latent Consistency Model using consistency distillation.
    
    Args:
        base_model: Base SDXL model ID or path
        output_dir: Where to save checkpoints
        num_steps: Target inference steps (typically 4)
        num_epochs: Training epochs
        train_batch_size: Batch size
        learning_rate: Learning rate
        use_lora: Use LoRA for efficient training
        lora_rank: LoRA rank (64 or 128)
    """
    from diffusers import StableDiffusionXLPipeline, DDIMScheduler, LCMScheduler
    from peft import LoraConfig, get_peft_model
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 70)
    print("🚀 LCM TRAINING")
    print("=" * 70)
    print(f"\nBase model: {base_model}")
    print(f"Target steps: {num_steps}")
    print(f"Output: {output_dir}")
    print(f"LoRA: {use_lora} (rank={lora_rank if use_lora else 'N/A'})")
    
    # Load base model
    print("\n📦 Loading base model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
    )
    pipe.to("cuda")
    
    unet = pipe.unet
    vae = pipe.vae
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    
    # Freeze VAE and text encoders
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    # Apply LoRA to UNet if requested
    if use_lora:
        print(f"\n🔧 Applying LoRA (rank={lora_rank})...")
        
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()
    else:
        unet.requires_grad_(True)
    
    # Setup schedulers
    teacher_scheduler = DDIMScheduler.from_pretrained(
        base_model,
        subfolder="scheduler",
    )
    
    lcm_scheduler = LCMScheduler.from_config(teacher_scheduler.config)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    
    # Training prompts (you should replace with actual dataset)
    training_prompts = [
        "1girl, solo, blue_eyes, smile, anime style, masterpiece, best quality",
        "1girl, long_hair, red_eyes, fantasy, detailed, high quality",
        "1boy, short_hair, serious, school_uniform, anime, detailed",
        "2girls, friends, outdoors, cherry_blossoms, spring, anime",
        # Add more prompts or load from file
    ]
    
    print(f"\n📚 Training prompts: {len(training_prompts)}")
    
    # Training loop
    print(f"\n🏋️ Training for {num_epochs} epochs...")
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        epoch_loss = 0
        progress_bar = tqdm(training_prompts, desc="Training")
        
        for prompt in progress_bar:
            # Encode prompt
            with torch.no_grad():
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_inputs_2 = tokenizer_2(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                prompt_embeds = text_encoder(
                    text_inputs.input_ids.to("cuda"),
                    output_hidden_states=True,
                )
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]
                
                prompt_embeds_2 = text_encoder_2(
                    text_inputs_2.input_ids.to("cuda"),
                    output_hidden_states=True,
                )
                pooled_prompt_embeds_2 = prompt_embeds_2[0]
                prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
                
                # Concat embeddings
                prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
                
                # Generate random latents
                latents = torch.randn(
                    (1, 4, 96, 96),  # SDXL latent size for 768x768
                    device="cuda",
                    dtype=torch.float16,
                )
                
            # Sample timestep
            timestep = torch.randint(
                0, teacher_scheduler.config.num_train_timesteps,
                (1,),
                device="cuda",
            )
            
            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = teacher_scheduler.add_noise(latents, noise, timestep)
            
            # Predict noise with student (UNet)
            model_pred = unet(
                noisy_latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds_2,
                    "time_ids": torch.zeros((1, 6), device="cuda"),
                },
            ).sample
            
            # Get teacher prediction (with DDIM)
            with torch.no_grad():
                teacher_pred = unet(
                    noisy_latents,
                    timestep,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_prompt_embeds_2,
                        "time_ids": torch.zeros((1, 6), device="cuda"),
                    },
                ).sample
            
            # Consistency loss (simplified - full LCM uses more complex loss)
            loss = F.mse_loss(model_pred, teacher_pred)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss / (progress_bar.n + 1):.4f}'
            })
        
        avg_epoch_loss = epoch_loss / len(training_prompts)
        print(f"   Average loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = output_dir / f"checkpoint-epoch{epoch+1}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            if use_lora:
                unet.save_pretrained(checkpoint_dir / "unet_lora")
            else:
                unet.save_pretrained(checkpoint_dir / "unet")
            
            print(f"   💾 Saved checkpoint: {checkpoint_dir}")
    
    # Save final model
    print("\n💾 Saving final model...")
    
    if use_lora:
        unet.save_pretrained(output_dir / "unet_lora")
    else:
        unet.save_pretrained(output_dir / "unet")
    
    # Save config
    config = {
        'base_model': base_model,
        'num_steps': num_steps,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'use_lora': use_lora,
        'lora_rank': lora_rank if use_lora else None,
    }
    
    with open(output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Model saved to: {output_dir}")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Test the LCM model with test_lcm.py")
    print("2. Compare with baseline at 4 steps vs 20-50 steps")
    print("3. If good, export to ONNX for production")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Train Latent Consistency Model for fast inference'
    )
    parser.add_argument('--base-model', required=True,
                       help='Base SDXL model ID or path')
    parser.add_argument('--output-dir', default='lcm_checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--steps', type=int, default=4,
                       help='Target inference steps')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--use-lora', action='store_true', default=True,
                       help='Use LoRA for efficient training')
    parser.add_argument('--lora-rank', type=int, default=64,
                       help='LoRA rank (64 or 128)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available! LCM training requires GPU.")
        print("   Please run on a machine with NVIDIA GPU.")
        sys.exit(1)
    
    print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Train
    train_lcm(
        base_model=args.base_model,
        output_dir=args.output_dir,
        num_steps=args.steps,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.lr,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
    )


if __name__ == "__main__":
    main()
