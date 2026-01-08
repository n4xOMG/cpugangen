#!/usr/bin/env python3
"""
Collect Trajectory Data for Universal Prior Training

Runs the pipeline and saves (z_t, timestep, noise_pred) at each step.
This data trains the TrajectoryPredictor to learn denoising patterns.

Usage:
    python scripts/collect_trajectory_data.py \\
        --prompts-file hypothesis/general_output.txt \\
        --max-samples 1000 \\
        --output data/trajectory_training
"""

import argparse
import gc
import sys
import warnings
from pathlib import Path
from typing import List
import random

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderTiny
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm

# Suppress harmless warnings
warnings.filterwarnings("ignore", message=".*scale_model_input.*")

sys.path.insert(0, str(Path(__file__).parent.parent))


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def load_prompts_from_file(filepath: str, max_samples: int = None) -> List[str]:
    """Load prompts from text file, one per line."""
    prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    if max_samples:
        prompts = prompts[:max_samples]
    return prompts


def load_pipeline(device: str = "auto"):
    """Load optimized pipeline."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading pipeline on {device.upper()} ({dtype})...")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "martineux/janku6",
        torch_dtype=dtype,
    )
    
    # SSD-1B UNet
    print("  Loading SSD-1B UNet...")
    try:
        ssd1b_unet = UNet2DConditionModel.from_pretrained(
            "segmind/SSD-1B", subfolder="unet", torch_dtype=dtype,
        )
    except Exception:
        ssd1b_pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=dtype)
        ssd1b_unet = ssd1b_pipe.unet
        del ssd1b_pipe
        clear_memory()
    pipeline.unet = ssd1b_unet
    
    # TAESD VAE
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=dtype)
    pipeline.vae = taesd
    
    # Lightning LoRA
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
    pipeline.load_lora_weights(load_file(ckpt))
    
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()
    pipeline = pipeline.to(device)
    
    print(f"  ✓ Pipeline ready on {device.upper()}")
    return pipeline, device


def collect_trajectory(
    pipeline,
    prompt: str,
    seed: int,
    num_steps: int = 4,
) -> List[dict]:
    """
    Run pipeline and collect (z_t, t, noise_pred) at each step.
    
    Returns list of dicts, one per step.
    """
    device = next(pipeline.unet.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Encode prompt
    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    
    # Prepare timesteps
    pipeline.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    
    # Initialize latent
    latents = torch.randn(
        (1, 4, 64, 64),
        generator=generator,
        dtype=prompt_embeds.dtype,
    ).to(device)
    latents = latents * pipeline.scheduler.init_noise_sigma
    
    # Prepare conditioning
    add_time_ids = pipeline._get_add_time_ids(
        (512, 512), (0, 0), (512, 512),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=1280,
    ).to(device)
    
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids,
    }
    
    # Run denoising and collect data at each step
    samples = []
    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            # Save current state BEFORE denoising
            z_t = latents.clone()
            
            # Run UNet
            latent_model_input = pipeline.scheduler.scale_model_input(latents, t)
            noise_pred = pipeline.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # Save sample
            samples.append({
                "z_t": z_t.cpu().squeeze(0),              # (4, 64, 64)
                "timestep": torch.tensor(t.item()),       # scalar
                "noise_pred": noise_pred.cpu().squeeze(0), # (4, 64, 64)
                "step_idx": step_idx,
            })
            
            # Update latent
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Collect trajectory training data")
    parser.add_argument("--prompts-file", type=str, default=None, help="Prompts file (one per line)")
    parser.add_argument("--num-random", type=int, default=500, help="Number of random prompts if no file")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit prompts from file")
    parser.add_argument("--output", type=str, default="data/trajectory_training", help="Output directory")
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    if args.prompts_file:
        prompts = load_prompts_from_file(args.prompts_file, args.max_samples)
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        # Generate simple random prompts
        tags = ["1girl", "1boy", "landscape", "anime", "blue_hair", "red_eyes", "school_uniform"]
        prompts = [", ".join(random.sample(tags, 4)) for _ in range(args.num_random)]
        print(f"Generated {len(prompts)} random prompts")
    
    print("="*60)
    print("COLLECTING TRAJECTORY DATA")
    print("="*60)
    print(f"Prompts: {len(prompts)}")
    print(f"Steps per prompt: 4")
    print(f"Total samples: {len(prompts) * 4}")
    print(f"Output: {output_dir}")
    print()
    
    # Load pipeline
    pipeline, device = load_pipeline()
    
    # Collect
    sample_idx = 0
    for i, prompt in enumerate(tqdm(prompts, desc="Collecting")):
        seed = args.seed_start + i
        
        try:
            trajectory = collect_trajectory(pipeline, prompt, seed, num_steps=4)
            
            # Save each step as separate sample
            for step_data in trajectory:
                sample_path = output_dir / f"sample_{sample_idx:06d}.pt"
                torch.save(step_data, sample_path)
                sample_idx += 1
                
        except Exception as e:
            print(f"\n  Error on prompt {i}: {e}")
            continue
    
    print(f"\n✓ Collected {sample_idx} samples to {output_dir}")
    
    del pipeline
    clear_memory()
    return 0


if __name__ == "__main__":
    sys.exit(main())
