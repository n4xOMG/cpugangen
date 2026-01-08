#!/usr/bin/env python3
"""
Collect Training Data for Structured Initialization

Generates (prompt_embedding, final_latent) pairs by running the 4-step pipeline.
These pairs are used to train the StructuredInitializer to predict good initial latents.

Usage:
    # Using your own prompts file (one prompt per line):
    python scripts/collect_struct_init_data.py \\
        --prompts-file path/to/prompts.txt \\
        --output data/struct_init_training

    # Or generate random prompts:
    python scripts/collect_struct_init_data.py \\
        --num-samples 500 \\
        --output data/struct_init_training
"""

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path
from typing import List
import random

import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderTiny
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm

# Suppress harmless diffusers warning
warnings.filterwarnings("ignore", message=".*scale_model_input.*")

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def load_prompts_from_file(filepath: str, convert_spaces_to_commas: bool = False) -> List[str]:
    """
    Load prompts from a text file, one prompt per line.
    
    Args:
        filepath: Path to the prompts file
        convert_spaces_to_commas: If True, convert "1girl white hair" to "1girl, white, hair"
                                  Usually not needed - SDXL understands space-separated tags
    """
    prompts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                if convert_spaces_to_commas:
                    # Convert spaces to commas (for models that prefer comma separation)
                    line = ', '.join(line.split())
                prompts.append(line)
    return prompts


def load_optimized_pipeline(device: str = "auto"):
    """
    Load the optimized pipeline (SSD-1B + Lightning + TAESD).
    
    Args:
        device: "auto" (detect GPU), "cuda", or "cpu"
    """
    # Auto-detect device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use fp16 on GPU for speed, fp32 on CPU
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading optimized pipeline on {device.upper()} ({dtype})...")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "martineux/janku6",
        torch_dtype=dtype,
    )
    
    # Replace UNet with SSD-1B
    print("  Loading SSD-1B UNet...")
    try:
        ssd1b_unet = UNet2DConditionModel.from_pretrained(
            "segmind/SSD-1B",
            subfolder="unet",
            torch_dtype=dtype,
        )
    except Exception:
        ssd1b_pipe = StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B",
            torch_dtype=dtype,
        )
        ssd1b_unet = ssd1b_pipe.unet
        del ssd1b_pipe
        clear_memory()
    
    pipeline.unet = ssd1b_unet
    
    # Replace VAE with TAESD
    taesd = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl",
        torch_dtype=dtype,
    )
    pipeline.vae = taesd
    
    # Apply Lightning LoRA
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


# Fallback prompt templates for random generation
PROMPT_TEMPLATES = [
    "1girl, {hair}_hair, {eye}_eyes, {outfit}, {pose}, anime style",
    "1boy, {hair}_hair, {outfit}, {pose}, anime style",
    "anime landscape, {scene}, {time}, {weather}, detailed",
    "anime portrait, {hair}_hair, {expression}, beautiful lighting",
]

HAIR_COLORS = ["blue", "pink", "black", "blonde", "red", "purple", "silver", "green", "brown", "white"]
EYE_COLORS = ["blue", "red", "green", "purple", "gold", "brown"]
OUTFITS = ["school_uniform", "dress", "casual_clothes", "armor", "kimono", "hoodie"]
POSES = ["standing", "sitting", "walking", "running", "fighting"]
SCENES = ["city", "forest", "beach", "mountains", "cafe", "classroom"]
TIMES = ["sunset", "night", "morning", "afternoon"]
WEATHER = ["clear", "cloudy", "rainy", "snowy"]
EXPRESSIONS = ["smiling", "serious", "surprised", "happy", "neutral"]


def generate_random_prompt() -> str:
    """Generate a random prompt from templates."""
    template = random.choice(PROMPT_TEMPLATES)
    
    prompt = template.format(
        hair=random.choice(HAIR_COLORS),
        eye=random.choice(EYE_COLORS),
        outfit=random.choice(OUTFITS),
        pose=random.choice(POSES),
        scene=random.choice(SCENES),
        time=random.choice(TIMES),
        weather=random.choice(WEATHER),
        expression=random.choice(EXPRESSIONS),
    )
    
    return prompt


def collect_sample(
    pipeline,
    prompt: str,
    seed: int,
    num_steps: int = 4,
) -> dict:
    """
    Run pipeline and collect (embedding, final_latent) pair.
    """
    device = pipeline.device if hasattr(pipeline, 'device') else torch.device("cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Encode prompt to get embeddings
    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    
    # Prepare timesteps
    pipeline.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipeline.scheduler.timesteps
    
    # Prepare initial latents (random)
    latents = torch.randn(
        (1, 4, 64, 64),
        generator=generator,
        dtype=prompt_embeds.dtype,  # Match pipeline dtype
    ).to(device)
    latents = latents * pipeline.scheduler.init_noise_sigma
    
    # Prepare added conditioning
    add_time_ids = pipeline._get_add_time_ids(
        (512, 512), (0, 0), (512, 512),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=1280,
    ).to(device)
    
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids,
    }
    
    # Run denoising loop
    with torch.no_grad():
        for t in timesteps:
            timestep = torch.tensor([t], device=device, dtype=latents.dtype)
            noise_pred = pipeline.unet(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    # Return the pair
    return {
        "prompt": prompt,
        "seed": seed,
        "pooled_embed": pooled_prompt_embeds.cpu().squeeze(0),  # (1280,)
        "target_latent": latents.cpu().squeeze(0),  # (4, 64, 64)
    }


def main():
    parser = argparse.ArgumentParser(description="Collect training data for structured initialization")
    
    parser.add_argument("--prompts-file", type=str, default=None, 
                        help="Path to text file with prompts (one per line)")
    parser.add_argument("--num-samples", type=int, default=500, 
                        help="Number of samples (only used if no prompts file)")
    parser.add_argument("--output", type=str, default="data/struct_init_training", 
                        help="Output directory")
    parser.add_argument("--seed-start", type=int, default=0, 
                        help="Starting seed for random variation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of prompts to process from file")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get prompts
    if args.prompts_file:
        print(f"Loading prompts from: {args.prompts_file}")
        prompts = load_prompts_from_file(args.prompts_file)
        if args.max_samples:
            prompts = prompts[:args.max_samples]
        print(f"  Loaded {len(prompts)} prompts")
    else:
        print(f"Generating {args.num_samples} random prompts...")
        prompts = [generate_random_prompt() for _ in range(args.num_samples)]
    
    print("="*60)
    print("COLLECTING STRUCTURED INITIALIZATION TRAINING DATA")
    print("="*60)
    print(f"Samples: {len(prompts)}")
    print(f"Output: {output_dir}")
    print()
    
    # Load pipeline (auto-detects GPU)
    pipeline, device = load_optimized_pipeline()
    print(f"Device: {device.upper()}\n")
    
    # Collect samples
    print(f"\nCollecting {len(prompts)} samples...")
    
    for i, prompt in enumerate(tqdm(prompts)):
        seed = args.seed_start + i
        
        try:
            sample = collect_sample(pipeline, prompt, seed, num_steps=4)
            
            # Save sample
            sample_path = output_dir / f"sample_{i:05d}.pt"
            torch.save(sample, sample_path)
            
        except Exception as e:
            print(f"\n  Error on sample {i}: {e}")
            continue
    
    print(f"\n✓ Collected {len(prompts)} samples to {output_dir}")
    
    # Cleanup
    del pipeline
    clear_memory()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

