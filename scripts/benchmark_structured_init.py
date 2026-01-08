#!/usr/bin/env python3
"""
Benchmark Structured Initialization

Compares:
- 4-step with random initialization (baseline)
- 3-step with structured initialization (target)

Usage:
    python scripts/benchmark_structured_init.py \\
        --checkpoint checkpoints/struct_init.pt \\
        --num-prompts 5 \\
        --output outputs/struct_init_benchmark
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import statistics

import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderTiny
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.optimization.structured_init import StructuredInitializer, smart_latent_init


BENCHMARK_SEEDS = [42, 123, 456, 789, 1024]


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def load_optimized_pipeline():
    """Load the optimized pipeline."""
    print("Loading optimized pipeline...")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "martineux/janku6",
        torch_dtype=torch.float32,
    )
    
    try:
        ssd1b_unet = UNet2DConditionModel.from_pretrained(
            "segmind/SSD-1B",
            subfolder="unet",
            torch_dtype=torch.float32,
        )
    except Exception:
        ssd1b_pipe = StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B",
            torch_dtype=torch.float32,
        )
        ssd1b_unet = ssd1b_pipe.unet
        del ssd1b_pipe
        clear_memory()
    
    pipeline.unet = ssd1b_unet
    
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float32)
    pipeline.vae = taesd
    
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
    pipeline.load_lora_weights(load_file(ckpt))
    
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()
    pipeline = pipeline.to("cpu")
    
    print("  ✓ Pipeline ready")
    return pipeline


def run_inference(
    pipeline,
    prompt: str,
    num_steps: int,
    seed: int,
    initializer: StructuredInitializer = None,
) -> tuple:
    """Run inference with optional structured initialization."""
    device = torch.device("cpu")
    
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
    
    # Initialize latents
    if initializer is not None:
        # Smart initialization
        latents = smart_latent_init(
            initializer=initializer,
            pooled_embed=pooled_prompt_embeds,
            seed=seed,
            init_noise_sigma=pipeline.scheduler.init_noise_sigma,
            variation_scale=0.7,
            structure_scale=0.2,
            device="cpu",
        )
    else:
        # Random initialization
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32)
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
    
    # Denoising loop
    start_time = time.time()
    
    with torch.no_grad():
        for t in tqdm(timesteps, desc=f"{num_steps}-step", leave=False):
            timestep = torch.tensor([t], device=device, dtype=latents.dtype)
            noise_pred = pipeline.unet(
                latents,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    elapsed = time.time() - start_time
    
    # Decode
    with torch.no_grad():
        latents_scaled = latents / pipeline.vae.config.scaling_factor
        image = pipeline.vae.decode(latents_scaled, return_dict=False)[0]
    
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
    
    return image, elapsed


def save_images(images: List[Image.Image], output_dir: Path, config_name: str, prompts: List[str]):
    """Save generated images."""
    images_dir = output_dir / "images" / config_name.replace(" ", "_").replace(":", "")
    images_dir.mkdir(exist_ok=True, parents=True)
    
    for i, (img, prompt) in enumerate(zip(images, prompts)):
        prompt_short = prompt[:50].replace(",", "").replace(" ", "_")
        img.save(images_dir / f"{i:02d}_{prompt_short}.png")
    
    return images_dir


def benchmark_config(
    config_name: str,
    pipeline,
    prompts: List[str],
    num_steps: int,
    initializer: StructuredInitializer = None,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """Benchmark a configuration."""
    print(f"\n{'='*60}")
    print(f"CONFIGURATION: {config_name}")
    print(f"{'='*60}")
    print(f"Steps: {num_steps}")
    print(f"Smart init: {'Yes' if initializer else 'No'}")
    
    times = []
    images = []
    
    print(f"\n🏃 Running {len(prompts)} iterations...")
    
    for i, prompt in enumerate(prompts):
        seed = BENCHMARK_SEEDS[i % len(BENCHMARK_SEEDS)]
        image, elapsed = run_inference(pipeline, prompt, num_steps, seed, initializer)
        
        times.append(elapsed)
        images.append(image)
        print(f"  [{i+1}/{len(prompts)}] {elapsed:.2f}s | Seed: {seed}")
    
    # Save images
    if output_dir:
        images_dir = save_images(images, output_dir, config_name, prompts)
        print(f"\n📸 Images saved to: {images_dir}")
    
    results = {
        "config_name": config_name,
        "num_steps": num_steps,
        "smart_init": initializer is not None,
        "times": times,
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }
    
    print(f"\n📊 Results:")
    print(f"  Mean:   {results['mean_time']:.2f}s ± {results['stdev']:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark structured initialization")
    
    parser.add_argument("--checkpoint", type=str, default=None, help="StructuredInitializer checkpoint")
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of test prompts")
    parser.add_argument("--output", type=str, default="outputs/struct_init_benchmark", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prompts = [
        "1girl, blue_hair, anime_style, detailed, school_uniform",
        "anime landscape, mountains, sunset, fantasy, castle",
        "1boy, fantasy_outfit, action_pose, sword, magic",
        "anime_portrait, detailed_face, beautiful_lighting, flowers",
        "2girls, cafe, sitting, talking, warm_atmosphere",
    ][:args.num_prompts]
    
    print("="*60)
    print("STRUCTURED INITIALIZATION BENCHMARK")
    print("="*60)
    print(f"Prompts: {len(prompts)}")
    print(f"Output: {output_dir}")
    
    # Load pipeline
    pipeline = load_optimized_pipeline()
    
    # Load initializer if provided
    initializer = None
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\nLoading structured initializer from {args.checkpoint}...")
        initializer = StructuredInitializer.load(args.checkpoint)
        print("  ✓ Initializer loaded")
    
    results = []
    
    # Baseline: 4-step random init
    result_baseline = benchmark_config(
        "Baseline: 4-step random init",
        pipeline, prompts, num_steps=4,
        initializer=None, output_dir=output_dir,
    )
    results.append(result_baseline)
    
    # Comparison: 3-step random init (to see quality loss)
    result_3step_random = benchmark_config(
        "3-step random init",
        pipeline, prompts, num_steps=3,
        initializer=None, output_dir=output_dir,
    )
    results.append(result_3step_random)
    
    # Target: 3-step smart init (if trained)
    if initializer:
        result_3step_smart = benchmark_config(
            "3-step SMART init",
            pipeline, prompts, num_steps=3,
            initializer=initializer, output_dir=output_dir,
        )
        results.append(result_3step_smart)
    
    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create report
    report_path = output_dir / "comparison_report.md"
    with open(report_path, "w") as f:
        f.write("# Structured Initialization Benchmark\n\n")
        f.write("## Performance Comparison\n\n")
        f.write("| Configuration | Steps | Mean Time (s) | Speedup |\n")
        f.write("|--------------|-------|---------------|----------|\n")
        
        baseline_time = results[0]["mean_time"]
        for r in results:
            speedup = baseline_time / r["mean_time"]
            f.write(f"| {r['config_name']} | {r['num_steps']} | {r['mean_time']:.2f} | {speedup:.2f}x |\n")
        
        f.write("\n## Quality Comparison\n\n")
        f.write("Check the images directory to compare quality visually.\n")
        f.write("Target: 3-step smart init should match 4-step random init quality.\n")
    
    print(f"\n✓ Results saved to: {results_path}")
    print(f"✓ Report saved to: {report_path}")
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    
    baseline_time = results[0]["mean_time"]
    for r in results:
        speedup = baseline_time / r["mean_time"]
        print(f"  {r['config_name']}: {speedup:.2f}x ({r['mean_time']:.1f}s)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
