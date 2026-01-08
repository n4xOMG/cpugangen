#!/usr/bin/env python3
"""
Benchmark Trajectory Prior

Compares:
- Baseline: Full 4-step UNet inference
- Predictor-assisted: Some steps use predictor, others use UNet

Usage:
    python scripts/benchmark_trajectory_prior.py \\
        --checkpoint checkpoints/trajectory_prior.pt \\
        --num-prompts 5 \\
        --output outputs/trajectory_benchmark
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.optimization.trajectory_prior import TrajectoryPredictor, predictor_assisted_denoise


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def load_pipeline():
    """Load optimized pipeline on CPU (target platform)."""
    print("Loading pipeline on CPU...")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "martineux/janku6",
        torch_dtype=torch.float32,
    )
    
    try:
        ssd1b_unet = UNet2DConditionModel.from_pretrained(
            "segmind/SSD-1B", subfolder="unet", torch_dtype=torch.float32,
        )
    except Exception:
        ssd1b_pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float32)
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


def run_baseline(pipeline, prompt: str, seed: int) -> tuple:
    """Run baseline 4-step inference."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    start = time.time()
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=generator,
        )
    elapsed = time.time() - start
    
    return result.images[0], elapsed


def run_predictor_assisted(
    pipeline,
    predictor: TrajectoryPredictor,
    prompt: str,
    seed: int,
    predictor_steps: List[int],
) -> tuple:
    """Run predictor-assisted inference."""
    device = torch.device("cpu")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # Encode prompt
    prompt_embeds, _, pooled_prompt_embeds, _ = pipeline.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    
    # Prepare timesteps
    pipeline.scheduler.set_timesteps(4, device=device)
    timesteps = pipeline.scheduler.timesteps
    
    # Initialize latent
    latents = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32)
    latents = latents * pipeline.scheduler.init_noise_sigma
    
    # Prepare conditioning
    add_time_ids = pipeline._get_add_time_ids(
        (512, 512), (0, 0), (512, 512),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=1280,
    )
    added_cond_kwargs = {
        "text_embeds": pooled_prompt_embeds,
        "time_ids": add_time_ids,
    }
    
    # Run hybrid inference
    start = time.time()
    latents = predictor_assisted_denoise(
        predictor=predictor,
        unet=pipeline.unet,
        scheduler=pipeline.scheduler,
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        added_cond_kwargs=added_cond_kwargs,
        predictor_steps=predictor_steps,
        blend_weight=0.0,
    )
    elapsed = time.time() - start
    
    # Decode
    with torch.no_grad():
        latents = latents / pipeline.vae.config.scaling_factor
        image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = pipeline.image_processor.postprocess(image, output_type="pil")[0]
    
    return image, elapsed


def benchmark_config(name: str, images: List, times: List, output_dir: Path):
    """Save results for a configuration."""
    images_dir = output_dir / "images" / name.replace(" ", "_")
    images_dir.mkdir(exist_ok=True, parents=True)
    
    for i, img in enumerate(images):
        img.save(images_dir / f"{i:02d}.png")
    
    return {
        "name": name,
        "times": times,
        "mean_time": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark trajectory prior")
    parser.add_argument("--checkpoint", type=str, required=True, help="Predictor checkpoint")
    parser.add_argument("--num-prompts", type=int, default=3, help="Number of test prompts")
    parser.add_argument("--output", type=str, default="outputs/trajectory_benchmark")
    
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
    
    seeds = [42, 123, 456, 789, 1024][:args.num_prompts]
    
    print("="*60)
    print("TRAJECTORY PRIOR BENCHMARK")
    print("="*60)
    print(f"Prompts: {len(prompts)}")
    print(f"Checkpoint: {args.checkpoint}")
    print()
    
    # Load pipeline
    pipeline = load_pipeline()
    
    # Load predictor
    print(f"\nLoading predictor from {args.checkpoint}...")
    predictor = TrajectoryPredictor.load(args.checkpoint, device="cpu")
    print("  ✓ Predictor loaded")
    
    results = []
    
    # Baseline: Full 4-step UNet
    print("\n" + "="*60)
    print("BASELINE: 4-step UNet")
    print("="*60)
    
    baseline_images = []
    baseline_times = []
    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        img, t = run_baseline(pipeline, prompt, seed)
        baseline_images.append(img)
        baseline_times.append(t)
        print(f"  [{i+1}/{len(prompts)}] {t:.2f}s")
    
    results.append(benchmark_config("Baseline 4-step", baseline_images, baseline_times, output_dir))
    
    # Test different predictor strategies
    strategies = [
        ("Pred steps [0]", [0]),           # Predictor on step 0 only
        ("Pred steps [0,2]", [0, 2]),      # Predictor on steps 0 and 2
        ("Pred steps [0,1,2]", [0, 1, 2]), # Predictor on first 3 steps
    ]
    
    for strat_name, pred_steps in strategies:
        print("\n" + "="*60)
        print(f"STRATEGY: {strat_name}")
        print("="*60)
        
        images = []
        times = []
        for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
            img, t = run_predictor_assisted(pipeline, predictor, prompt, seed, pred_steps)
            images.append(img)
            times.append(t)
            print(f"  [{i+1}/{len(prompts)}] {t:.2f}s")
        
        results.append(benchmark_config(strat_name, images, times, output_dir))
    
    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    baseline_time = results[0]["mean_time"]
    for r in results:
        speedup = baseline_time / r["mean_time"]
        print(f"  {r['name']}: {r['mean_time']:.1f}s ({speedup:.2f}x)")
    
    print(f"\n✓ Results saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
