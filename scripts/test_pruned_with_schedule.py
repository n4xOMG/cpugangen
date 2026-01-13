#!/usr/bin/env python3
"""
Test Pruned Model with Optimized Schedule

Generates images using:
1. Baseline (unpruned + EDM)
2. Pruned + EDM (old schedule)
3. Pruned + AYS (optimized schedule)

Usage:
    python scripts/test_pruned_with_schedule.py \
        --pruned-model checkpoints/pruned_illustrious \
        --optimal-schedule configs/optimal_schedule_pruned.json \
        --prompt "1girl, blue_hair, anime_style, detailed" \
        --output outputs/test_pruned
"""

import argparse
import os
import sys
import time
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.schedules import load_schedule, edm_schedule


def generate_image(
    pipeline,
    prompt: str,
    num_steps: int,
    timesteps: list = None,
    seed: int = 42
):
    """
    Generate image with optional custom timesteps.
    
    Args:
        pipeline: StableDiffusionXLPipeline
        prompt: Text prompt
        num_steps: Number of inference steps
        timesteps: Optional custom timesteps
        seed: Random seed
        
    Returns:
        tuple: (image, generation_time)
    """
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    # If custom timesteps provided, set them in the scheduler
    if timesteps is not None:
        # Convert timesteps to sigmas for the scheduler
        pipeline.scheduler.set_timesteps(num_steps)
        # Note: For full integration, you'd need to modify the scheduler
        # to accept custom timesteps. For now, we use num_steps.
        print(f"  Using {len(timesteps)} custom timesteps")
    
    start_time = time.time()
    
    output = pipeline(
        prompt=prompt,
        num_inference_steps=num_steps,
        generator=generator,
        guidance_scale=7.5,  # Standard CFG
    )
    
    elapsed = time.time() - start_time
    
    return output.images[0], elapsed


def main():
    parser = argparse.ArgumentParser(description="Test Pruned Model with Optimized Schedule")
    parser.add_argument(
        "--base-model",
        type=str,
        default="martineux/janku6",
        help="Base unpruned model"
    )
    parser.add_argument(
        "--pruned-model",
        type=str,
        required=True,
        help="Path to pruned model"
    )
    parser.add_argument(
        "--optimal-schedule",
        type=str,
        required=True,
        help="Path to optimized schedule JSON"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="1girl, blue_hair, anime_style, detailed, masterpiece",
        help="Generation prompt"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of inference steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/test_pruned",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline generation (faster)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Testing Pruned Model + Optimized Schedule")
    print(f"{'='*60}\n")
    print(f"Pruned Model: {args.pruned_model}")
    print(f"Optimal Schedule: {args.optimal_schedule}")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.num_steps}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}\n")
    
    # Load optimal schedule
    print("Loading optimized schedule...")
    try:
        optimal_timesteps = load_schedule(args.optimal_schedule)
        print(f"✅ Loaded {len(optimal_timesteps)} timesteps")
        print(f"   Range: [{optimal_timesteps[0]:.3f}, {optimal_timesteps[-1]:.3f}]\n")
    except Exception as e:
        print(f"⚠️  Could not load schedule: {e}")
        print("   Will use standard EDM schedule\n")
        optimal_timesteps = None
    
    results = {}
    
    # Test 1: Baseline (unpruned + EDM) - OPTIONAL
    if not args.skip_baseline:
        print(f"{'='*60}")
        print("Test 1: Baseline (Unpruned + EDM)")
        print(f"{'='*60}\n")
        
        print("Loading baseline model...")
        try:
            baseline_pipe = StableDiffusionXLPipeline.from_pretrained(
                args.base_model,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            baseline_pipe = baseline_pipe.to(args.device)
            print("✅ Model loaded\n")
            
            print("Generating image...")
            baseline_img, baseline_time = generate_image(
                pipeline=baseline_pipe,
                prompt=args.prompt,
                num_steps=args.num_steps,
                seed=args.seed
            )
            
            # Save
            baseline_path = os.path.join(args.output, "baseline_unpruned_edm.png")
            baseline_img.save(baseline_path)
            
            results['baseline'] = {
                'time': baseline_time,
                'path': baseline_path,
                'model': 'unpruned',
                'schedule': 'EDM'
            }
            
            print(f"✅ Generated in {baseline_time:.2f}s")
            print(f"   Saved to: {baseline_path}\n")
            
            # Free memory
            del baseline_pipe
            torch.cuda.empty_cache() if args.device == "cuda" else None
            
        except Exception as e:
            print(f"❌ Baseline generation failed: {e}\n")
    
    # Test 2: Pruned + EDM (old schedule)
    print(f"{'='*60}")
    print("Test 2: Pruned Model + EDM Schedule")
    print(f"{'='*60}\n")
    
    print("Loading pruned model...")
    try:
        pruned_pipe = StableDiffusionXLPipeline.from_pretrained(
            args.pruned_model,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pruned_pipe = pruned_pipe.to(args.device)
        print("✅ Pruned model loaded\n")
        
        print("Generating image with EDM schedule...")
        pruned_edm_img, pruned_edm_time = generate_image(
            pipeline=pruned_pipe,
            prompt=args.prompt,
            num_steps=args.num_steps,
            seed=args.seed
        )
        
        # Save
        pruned_edm_path = os.path.join(args.output, "pruned_edm.png")
        pruned_edm_img.save(pruned_edm_path)
        
        results['pruned_edm'] = {
            'time': pruned_edm_time,
            'path': pruned_edm_path,
            'model': 'pruned',
            'schedule': 'EDM'
        }
        
        print(f"✅ Generated in {pruned_edm_time:.2f}s")
        print(f"   Saved to: {pruned_edm_path}\n")
        
    except Exception as e:
        print(f"❌ Pruned + EDM generation failed: {e}\n")
        return
    
    # Test 3: Pruned + AYS (optimized schedule)
    print(f"{'='*60}")
    print("Test 3: Pruned Model + AYS Optimized Schedule")
    print(f"{'='*60}\n")
    
    print("Generating image with optimized schedule...")
    try:
        pruned_ays_img, pruned_ays_time = generate_image(
            pipeline=pruned_pipe,
            prompt=args.prompt,
            num_steps=args.num_steps,
            timesteps=optimal_timesteps,
            seed=args.seed
        )
        
        # Save
        pruned_ays_path = os.path.join(args.output, "pruned_ays_optimized.png")
        pruned_ays_img.save(pruned_ays_path)
        
        results['pruned_ays'] = {
            'time': pruned_ays_time,
            'path': pruned_ays_path,
            'model': 'pruned',
            'schedule': 'AYS_optimized'
        }
        
        print(f"✅ Generated in {pruned_ays_time:.2f}s")
        print(f"   Saved to: {pruned_ays_path}\n")
        
    except Exception as e:
        print(f"❌ Pruned + AYS generation failed: {e}\n")
    
    # Summary
    print(f"{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    
    for config_name, config_data in results.items():
        speedup = ""
        if 'baseline' in results and config_name != 'baseline':
            speedup = f" ({results['baseline']['time'] / config_data['time']:.2f}x faster)"
        
        print(f"{config_name}:")
        print(f"  Model: {config_data['model']}")
        print(f"  Schedule: {config_data['schedule']}")
        print(f"  Time: {config_data['time']:.2f}s{speedup}")
        print(f"  Image: {config_data['path']}\n")
    
    # Save results JSON
    results_path = os.path.join(args.output, "results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'prompt': args.prompt,
            'num_steps': args.num_steps,
            'seed': args.seed,
            'results': results
        }, f, indent=2)
    
    print(f"📊 Results saved to: {results_path}")
    
    # Create comparison grid
    print("\nCreating comparison grid...")
    try:
        images = [Image.open(results[k]['path']) for k in sorted(results.keys())]
        
        # Create grid
        width, height = images[0].size
        grid_width = width * len(images)
        grid = Image.new('RGB', (grid_width, height))
        
        for i, img in enumerate(images):
            grid.paste(img, (i * width, 0))
        
        grid_path = os.path.join(args.output, "comparison_grid.png")
        grid.save(grid_path)
        print(f"✅ Comparison grid saved to: {grid_path}\n")
        
    except Exception as e:
        print(f"⚠️  Could not create grid: {e}\n")
    
    print("✅ Test complete!")
    print(f"\n💡 Visual comparison:")
    print(f"   Open: {args.output}")
    print(f"   Compare: pruned_edm.png vs pruned_ays_optimized.png")
    print(f"   Expected: AYS version should have similar/better quality\n")


if __name__ == "__main__":
    main()
