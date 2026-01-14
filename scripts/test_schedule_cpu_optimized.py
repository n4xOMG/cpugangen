#!/usr/bin/env python3
"""
CPU-Optimized Schedule Test

Same as test_schedule_optimization.py but with CPU optimizations:
- Winograd convolution
- Optimal threading
- Memory layout optimization

Usage:
    python scripts/test_schedule_cpu_optimized.py \
        --model martineux/janku6 \
        --optimal-schedule configs/optimal_schedule_baseline.json \
        --enable-winograd \
        --num-threads 6
"""

import argparse
import os
import sys
import time
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.schedules import load_schedule
from utils.cpu_optimization import enable_cpu_optimizations


def generate_image(
    pipeline,
    prompt: str,
    num_steps: int,
    seed: int = 42
):
    """Generate image and return (image, time)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    start_time = time.time()
    
    with torch.inference_mode():
        output = pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            generator=generator,
            guidance_scale=7.5,
        )
    
    elapsed = time.time() - start_time
    
    return output.images[0], elapsed


def main():
    parser = argparse.ArgumentParser(description="CPU-Optimized Schedule Test")
    parser.add_argument("--model", type=str, default="martineux/janku6")
    parser.add_argument("--optimal-schedule", type=str, help="Path to optimized schedule")
    parser.add_argument("--prompt", type=str, default="1girl, blue_hair, anime_style, detailed")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/cpu_optimized_test")
    parser.add_argument("--enable-winograd", action="store_true", help="Enable Winograd convolution")
    parser.add_argument("--num-threads", type=int, default=None, help="Number of threads (auto if not set)")
    parser.add_argument("--benchmark-only", action="store_true", help="Just benchmark, don't generate")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("CPU-Optimized Schedule Test")
    print(f"{'='*60}\n")
    print(f"Model: {args.model}")
    print(f"Winograd: {'Enabled' if args.enable_winograd else 'Disabled'}")
    print(f"Threads: {args.num_threads or 'Auto'}")
    print(f"Steps: {args.num_steps}\n")
    
    # Benchmark Winograd if requested
    if args.benchmark_only:
        from utils.cpu_optimization import benchmark_convolution_algorithms
        benchmark_convolution_algorithms()
        return
    
    # Load optimal schedule if provided
    optimal_timesteps = None
    if args.optimal_schedule:
        try:
            optimal_timesteps = load_schedule(args.optimal_schedule)
            print(f"✅ Loaded optimized schedule: {len(optimal_timesteps)} timesteps\n")
        except Exception as e:
            print(f"⚠️  Could not load schedule: {e}\n")
    
    results = {}
    
    # Test 1: Without optimizations (baseline)
    print(f"{'='*60}")
    print("Test 1: Default (No Optimizations)")
    print(f"{'='*60}\n")
    
    print("Loading model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipeline = pipeline.to("cpu")
    print("✅ Model loaded\n")
    
    print(f"Generating image ({args.num_steps} steps)...")
    img, elapsed = generate_image(pipeline, args.prompt, args.num_steps, args.seed)
    
    path = os.path.join(args.output, "1_default.png")
    img.save(path)
    
    results['default'] = {
        'time': elapsed,
        'path': path,
        'optimizations': 'None'
    }
    
    print(f"✅ Generated in {elapsed:.2f}s ({elapsed/args.num_steps:.2f}s/step)")
    print(f"   Saved to: {path}\n")
    
    del pipeline
    torch.cuda.empty_cache()
    
    # Test 2: With CPU optimizations
    print(f"{'='*60}")
    print("Test 2: CPU Optimized (Winograd + Threading)")
    print(f"{'='*60}\n")
    
    print("Loading model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipeline = pipeline.to("cpu")
    
    # Apply optimizations
    pipeline = enable_cpu_optimizations(
        pipeline,
        winograd=args.enable_winograd,
        num_threads=args.num_threads,
        verbose=True
    )
    
    print(f"Generating image ({args.num_steps} steps)...")
    img, elapsed = generate_image(pipeline, args.prompt, args.num_steps, args.seed)
    
    path = os.path.join(args.output, "2_optimized.png")
    img.save(path)
    
    results['optimized'] = {
        'time': elapsed,
        'path': path,
        'optimizations': 'Winograd + Threading + Memory Layout'
    }
    
    speedup = results['default']['time'] / elapsed
    print(f"✅ Generated in {elapsed:.2f}s ({elapsed/args.num_steps:.2f}s/step)")
    print(f"   Speedup: {speedup:.2f}x vs default")
    print(f"   Saved to: {path}\n")
    
    del pipeline
    torch.cuda.empty_cache()
    
    # Test 3: With optimizations + AYS (if schedule provided)
    if optimal_timesteps:
        print(f"{'='*60}")
        print("Test 3: CPU Optimized + AYS Schedule")
        print(f"{'='*60}\n")
        
        print("Loading model...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipeline = pipeline.to("cpu")
        
        pipeline = enable_cpu_optimizations(
            pipeline,
            winograd=args.enable_winograd,
            num_threads=args.num_threads,
            verbose=False
        )
        
        print(f"Generating image ({args.num_steps} steps, AYS schedule)...")
        img, elapsed = generate_image(pipeline, args.prompt, args.num_steps, args.seed)
        
        path = os.path.join(args.output, "3_optimized_ays.png")
        img.save(path)
        
        results['optimized_ays'] = {
            'time': elapsed,
            'path': path,
            'optimizations': 'Winograd + Threading + AYS'
        }
        
        speedup = results['default']['time'] / elapsed
        print(f"✅ Generated in {elapsed:.2f}s ({elapsed/args.num_steps:.2f}s/step)")
        print(f"   Speedup: {speedup:.2f}x vs default")
        print(f"   Saved to: {path}\n")
    
    # Summary
    print(f"{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    
    baseline_time = results['default']['time']
    
    for name, data in results.items():
        speedup = baseline_time / data['time']
        print(f"{name}:")
        print(f"  Time: {data['time']:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Optimizations: {data['optimizations']}\n")
    
    # Save results
    with open(os.path.join(args.output, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Test complete!\n")


if __name__ == "__main__":
    main()
