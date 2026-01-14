#!/usr/bin/env python3
"""
Test PAS + Lightning LoRA

Benchmark Phase-aware Sampling combined with Lightning LoRA.

Usage:
    # Test PAS alone
    python scripts/test_pas.py \
        --model martineux/janku6 \
        --pas-config configs/pas_calibration.json \
        --prompt "1girl, blue_hair, anime_style"
    
    # Test PAS + Lightning
    python scripts/test_pas.py \
        --model martineux/janku6 \
        --pas-config configs/pas_calibration.json \
        --enable-lightning \
        --lightning-steps 4
"""

import argparse
import os
import sys
import time
import json
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.pas_pipeline import create_pas_pipeline
from diffusers import StableDiffusionXLPipeline


def benchmark_baseline(model_id, prompt, num_steps, device, seed=42):
    """Baseline without PAS."""
    print(f"\n{'='*60}")
    print("Baseline (No PAS)")
    print(f"{'='*60}\n")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipeline = pipeline.to(device)
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"Generating ({num_steps} steps)...")
    start = time.time()
    
    output = pipeline(
        prompt=prompt,
        num_inference_steps=num_steps,
        generator=generator,
        guidance_scale=7.5
    )
    
    elapsed = time.time() - start
    
    print(f"✅ Generated in {elapsed:.2f}s ({elapsed/num_steps:.2f}s/step)\n")
    
    del pipeline
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return output.images[0], elapsed


def benchmark_pas(model_id, pas_config, prompt, num_steps, enable_lightning, lightning_steps, device, seed=42):
    """PAS (with optional Lightning)."""
    config_name = "PAS + Lightning" if enable_lightning else "PAS Only"
    steps = lightning_steps if enable_lightning else num_steps
    
    print(f"\n{'='*60}")
    print(config_name)
    print(f"{'='*60}\n")
    
    pipeline = create_pas_pipeline(
        model_id=model_id,
        pas_config_path=pas_config,
        enable_lightning=enable_lightning,
        lightning_steps=lightning_steps,
        device=device,
        verbose=True
    )
    
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"Generating ({steps} steps)...")
    start = time.time()
    
    output = pipeline(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
        guidance_scale=0.0 if enable_lightning else 7.5
    )
    
    elapsed = time.time() - start
    
    # Get PAS statistics
    stats = pipeline.get_statistics()
    
    print(f"\n✅ Generated in {elapsed:.2f}s ({elapsed/steps:.2f}s/step)")
    print(f"\nPAS Statistics:")
    print(f"  Full U-Net calls: {stats['full_unet_calls']}")
    print(f"  Partial U-Net calls: {stats['partial_unet_calls']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Speedup estimate: {stats['speedup_estimate']:.2f}x")
    print(f"  Block skip rate: {stats['block_skip_rate']*100:.1f}%\n")
    
    del pipeline
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return output.images[0], elapsed, stats


def main():
    parser = argparse.ArgumentParser(description="Test PAS + Lightning")
    parser.add_argument("--model", type=str, default="martineux/janku6")
    parser.add_argument("--pas-config", type=str, required=True, help="Path to PAS calibration JSON")
    parser.add_argument("--prompt", type=str, default="1girl, blue_hair, anime_style, detailed, masterpiece")
    parser.add_argument("--num-steps", type=int, default=20, help="Steps for baseline/PAS")
    parser.add_argument("--lightning-steps", type=int, default=4, help="Steps for Lightning")
    parser.add_argument("--enable-lightning", action="store_true", help="Enable Lightning LoRA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/pas_test")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline (faster)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("PAS + Lightning Test")
    print(f"{'='*60}\n")
    print(f"Model: {args.model}")
    print(f"PAS Config: {args.pas_config}")
    print(f"Lightning: {'Enabled' if args.enable_lightning else 'Disabled'}")
    print(f"Device: {args.device}\n")
    
    results = {}
    
    # Test 1: Baseline (optional)
    if not args.skip_baseline:
        img, elapsed = benchmark_baseline(
            args.model,
            args.prompt,
            args.num_steps,
            args.device,
            args.seed
        )
        
        path = os.path.join(args.output, "1_baseline.png")
        img.save(path)
        
        results['baseline'] = {
            'time': elapsed,
            'steps': args.num_steps,
            'path': path
        }
    
    # Test 2: PAS (with optional Lightning)
    img, elapsed, stats = benchmark_pas(
        args.model,
        args.pas_config,
        args.prompt,
        args.num_steps,
        args.enable_lightning,
        args.lightning_steps,
        args.device,
        args.seed
    )
    
    config_name = "pas_lightning" if args.enable_lightning else "pas_only"
    path = os.path.join(args.output, f"2_{config_name}.png")
    img.save(path)
    
    results[config_name] = {
        'time': elapsed,
        'steps': args.lightning_steps if args.enable_lightning else args.num_steps,
        'path': path,
        'pas_stats': stats
    }
    
    # Summary
    print(f"{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    
    baseline_time = results.get('baseline', {}).get('time', 0)
    
    for name, data in results.items():
        speedup = ""
        if baseline_time > 0 and name != 'baseline':
            speedup = f" ({baseline_time / data['time']:.2f}x faster)"
        
        print(f"{name}:")
        print(f"  Time: {data['time']:.2f}s{speedup}")
        print(f"  Steps: {data['steps']}")
        print(f"  Image: {data['path']}")
        
        if 'pas_stats' in data:
            print(f"  PAS speedup estimate: {data['pas_stats']['speedup_estimate']:.2f}x")
        print()
    
    # Save results
    results_path = os.path.join(args.output, "results.json")
    
    # Make stats serializable
    for key in results:
        if 'pas_stats' in results[key]:
            results[key]['pas_stats'] = {k: float(v) if isinstance(v, (int, float)) else v 
                                          for k, v in results[key]['pas_stats'].items()}
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved to: {results_path}\n")
    
    # Recommendations
    if baseline_time > 0 and config_name in results:
        total_speedup = baseline_time / results[config_name]['time']
        
        print("💡 Results:")
        print(f"   Total speedup: {total_speedup:.2f}x")
        
        if total_speedup > 5:
            print(f"   🚀 EXCELLENT! {total_speedup:.1f}x speedup achieved!")
        elif total_speedup > 2:
            print(f"   ✅ Good speedup: {total_speedup:.1f}x")
        else:
            print(f"   ⚠️  Modest speedup: {total_speedup:.1f}x - try tuning PAS parameters")
        
        print()


if __name__ == "__main__":
    main()
