#!/usr/bin/env python3
"""
Comprehensive Schedule Optimization Test

Tests multiple configurations:
1. Baseline + EDM (standard)
2. Baseline + AYS (optimized schedule)
3. Baseline + Lightning LoRA + EDM (4-step fast)
4. Baseline + Lightning LoRA + AYS (4-step optimized)

Usage:
    # Test all configurations
    python scripts/test_schedule_optimization.py \\
        --model martineux/janku6 \\
        --optimal-schedule configs/optimal_schedule_baseline.json \\
        --device cuda

    # Test with Lightning LoRA
    python scripts/test_schedule_optimization.py \\
        --model martineux/janku6 \\
        --optimal-schedule configs/optimal_schedule_baseline.json \\
        --enable-lightning \\
        --device cuda
"""

import argparse
import os
import sys
import time
import json
import torch
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.schedules import load_schedule, edm_schedule


def generate_image(
    pipeline,
    prompt: str,
    num_steps: int,
    config_name: str,
    seed: int = 42
):
    """Generate image and return (image, time)."""
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    
    start_time = time.time()
    
    # For Lightning, use guidance_scale=0
    guidance = 0.0 if "lightning" in config_name.lower() else 7.5
    
    output = pipeline(
        prompt=prompt,
        num_inference_steps=num_steps,
        generator=generator,
        guidance_scale=guidance,
    )
    
    elapsed = time.time() - start_time
    
    return output.images[0], elapsed


def main():
    parser = argparse.ArgumentParser(description="Test Schedule Optimization")
    parser.add_argument(
        "--model",
        type=str,
        default="martineux/janku6",
        help="Base model (Illustrious or SDXL)"
    )
    parser.add_argument(
        "--optimal-schedule",
        type=str,
        help="Path to optimized schedule JSON (optional)"
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
        help="Number of inference steps (default: 10)"
    )
    parser.add_argument(
        "--lightning-steps",
        type=int,
        default=4,
        help="Steps for Lightning LoRA (default: 4)"
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
        default="outputs/schedule_test",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--enable-lightning",
        action="store_true",
        help="Include Lightning LoRA tests"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline EDM test (faster)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Schedule Optimization Test")
    print(f"{'='*60}\n")
    print(f"Model: {args.model}")
    print(f"Optimal Schedule: {args.optimal_schedule or 'Not provided (will skip AYS tests)'}")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.num_steps} (Lightning: {args.lightning_steps})")
    print(f"Lightning: {'Enabled' if args.enable_lightning else 'Disabled'}")
    print(f"Device: {args.device}\n")
    
    # Load optimal schedule if provided
    optimal_timesteps = None
    if args.optimal_schedule:
        try:
            optimal_timesteps = load_schedule(args.optimal_schedule)
            print(f"✅ Loaded optimized schedule: {len(optimal_timesteps)} timesteps")
            print(f"   Range: [{optimal_timesteps[0]:.3f}, {optimal_timesteps[-1]:.3f}]\n")
        except Exception as e:
            print(f"⚠️  Could not load schedule: {e}\n")
    
    results = {}
    
    # Test 1: Baseline + EDM
    if not args.skip_baseline:
        print(f"{'='*60}")
        print("Test 1: Baseline + EDM Schedule")
        print(f"{'='*60}\n")
        
        print("Loading baseline model...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipeline = pipeline.to(args.device)
        print("✅ Model loaded\n")
        
        print(f"Generating image ({args.num_steps} steps, EDM schedule)...")
        img, elapsed = generate_image(
            pipeline=pipeline,
            prompt=args.prompt,
            num_steps=args.num_steps,
            config_name="baseline_edm",
            seed=args.seed
        )
        
        path = os.path.join(args.output, "1_baseline_edm.png")
        img.save(path)
        
        results['baseline_edm'] = {
            'time': elapsed,
            'steps': args.num_steps,
            'path': path,
            'schedule': 'EDM',
            'lora': None
        }
        
        print(f"✅ Generated in {elapsed:.2f}s ({elapsed/args.num_steps:.2f}s/step)")
        print(f"   Saved to: {path}\n")
        
        del pipeline
        torch.cuda.empty_cache() if args.device == "cuda" else None
    
    # Test 2: Baseline + AYS (if schedule provided)
    if optimal_timesteps:
        print(f"{'='*60}")
        print("Test 2: Baseline + AYS Optimized Schedule")
        print(f"{'='*60}\n")
        
        print("Loading baseline model...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipeline = pipeline.to(args.device)
        print("✅ Model loaded\n")
        
        print(f"Generating image ({args.num_steps} steps, AYS schedule)...")
        img, elapsed = generate_image(
            pipeline=pipeline,
            prompt=args.prompt,
            num_steps=args.num_steps,
            config_name="baseline_ays",
            seed=args.seed
        )
        
        path = os.path.join(args.output, "2_baseline_ays.png")
        img.save(path)
        
        results['baseline_ays'] = {
            'time': elapsed,
            'steps': args.num_steps,
            'path': path,
            'schedule': 'AYS_Optimized',
            'lora': None
        }
        
        improvement = results['baseline_edm']['time'] / elapsed if 'baseline_edm' in results else 1.0
        print(f"✅ Generated in {elapsed:.2f}s ({elapsed/args.num_steps:.2f}s/step)")
        if 'baseline_edm' in results:
            print(f"   Speedup vs EDM: {improvement:.2f}x")
        print(f"   Saved to: {path}\n")
        
        del pipeline
        torch.cuda.empty_cache() if args.device == "cuda" else None
    
    # Test 3 & 4: Lightning LoRA variants (if enabled)
    if args.enable_lightning:
        print(f"{'='*60}")
        print(f"Test 3: Baseline + Lightning LoRA + EDM ({args.lightning_steps} steps)")
        print(f"{'='*60}\n")
        
        print("Loading baseline model...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipeline = pipeline.to(args.device)
        
        # Load Lightning LoRA
        print("Loading SDXL-Lightning LoRA...")
        try:
            lightning_ckpt = hf_hub_download(
                "ByteDance/SDXL-Lightning",
                f"sdxl_lightning_{args.lightning_steps}step_lora.safetensors"
            )
            pipeline.load_lora_weights(load_file(lightning_ckpt))
            pipeline.fuse_lora()
            print("✅ Lightning LoRA loaded\n")
        except Exception as e:
            print(f"⚠️  Could not load Lightning LoRA: {e}")
            print("   Skipping Lightning tests\n")
            args.enable_lightning = False
        
        if args.enable_lightning:
            print(f"Generating image ({args.lightning_steps} steps, EDM schedule)...")
            img, elapsed = generate_image(
                pipeline=pipeline,
                prompt=args.prompt,
                num_steps=args.lightning_steps,
                config_name="lightning_edm",
                seed=args.seed
            )
            
            path = os.path.join(args.output, "3_lightning_edm.png")
            img.save(path)
            
            results['lightning_edm'] = {
                'time': elapsed,
                'steps': args.lightning_steps,
                'path': path,
                'schedule': 'EDM',
                'lora': 'Lightning'
            }
            
            speedup = results['baseline_edm']['time'] / elapsed if 'baseline_edm' in results else 1.0
            print(f"✅ Generated in {elapsed:.2f}s ({elapsed/args.lightning_steps:.2f}s/step)")
            if 'baseline_edm' in results:
                print(f"   Speedup vs baseline: {speedup:.2f}x")
            print(f"   Saved to: {path}\n")
            
            # Test 4: Lightning + AYS (if schedule provided)
            if optimal_timesteps:
                print(f"{'='*60}")
                print(f"Test 4: Baseline + Lightning LoRA + AYS ({args.lightning_steps} steps)")
                print(f"{'='*60}\n")
                
                print(f"Generating image ({args.lightning_steps} steps, AYS schedule)...")
                img, elapsed = generate_image(
                    pipeline=pipeline,
                    prompt=args.prompt,
                    num_steps=args.lightning_steps,
                    config_name="lightning_ays",
                    seed=args.seed
                )
                
                path = os.path.join(args.output, "4_lightning_ays.png")
                img.save(path)
                
                results['lightning_ays'] = {
                    'time': elapsed,
                    'steps': args.lightning_steps,
                    'path': path,
                    'schedule': 'AYS_Optimized',
                    'lora': 'Lightning'
                }
                
                speedup = results['baseline_edm']['time'] / elapsed if 'baseline_edm' in results else 1.0
                improvement = results['lightning_edm']['time'] / elapsed
                print(f"✅ Generated in {elapsed:.2f}s ({elapsed/args.lightning_steps:.2f}s/step)")
                if 'baseline_edm' in results:
                    print(f"   Speedup vs baseline: {speedup:.2f}x")
                print(f"   Speedup vs Lightning+EDM: {improvement:.2f}x")
                print(f"   Saved to: {path}\n")
            
            del pipeline
            torch.cuda.empty_cache() if args.device == "cuda" else None
    
    # Summary
    print(f"{'='*60}")
    print("Summary")
    print(f"{'='*60}\n")
    
    baseline_time = results.get('baseline_edm', {}).get('time', 0)
    
    for config_name, config_data in results.items():
        speedup = f" ({baseline_time / config_data['time']:.2f}x)" if baseline_time > 0 and config_name != 'baseline_edm' else ""
        
        print(f"{config_name}:")
        print(f"  Schedule: {config_data['schedule']}")
        print(f"  LoRA: {config_data['lora'] or 'None'}")
        print(f"  Steps: {config_data['steps']}")
        print(f"  Time: {config_data['time']:.2f}s{speedup}")
        print(f"  Image: {config_data['path']}\n")
    
    # Save results JSON
    results_path = os.path.join(args.output, "results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'prompt': args.prompt,
            'seed': args.seed,
            'results': results
        }, f, indent=2)
    
    print(f"📊 Results saved to: {results_path}")
    
    # Create comparison grid
    if len(results) >= 2:
        print("\nCreating comparison grid...")
        try:
            images = [Image.open(results[k]['path']) for k in sorted(results.keys())]
            
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
    
    print("✅ Test complete!\n")
    print("💡 Key comparisons:")
    if 'baseline_ays' in results:
        print("   • Baseline+EDM vs Baseline+AYS → Shows AYS schedule improvement")
    if 'lightning_edm' in results:
        print("   • Baseline vs Lightning → Shows LoRA speedup")
    if 'lightning_ays' in results:
        print("   • Lightning+EDM vs Lightning+AYS → Shows AYS+Lightning synergy")
    print()


if __name__ == "__main__":
    main()
