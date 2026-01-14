#!/usr/bin/env python3
"""
Comprehensive CPU Optimization Benchmark

Tests multiple configurations with multiple images to get reliable results:
1. Baseline (no optimizations)
2. Winograd only
3. Threading only
4. channels_last only
5. All combined
6. Lightning LoRA variants

Usage:
    python scripts/benchmark_cpu_optimizations.py \
        --model martineux/janku6 \
        --num-images 3 \
        --num-steps 10
"""

import argparse
import os
import sys
import time
import json
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def apply_optimizations(pipeline, winograd=False, threading=False, channels_last=False, num_threads=6):
    """Apply specific optimizations."""
    
    if winograd:
        os.environ['NNPACK_CONVOLUTION_ALGORITHM'] = 'WINOGRAD'
        torch.backends.mkldnn.enabled = True
    else:
        os.environ.pop('NNPACK_CONVOLUTION_ALGORITHM', None)
        torch.backends.mkldnn.enabled = False
    
    if threading:
        torch.set_num_threads(num_threads)
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
    else:
        # Reset to default
        import multiprocessing
        default_threads = multiprocessing.cpu_count()
        torch.set_num_threads(default_threads)
    
    if channels_last:
        if hasattr(pipeline, 'unet'):
            pipeline.unet = pipeline.unet.to(memory_format=torch.channels_last)
        if hasattr(pipeline, 'vae'):
            pipeline.vae = pipeline.vae.to(memory_format=torch.channels_last)
    else:
        # Reset to contiguous
        if hasattr(pipeline, 'unet'):
            pipeline.unet = pipeline.unet.to(memory_format=torch.contiguous_format)
        if hasattr(pipeline, 'vae'):
            pipeline.vae = pipeline.vae.to(memory_format=torch.contiguous_format)
    
    # Always enable inference mode
    torch.set_grad_enabled(False)
    if hasattr(pipeline, 'unet'):
        pipeline.unet.eval()
    if hasattr(pipeline, 'vae'):
        pipeline.vae.eval()
    
    return pipeline


def benchmark_config(pipeline, prompt, num_steps, num_images, seed_base=42, use_lightning=False):
    """Benchmark a configuration with multiple images."""
    times = []
    
    guidance = 0.0 if use_lightning else 7.5
    
    for i in range(num_images):
        generator = torch.Generator(device="cpu").manual_seed(seed_base + i)
        
        start = time.time()
        with torch.inference_mode():
            _ = pipeline(
                prompt=prompt,
                num_inference_steps=num_steps,
                generator=generator,
                guidance_scale=guidance,
            )
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"    Image {i+1}/{num_images}: {elapsed:.2f}s")
    
    return {
        'times': times,
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive CPU Optimization Benchmark")
    parser.add_argument("--model", type=str, default="martineux/janku6")
    parser.add_argument("--prompt", type=str, default="1girl, blue_hair, anime_style")
    parser.add_argument("--num-images", type=int, default=3, help="Images per config (for averaging)")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--lightning-steps", type=int, default=4)
    parser.add_argument("--num-threads", type=int, default=6)
    parser.add_argument("--output", type=str, default="outputs/cpu_benchmark")
    parser.add_argument("--skip-lightning", action="store_true", help="Skip Lightning tests")
    parser.add_argument("--pas-config", type=str, help="Path to PAS calibration JSON (enables PAS tests)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("CPU Optimization Benchmark")
    print(f"{'='*70}\n")
    print(f"Model: {args.model}")
    print(f"Images per config: {args.num_images}")
    print(f"Steps: {args.num_steps}")
    print(f"Threads: {args.num_threads}\n")
    
    results = {}
    
    # Configuration matrix
    configs = [
        ("baseline", False, False, False),
        ("winograd_only", True, False, False),
        ("threading_only", False, True, False),
        ("channels_last_only", False, False, True),
        ("winograd_threading", True, True, False),
        ("all_optimizations", True, True, True),
    ]
    
    # Test each configuration
    for config_name, winograd, threading, channels_last in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config_name}")
        print(f"  Winograd: {winograd}, Threading: {threading}, channels_last: {channels_last}")
        print(f"{'='*70}\n")
        
        # Load pipeline
        print("  Loading model...")
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipeline = pipeline.to("cpu")
        
        # Apply optimizations
        pipeline = apply_optimizations(
            pipeline,
            winograd=winograd,
            threading=threading,
            channels_last=channels_last,
            num_threads=args.num_threads
        )
        
        # Benchmark
        print(f"  Generating {args.num_images} images...")
        stats = benchmark_config(
            pipeline,
            args.prompt,
            args.num_steps,
            args.num_images
        )
        
        results[config_name] = stats
        
        print(f"\n  Results:")
        print(f"    Mean: {stats['mean']:.2f}s ± {stats['std']:.2f}s")
        print(f"    Range: [{stats['min']:.2f}s, {stats['max']:.2f}s]")
        print(f"    Per-step: {stats['mean']/args.num_steps:.2f}s")
        
        del pipeline
        torch.cuda.empty_cache()
    
    # Lightning LoRA tests (if enabled)
    if not args.skip_lightning:
        print(f"\n{'='*70}")
        print("Lightning LoRA Tests")
        print(f"{'='*70}\n")
        
        lightning_configs = [
            ("lightning_baseline", False, False, False),
            ("lightning_optimized", True, True, True),
        ]
        
        for config_name, winograd, threading, channels_last in lightning_configs:
            print(f"\n{'='*70}")
            print(f"Testing: {config_name} ({args.lightning_steps} steps)")
            print(f"{'='*70}\n")
            
            # Load pipeline
            print("  Loading model...")
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.model,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            pipeline = pipeline.to("cpu")
            
            # Load Lightning LoRA
            print("  Loading Lightning LoRA...")
            try:
                lightning_ckpt = hf_hub_download(
                    "ByteDance/SDXL-Lightning",
                    f"sdxl_lightning_{args.lightning_steps}step_lora.safetensors"
                )
                pipeline.load_lora_weights(lightning_ckpt)
                pipeline.fuse_lora()
                
                pipeline.scheduler = EulerDiscreteScheduler.from_config(
                    pipeline.scheduler.config,
                    timestep_spacing="trailing"
                )
                print("  ✅ Lightning LoRA loaded")
            except Exception as e:
                print(f"  ❌ Lightning LoRA failed: {e}")
                continue
            
            # Apply optimizations
            pipeline = apply_optimizations(
                pipeline,
                winograd=winograd,
                threading=threading,
                channels_last=channels_last,
                num_threads=args.num_threads
            )
            
            # Benchmark
            print(f"  Generating {args.num_images} images...")
            stats = benchmark_config(
                pipeline,
                args.prompt,
                args.lightning_steps,
                args.num_images,
                use_lightning=True
            )
            
            results[config_name] = stats
            results[config_name]['steps'] = args.lightning_steps
            
            print(f"\n  Results:")
            print(f"    Mean: {stats['mean']:.2f}s ± {stats['std']:.2f}s")
            print(f"    Per-step: {stats['mean']/args.lightning_steps:.2f}s")
            
            del pipeline
            torch.cuda.empty_cache()
    
    # PAS tests (if config provided)
    if args.pas_config:
        print(f"\n{'='*70}")
        print("PAS (Phase-aware Sampling) Tests")
        print(f"{'='*70}\n")
        
        # Import PAS modules
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from algorithms.pas_pipeline import create_pas_pipeline
        
        pas_configs = [
            ("pas_only", False),
            ("pas_lightning", True),
        ]
        
        for config_name, use_lightning in pas_configs:
            if use_lightning and args.skip_lightning:
                continue
            
            steps = args.lightning_steps if use_lightning else args.num_steps
            
            print(f"\n{'='*70}")
            print(f"Testing: {config_name} ({steps} steps)")
            print(f"{'='*70}\n")
            
            # Create PAS pipeline
            try:
                pas_pipeline = create_pas_pipeline(
                    model_id=args.model,
                    pas_config_path=args.pas_config,
                    enable_lightning=use_lightning,
                    lightning_steps=args.lightning_steps,
                    device="cpu",
                    verbose=True
                )
                
                # Benchmark PAS
                times = []
                for i in range(args.num_images):
                    generator = torch.Generator(device="cpu").manual_seed(42 + i)
                    
                    start = time.time()
                    with torch.inference_mode():
                        _ = pas_pipeline(
                            prompt=args.prompt,
                            num_inference_steps=steps,
                            generator=generator,
                            guidance_scale=0.0 if use_lightning else 7.5
                        )
                    elapsed = time.time() - start
                    times.append(elapsed)
                    
                    print(f"    Image {i+1}/{args.num_images}: {elapsed:.2f}s")
                
                stats = {
                    'times': times,
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'steps': steps
                }
                
                # Get PAS statistics
                pas_stats = pas_pipeline.get_statistics()
                stats['pas_stats'] = pas_stats
                
                results[config_name] = stats
                
                print(f"\n  Results:")
                print(f"    Mean: {stats['mean']:.2f}s ± {stats['std']:.2f}s")
                print(f"    PAS speedup: {pas_stats['speedup_estimate']:.2f}x")
                print(f"    Block skip rate: {pas_stats['block_skip_rate']*100:.1f}%")
                
                del pas_pipeline
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ❌ PAS test failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    baseline_mean = results['baseline']['mean']
    
    # Sort by mean time
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mean'])
    
    print(f"{'Configuration':<25} {'Mean (s)':<12} {'Speedup':<10} {'Variance'}")
    print("-" * 70)
    
    for config_name, stats in sorted_results:
        speedup = baseline_mean / stats['mean']
        speedup_str = f"{speedup:.2f}x"
        if speedup < 1.0:
            speedup_str += " (SLOWER)"
        elif speedup > 1.2:
            speedup_str += " ✅"
        
        variance_pct = (stats['std'] / stats['mean']) * 100
        
        print(f"{config_name:<25} {stats['mean']:>7.2f} ± {stats['std']:<4.1f} {speedup_str:<14} {variance_pct:>4.1f}%")
    
    # Save results
    results_path = os.path.join(args.output, "benchmark_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types
        serializable = {}
        for k, v in results.items():
            serializable[k] = {
                'times': [float(t) for t in v['times']],
                'mean': float(v['mean']),
                'std': float(v['std']),
                'min': float(v['min']),
                'max': float(v['max']),
                'steps': v.get('steps', args.num_steps)
            }
        json.dump(serializable, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_path}\n")
    
    # Recommendations
    print("💡 Recommendations:")
    best_config = sorted_results[0][0]
    best_speedup = baseline_mean / sorted_results[0][1]['mean']
    
    if best_speedup > 1.1:
        print(f"   ✅ Use '{best_config}' for {best_speedup:.2f}x speedup")
    elif best_config == 'baseline':
        print("   ⚠️  Optimizations don't help on this CPU - stick with baseline")
    else:
        print(f"   ⚠️  Best config '{best_config}' only {best_speedup:.2f}x - marginal improvement")
    
    print()


if __name__ == "__main__":
    main()
