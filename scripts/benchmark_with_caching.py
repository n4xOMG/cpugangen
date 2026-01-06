#!/usr/bin/env python3
"""
Benchmark with Deterministic Caching and Process Pinning

Comprehensive benchmark comparing different optimization configurations:
- Baseline (no optimizations)
- Caching only
- Process pinning only
- Full optimization (caching + pinning)

Usage:
    python scripts/benchmark_with_caching.py \\
        --model segmind/SSD-1B \\
        --steps 25 \\
        --num-samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import statistics

import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.optimization import (
    CacheContext,
    optimize_for_inference,
    get_cpu_info,
)


def benchmark_configuration(
    config_name: str,
    pipeline,
    prompts: List[str],
    num_steps: int,
    enable_caching: bool,
    cache_size: int,
    warmup: int = 1,
) -> Dict[str, Any]:
    """
    Benchmark a specific configuration.
    
    Returns:
        Dictionary with timing statistics and cache info
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config_name}")
    print(f"{'='*60}")
    print(f"Caching: {'Enabled' if enable_caching else 'Disabled'}")
    print(f"Samples: {len(prompts)} + {warmup} warmup")
    
    times = []
    cache_stats = None
    
    # Use cache context if caching enabled
    if enable_caching:
        cache_ctx = CacheContext(
            enabled=True,
            max_size=cache_size,
            clear_on_enter=True,
            print_stats_on_exit=False  # We'll print manually
        )
        cache = cache_ctx.__enter__()
    else:
        cache_ctx = None
        cache = None
    
    try:
        # Warmup
        if warmup > 0:
            print(f"🔥 Warmup ({warmup} runs)...")
            for i in range(warmup):
                guidance_scale = 0.0 if hasattr(pipeline, 'apply_lightning') and pipeline.apply_lightning else 7.5
                _ = pipeline(
                    prompt=prompts[0],
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
                print(f"  Warmup {i+1}/{warmup} complete")
        
        # Reset cache stats after warmup
        if cache:
            cache.reset_statistics()
        
        # Benchmark runs
        print(f"🏃 Running {len(prompts)} benchmark iterations...")
        for i, prompt in enumerate(prompts):
            start = time.time()
            
            guidance_scale = 0.0 if hasattr(pipeline, 'apply_lightning') and pipeline.apply_lightning else 7.5
            _ = pipeline(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            )
            
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.2f}s")
        
        # Get cache statistics
        if cache:
            cache_stats = cache.get_statistics()
    
    finally:
        # Clean up cache context
        if cache_ctx:
            cache_ctx.__exit__(None, None, None)
    
    # Calculate statistics
    results = {
        "config_name": config_name,
        "num_samples": len(prompts),
        "times": times,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min": min(times),
        "max": max(times),
        "throughput": 1 / statistics.mean(times),
    }
    
    # Add cache stats if available
    if cache_stats:
        results["cache"] = cache_stats
    
    # Print summary
    print(f"\n📊 Results:")
    print(f"  Mean:   {results['mean']:.2f}s ± {results['stdev']:.2f}s")
    print(f"  Median: {results['median']:.2f}s")
    print(f"  Range:  {results['min']:.2f}s - {results['max']:.2f}s")
    print(f"  Jitter: {results['stdev']:.2f}s ({results['stdev']/results['mean']*100:.1f}%)")
    
    if cache_stats:
        print(f"\n💾 Cache Stats:")
        print(f"  Hit Rate:  {cache_stats['hit_rate']:.2%}")
        print(f"  Hits:      {cache_stats['hits']}")
        print(f"  Misses:    {cache_stats['misses']}")
        print(f"  Memory:    {cache_stats['memory_mb']:.2f} MB")
    
    return results


def compare_results(baseline: Dict, optimized: Dict) -> Dict[str, float]:
    """
    Compare baseline vs optimized configuration.
    
    Returns:
        Dictionary with improvement metrics
    """
    speedup = baseline["mean"] / optimized["mean"]
    
    # Jitter reduction (lower is better)
    baseline_jitter = baseline["stdev"]
    optimized_jitter = optimized["stdev"]
    jitter_reduction = ((baseline_jitter - optimized_jitter) / baseline_jitter * 100) if baseline_jitter > 0 else 0
    
    # Consistency improvement (coefficient of variation)
    baseline_cv = baseline["stdev"] / baseline["mean"] if baseline["mean"] > 0 else 0
    optimized_cv = optimized["stdev"] / optimized["mean"] if optimized["mean"] > 0 else 0
    cv_improvement = ((baseline_cv - optimized_cv) / baseline_cv * 100) if baseline_cv > 0 else 0
    
    return {
        "speedup": speedup,
        "time_saved_per_image": baseline["mean"] - optimized["mean"],
        "jitter_reduction_pct": jitter_reduction,
        "cv_improvement_pct": cv_improvement,
        "throughput_increase": optimized["throughput"] / baseline["throughput"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark caching and process pinning optimizations"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="martineux/janku6",
        help="Base Illustrious model",
    )
    parser.add_argument(
        "--student-unet",
        type=str,
        default="segmind/SSD-1B",
        help="Student UNet model (SSD-1B)",
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        default=True,
        help="Use hybrid (Illustrious + SSD-1B UNet)",
    )
    parser.add_argument(
        "--lightning-steps",
        type=int,
        choices=[2, 4, 8],
        default=4,
        help="Lightning steps (2, 4, or 8) - enables SDXL-Lightning LoRA",
    )
    parser.add_argument(
        "--no-lightning",
        action="store_true",
        help="Disable Lightning LoRA (use standard inference)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (default: matches --lightning-steps)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples per configuration",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=128,
        help="Cache size (number of entries)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        nargs="+",
        help="CPU cores to pin to (e.g., 0 1 2 3)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        help="Number of threads (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/caching_benchmark",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline benchmark",
    )
    parser.add_argument(
        "--skip-pinning",
        action="store_true",
        help="Skip process pinning configurations",
    )
    
    args = parser.parse_args()
    
    # Set default steps to match lightning if not specified
    if args.steps is None:
        args.steps = args.lightning_steps
    
    apply_lightning = not args.no_lightning
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Test prompts
    prompts = [
        "1girl, solo, blue_hair, beautiful_eyes, anime_style, detailed",
        "landscape, mountains, sunset, anime_style, highly_detailed",
        "2girls, sitting, cafe, talking, warm_lighting, detailed",
        "1boy, solo, black_hair, serious, school_uniform, detailed",
        "scenery, cherry_blossoms, spring, beautiful, anime_style",
        "1girl, red_hair, action_pose, dynamic, detailed_background",
        "fantasy_landscape, castle, clouds, epic, highly_detailed",
        "anime_portrait, close_up, detailed_face, beautiful_lighting",
        "night_scene, city, neon_lights, cyberpunk, detailed",
        "ocean_view, waves, sunset, peaceful, anime_style",
    ]
    prompts = prompts[:args.num_samples]
    
    print("="*70)
    print("DETERMINISTIC CACHING & PROCESS PINNING BENCHMARK")
    print("="*70)
    print(f"Base model: {args.base_model}")
    if args.use_hybrid:
        print(f"Student UNet: {args.student_unet}")
    print(f"Lightning: {'Enabled' if apply_lightning else 'Disabled'} ({args.lightning_steps}-step)")
    print(f"Steps: {args.steps}")
    print(f"Samples: {len(prompts)}")
    print(f"Cache size: {args.cache_size}")
    
    # Print CPU info
    cpu_info = get_cpu_info()
    print(f"\nCPU Info:")
    print(f"  Physical cores: {cpu_info['physical_cores']}")
    print(f"  Logical cores:  {cpu_info['logical_cores']}")
    
    # Apply process pinning if requested
    if not args.skip_pinning and (args.cores or args.num_threads):
        print("\n" + "="*70)
        print("Applying Process Optimizations")
        print("="*70)
        
        optimization_config = optimize_for_inference(
            cores=args.cores,
            num_threads=args.num_threads,
            prefer_physical=True,
            verbose=True
        )
    
    # Load pipeline (shared across all tests)
    print("\n" + "="*70)
    print("Loading Pipeline")
    print("="*70)
    
    # Load base Illustrious pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
    )
    
    # Replace UNet with SSD-1B if hybrid mode
    if args.use_hybrid:
        print(f"Loading SSD-1B UNet from {args.student_unet}...")
        from diffusers import UNet2DConditionModel
        
        try:
            ssd1b_unet = UNet2DConditionModel.from_pretrained(
                args.student_unet,
                subfolder="unet",
                torch_dtype=torch.float32,
            )
        except Exception:
            # Fallback: load entire pipeline and extract UNet
            print("  Loading entire SSD-1B pipeline...")
            ssd1b_pipe = StableDiffusionXLPipeline.from_pretrained(
                args.student_unet,
                torch_dtype=torch.float32,
            )
            ssd1b_unet = ssd1b_pipe.unet
            del ssd1b_pipe
        
        # Replace UNet
        pipeline.unet = ssd1b_unet
        print("  ✓ UNet replaced with SSD-1B")
    
    # Apply Lightning LoRA if enabled
    if apply_lightning:
        print(f"Loading Lightning {args.lightning_steps}-step LoRA...")
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from diffusers import EulerDiscreteScheduler
        
        ckpt_map = {
            2: "sdxl_lightning_2step_lora.safetensors",
            4: "sdxl_lightning_4step_lora.safetensors",
            8: "sdxl_lightning_8step_lora.safetensors"
        }
        ckpt = hf_hub_download("ByteDance/SDXL-Lightning", ckpt_map[args.lightning_steps])
        pipeline.load_lora_weights(load_file(ckpt))
        
        # Configure scheduler for Lightning
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing",
            prediction_type="epsilon",
        )
        
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()
        print("  ✓ Lightning LoRA applied and fused")
    
    pipeline = pipeline.to("cpu")
    
    # Store apply_lightning for use in benchmarks
    pipeline.apply_lightning = apply_lightning
    
    print("✓ Pipeline loaded")
    
    # Run benchmarks
    results = {}
    
    # 1. Baseline (no optimizations)
    if not args.skip_baseline:
        results["baseline"] = benchmark_configuration(
            config_name="Baseline (No Optimizations)",
            pipeline=pipeline,
            prompts=prompts,
            num_steps=args.steps,
            enable_caching=False,
            cache_size=args.cache_size,
            warmup=1,
        )
    
    # 2. Caching only
    results["caching_only"] = benchmark_configuration(
        config_name="Caching Only",
        pipeline=pipeline,
        prompts=prompts,
        num_steps=args.steps,
        enable_caching=True,
        cache_size=args.cache_size,
        warmup=1,
    )
    
    # 3. Full optimization (if process pinning was applied)
    if not args.skip_pinning and (args.cores or args.num_threads):
        # Note: Process pinning is already applied globally
        # This config combines pinning (already active) + caching
        results["full_optimization"] = benchmark_configuration(
            config_name="Full Optimization (Caching + Pinning)",
            pipeline=pipeline,
            prompts=prompts,
            num_steps=args.steps,
            enable_caching=True,
            cache_size=args.cache_size,
            warmup=1,
        )
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    # Create comparison table
    print(f"\n{'Configuration':<30} {'Mean Time':>12} {'Jitter':>10} {'Throughput':>12}")
    print("-"*70)
    
    for config_name, result in results.items():
        jitter_pct = (result['stdev'] / result['mean'] * 100) if result['mean'] > 0 else 0
        print(f"{result['config_name']:<30} {result['mean']:>10.2f}s {jitter_pct:>9.1f}% {result['throughput']:>11.3f} img/s")
    
    # Calculate improvements
    if "baseline" in results:
        baseline = results["baseline"]
        
        print("\n" + "="*70)
        print("IMPROVEMENT METRICS")
        print("="*70)
        
        for config_name, result in results.items():
            if config_name == "baseline":
                continue
            
            comparison = compare_results(baseline, result)
            
            print(f"\n{result['config_name']}:")
            print(f"  Speedup:          {comparison['speedup']:.3f}x")
            print(f"  Time saved:       {comparison['time_saved_per_image']:.2f}s per image")
            print(f"  Jitter reduction: {comparison['jitter_reduction_pct']:.1f}%")
            print(f"  Throughput gain:  {(comparison['throughput_increase']-1)*100:.1f}%")
            
            # Success indicators
            if comparison['speedup'] >= 1.10:
                print(f"  ✓ Significant speedup achieved!")
            if comparison['jitter_reduction_pct'] >= 30:
                print(f"  ✓ Excellent jitter reduction!")
    
    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Save summary report
    report_path = output_dir / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write("# Deterministic Caching & Process Pinning Benchmark\n\n")
        f.write(f"**Model**: {args.model}\n")
        f.write(f"**Steps**: {args.steps}\n")
        f.write(f"**Samples**: {len(prompts)}\n")
        f.write(f"**Cache Size**: {args.cache_size}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Configuration | Mean Time | Std Dev | Jitter % | Throughput |\n")
        f.write("|--------------|-----------|---------|----------|------------|\n")
        
        for config_name, result in results.items():
            jitter_pct = (result['stdev'] / result['mean'] * 100) if result['mean'] > 0 else 0
            f.write(f"| {result['config_name']} | {result['mean']:.2f}s | {result['stdev']:.2f}s | {jitter_pct:.1f}% | {result['throughput']:.3f} img/s |\n")
        
        if "baseline" in results:
            f.write("\n## Improvements vs Baseline\n\n")
            baseline = results["baseline"]
            
            for config_name, result in results.items():
                if config_name == "baseline":
                    continue
                
                comparison = compare_results(baseline, result)
                f.write(f"\n### {result['config_name']}\n\n")
                f.write(f"- **Speedup**: {comparison['speedup']:.3f}x\n")
                f.write(f"- **Time Saved**: {comparison['time_saved_per_image']:.2f}s per image\n")
                f.write(f"- **Jitter Reduction**: {comparison['jitter_reduction_pct']:.1f}%\n")
                f.write(f"- **Throughput Increase**: {(comparison['throughput_increase']-1)*100:.1f}%\n")
                
                if "cache" in result:
                    cache = result["cache"]
                    f.write(f"\n**Cache Statistics**:\n")
                    f.write(f"- Hit Rate: {cache['hit_rate']:.2%}\n")
                    f.write(f"- Memory Usage: {cache['memory_mb']:.2f} MB\n")
    
    print(f"💾 Report saved to: {report_path}")
    
    print("\n" + "="*70)
    print("✓ Benchmark Complete")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
