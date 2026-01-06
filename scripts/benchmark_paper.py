#!/usr/bin/env python3
"""
Publication-Quality Benchmark Script

Tests all optimization configurations for research paper comparison:
- Baseline A: Original Illustrious SDXL (20 steps, no optimizations)
- Baseline B: Illustrious + SDXL-Lightning (4 steps)
- Baseline C: Illustrious + SSD-1B UNet + SDXL-Lightning (hybrid)
- Baseline D: Illustrious + SSD-1B UNet + SDXL-Lightning + Caching (full)

Usage:
    python scripts/benchmark_paper.py \\
        --num-prompts 2 \\
        --output outputs/paper_benchmark
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
import psutil
import numpy as np
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.optimization import CacheContext, get_global_cache


def clear_memory():
    """Aggressively clear memory between benchmark runs."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(2)  # Let OS reclaim memory


def get_memory_usage() -> float:
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def benchmark_configuration(
    config_name: str,
    load_fn: callable,
    prompts: List[str],
    num_steps: int,
    guidance_scale: float,
    warmup: int = 1,
) -> Dict[str, Any]:
    """
    Benchmark a single configuration.
    
    Args:
        config_name: Configuration name
        load_fn: Function that returns loaded pipeline
        prompts: List of prompts to test
        num_steps: Inference steps
        guidance_scale: CFG scale
        warmup: Warmup iterations
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"\n{'='*70}")
    print(f"CONFIGURATION: {config_name}")
    print(f"{'='*70}")
    print(f"Steps: {num_steps}, Guidance: {guidance_scale}")
    print(f"Prompts: {len(prompts)} + {warmup} warmup")
    
    memory_before = get_memory_usage()
    print(f"Memory before loading: {memory_before:.2f} GB")
    
    # Load pipeline
    print("\nLoading pipeline...")
    start_load = time.time()
    pipeline = load_fn()
    load_time = time.time() - start_load
    
    memory_after_load = get_memory_usage()
    print(f"✓ Pipeline loaded in {load_time:.1f}s")
    print(f"Memory after loading: {memory_after_load:.2f} GB (+{memory_after_load-memory_before:.2f} GB)")
    
    # Warmup
    if warmup > 0:
        print(f"\n🔥 Warmup ({warmup} runs)...")
        for i in range(warmup):
            _ = pipeline(
                prompt=prompts[0],
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            )
            print(f"  Warmup {i+1}/{warmup} complete")
    
    # Reset cache if using caching
    if hasattr(pipeline, '_cache_context'):
        pipeline._cache_context.reset_statistics()
    
    # Benchmark runs
    times = []
    memory_peaks = []
    
    print(f"\n🏃 Running {len(prompts)} benchmark iterations...")
    for i, prompt in enumerate(prompts):
        memory_before_iter = get_memory_usage()
        
        start = time.time()
        _ = pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        elapsed = time.time() - start
        
        memory_after_iter = get_memory_usage()
        memory_peak = memory_after_iter
        
        times.append(elapsed)
        memory_peaks.append(memory_peak)
        
        print(f"  [{i+1}/{len(prompts)}] {elapsed:.2f}s | Peak: {memory_peak:.2f} GB")
    
    # Get cache statistics if available
    cache_stats = None
    if hasattr(pipeline, '_cache_context'):
        cache_stats = pipeline._cache_context.get_statistics()
    
    # Cleanup pipeline
    del pipeline
    clear_memory()
    
    memory_after_cleanup = get_memory_usage()
    print(f"\nMemory after cleanup: {memory_after_cleanup:.2f} GB")
    
    # Calculate statistics
    results = {
        "config_name": config_name,
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "num_prompts": len(prompts),
        "load_time": load_time,
        "times": times,
        "mean_time": statistics.mean(times),
        "median_time": statistics.median(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_time": min(times),
        "max_time": max(times),
        "memory_loaded": memory_after_load,
        "memory_peak_avg": statistics.mean(memory_peaks),
        "memory_peak_max": max(memory_peaks),
    }
    
    if cache_stats:
        results["cache"] = cache_stats
    
    # Print summary
    print(f"\n📊 Results:")
    print(f"  Mean:   {results['mean_time']:.2f}s ± {results['stdev']:.2f}s")
    print(f"  Median: {results['median_time']:.2f}s")
    print(f"  Range:  {results['min_time']:.2f}s - {results['max_time']:.2f}s")
    print(f"  Jitter: {results['stdev']/results['mean_time']*100:.1f}%")
    
    if cache_stats:
        print(f"\n💾 Cache Stats:")
        print(f"  Hit Rate:  {cache_stats['hit_rate']:.2%}")
        print(f"  Hits:      {cache_stats['hits']}")
        print(f"  Misses:    {cache_stats['misses']}")
        print(f"  Memory:    {cache_stats['memory_mb']:.2f} MB")
    
    return results


def load_baseline_a() -> StableDiffusionXLPipeline:
    """Baseline A: Original Illustrious SDXL (no optimizations)."""
    print("Loading Baseline A: Original Illustrious SDXL...")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "martineux/janku6",
        torch_dtype=torch.float32,
    )
    pipeline = pipeline.to("cpu")
    
    print("  ✓ Original Illustrious SDXL loaded")
    return pipeline


def load_baseline_b() -> StableDiffusionXLPipeline:
    """Baseline B: Illustrious + SDXL-Lightning."""
    print("Loading Baseline B: Illustrious + SDXL-Lightning...")
    
    # Load base
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "martineux/janku6",
        torch_dtype=torch.float32,
    )
    
    # Apply Lightning LoRA
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
    pipeline.load_lora_weights(load_file(ckpt))
    
    # Configure scheduler
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()
    
    pipeline = pipeline.to("cpu")
    
    print("  ✓ Illustrious + Lightning LoRA loaded")
    return pipeline


def load_baseline_c() -> StableDiffusionXLPipeline:
    """Baseline C: Illustrious + SSD-1B UNet + SDXL-Lightning (hybrid)."""
    print("Loading Baseline C: Illustrious + SSD-1B UNet + Lightning...")
    
    # Load base
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "martineux/janku6",
        torch_dtype=torch.float32,
    )
    
    # Replace UNet with SSD-1B
    print("  Loading SSD-1B UNet...")
    try:
        ssd1b_unet = UNet2DConditionModel.from_pretrained(
            "segmind/SSD-1B",
            subfolder="unet",
            torch_dtype=torch.float32,
        )
    except Exception:
        print("  Loading entire SSD-1B pipeline...")
        ssd1b_pipe = StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B",
            torch_dtype=torch.float32,
        )
        ssd1b_unet = ssd1b_pipe.unet
        del ssd1b_pipe
        clear_memory()
    
    pipeline.unet = ssd1b_unet
    print("  ✓ UNet replaced with SSD-1B")
    
    # Apply Lightning LoRA
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
    pipeline.load_lora_weights(load_file(ckpt))
    
    # Configure scheduler
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()
    
    pipeline = pipeline.to("cpu")
    
    print("  ✓ Hybrid pipeline loaded")
    return pipeline


def load_baseline_d() -> StableDiffusionXLPipeline:
    """Baseline D: Illustrious + SSD-1B + Lightning + Caching (full optimization)."""
    print("Loading Baseline D: Full Optimization (+ Caching)...")
    
    # Initialize cache
    cache = get_global_cache(max_size=256, enabled=True)
    print("  ✓ Cache initialized (256 entries)")
    
    # Load base
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "martineux/janku6",
        torch_dtype=torch.float32,
    )
    
    # Replace UNet with SSD-1B
    print("  Loading SSD-1B UNet...")
    try:
        ssd1b_unet = UNet2DConditionModel.from_pretrained(
            "segmind/SSD-1B",
            subfolder="unet",
            torch_dtype=torch.float32,
        )
    except Exception:
        print("  Loading entire SSD-1B pipeline...")
        ssd1b_pipe = StableDiffusionXLPipeline.from_pretrained(
            "segmind/SSD-1B",
            torch_dtype=torch.float32,
        )
        ssd1b_unet = ssd1b_pipe.unet
        del ssd1b_pipe
        clear_memory()
    
    pipeline.unet = ssd1b_unet
    print("  ✓ UNet replaced with SSD-1B")
    
    # Apply Lightning LoRA
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
    pipeline.load_lora_weights(load_file(ckpt))
    
    # Configure scheduler
    pipeline.scheduler = EulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    pipeline.fuse_lora()
    pipeline.unload_lora_weights()
    
    pipeline = pipeline.to("cpu")
    
    # Attach cache context
    cache_ctx = CacheContext(enabled=True, max_size=256, clear_on_enter=True, print_stats_on_exit=False)
    cache_ctx.__enter__()
    pipeline._cache_context = cache_ctx
    
    print("  ✓ Full optimization pipeline loaded")
    return pipeline


def create_comparison_table(results: List[Dict[str, Any]], output_dir: Path):
    """Create publication-ready comparison table."""
    
    # Create markdown table
    md_path = output_dir / "comparison_table.md"
    with open(md_path, "w") as f:
        f.write("# Optimization Pipeline Comparison\n\n")
        f.write("**Hardware**: Intel i5-1135G7 (4-core CPU)\n\n")
        f.write("## Performance Comparison\n\n")
        
        # Main table
        f.write("| Configuration | Steps | Mean Time (s) | Std Dev (s) | Speedup | Memory (GB) | Cache Hit Rate |\n")
        f.write("|--------------|-------|---------------|-------------|---------|-------------|----------------|\n")
        
        baseline_time = results[0]["mean_time"]
        
        for r in results:
            speedup = baseline_time / r["mean_time"]
            cache_rate = r.get("cache", {}).get("hit_rate", 0) * 100 if "cache" in r else 0
            
            f.write(f"| {r['config_name']} | {r['num_steps']} | {r['mean_time']:.2f} | {r['stdev']:.2f} | "
                   f"{speedup:.2f}x | {r['memory_peak_avg']:.2f} | {cache_rate:.1f}% |\n")
        
        f.write("\n## Detailed Metrics\n\n")
        for r in results:
            f.write(f"### {r['config_name']}\n\n")
            f.write(f"- **Steps**: {r['num_steps']}\n")
            f.write(f"- **Guidance Scale**: {r['guidance_scale']}\n")
            f.write(f"- **Mean Time**: {r['mean_time']:.2f}s ± {r['stdev']:.2f}s\n")
            f.write(f"- **Median Time**: {r['median_time']:.2f}s\n")
            f.write(f"- **Range**: {r['min_time']:.2f}s - {r['max_time']:.2f}s\n")
            f.write(f"- **Jitter**: {r['stdev']/r['mean_time']*100:.1f}%\n")
            f.write(f"- **Memory (Loaded)**: {r['memory_loaded']:.2f} GB\n")
            f.write(f"- **Memory (Peak Avg)**: {r['memory_peak_avg']:.2f} GB\n")
            
            if "cache" in r:
                cache = r["cache"]
                f.write(f"- **Cache Hit Rate**: {cache['hit_rate']:.2%}\n")
                f.write(f"- **Cache Hits/Misses**: {cache['hits']}/{cache['misses']}\n")
                f.write(f"- **Cache Memory**: {cache['memory_mb']:.2f} MB\n")
            
            f.write("\n")
    
    print(f"\n✓ Comparison table saved to: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Publication-quality benchmark")
    
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=2,
        help="Number of test prompts per configuration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/paper_benchmark",
        help="Output directory",
    )
    parser.add_argument(
        "--skip-baseline-a",
        action="store_true",
        help="Skip Baseline A (original SDXL, very slow)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Test prompts
    prompts = [
        "1girl, blue_hair, anime_style, detailed, beautiful_eyes",
        "anime landscape, mountains, sunset, highly_detailed",
        "1boy, fantasy_outfit, action_pose, dynamic, detailed",
        "anime_portrait, close_up, detailed_face, beautiful_lighting",
        "2girls, cafe, sitting, talking, warm_atmosphere, detailed",
    ]
    prompts = prompts[:args.num_prompts]
    
    print("="*70)
    print("PUBLICATION BENCHMARK - OPTIMIZATION COMPARISON")
    print("="*70)
    print(f"System: Intel i5-1135G7 (4-core CPU)")
    print(f"Prompts per config: {len(prompts)}")
    print(f"Output: {output_dir}")
    print()
    
    results = []
    
    # Baseline A: Original Illustrious SDXL (50 steps)
    if not args.skip_baseline_a:
        try:
            result_a = benchmark_configuration(
                config_name="Baseline A: Original Illustrious",
                load_fn=load_baseline_a,
                prompts=prompts,
                num_steps=20,
                guidance_scale=7.5,
                warmup=1,
            )
            results.append(result_a)
        except Exception as e:
            print(f"\n⚠️ Baseline A failed: {e}")
            print("Continuing with other configurations...")
    else:
        print("\nℹ️ Skipping Baseline A (--skip-baseline-a)")
    
    # Baseline B: Illustrious + Lightning (4 steps)
    result_b = benchmark_configuration(
        config_name="Baseline B: Illustrious + Lightning",
        load_fn=load_baseline_b,
        prompts=prompts,
        num_steps=4,
        guidance_scale=0.0,  # Lightning requires 0.0
        warmup=1,
    )
    results.append(result_b)
    
    # Baseline C: Hybrid (Illustrious + SSD-1B + Lightning)
    result_c = benchmark_configuration(
        config_name="Baseline C: Hybrid (+ SSD-1B UNet)",
        load_fn=load_baseline_c,
        prompts=prompts,
        num_steps=4,
        guidance_scale=0.0,
        warmup=1,
    )
    results.append(result_c)
    
    # Baseline D: Full Optimization (+ Caching)
    result_d = benchmark_configuration(
        config_name="Baseline D: Full Optimization (+ Caching)",
        load_fn=load_baseline_d,
        prompts=prompts,
        num_steps=4,
        guidance_scale=0.0,
        warmup=1,
    )
    results.append(result_d)
    
    # Save raw results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✓ Raw results saved to: {results_path}")
    
    # Create comparison table
    create_comparison_table(results, output_dir)
    
    # Print final summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    
    baseline_time = results[0]["mean_time"]
    print(f"\nSpeedup Summary (vs {'Baseline A' if not args.skip_baseline_a else 'Baseline B'}):")
    for r in results:
        speedup = baseline_time / r["mean_time"]
        print(f"  {r['config_name']}: {speedup:.2f}x ({r['mean_time']:.1f}s)")
    
    print(f"\nAll results saved to: {output_dir}/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
