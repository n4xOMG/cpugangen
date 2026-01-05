#!/usr/bin/env python3
"""
Benchmark quantized pipeline on CPU.

Measures inference time and compares FP32 vs INT8 performance.

Usage:
    python scripts/benchmark_cpu.py \
        --num_samples 10 \
        --steps 25
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.sdxl_toe_pipeline import create_toe_pipeline
from hqpd.quantization import (
    quantize_linear_layers,
    configure_cpu_inference,
)


def benchmark_pipeline(pipeline, prompts, num_inference_steps=25, warmup=1):
    """
    Benchmark pipeline inference time.
    
    Args:
        pipeline: SDXL pipeline
        prompts: List of prompts to test
        num_inference_steps: Denoising steps
        warmup: Number of warmup runs
        
    Returns:
        Dictionary with timing statistics
    """
    print(f"   Running {len(prompts)} samples + {warmup} warmup...")
    
    times = []
    
    # Warmup
    for i in range(warmup):
        _ = pipeline(
            prompt=prompts[0],
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
        )
    
    # Actual benchmark
    for i, prompt in enumerate(prompts):
        print(f"   Sample {i+1}/{len(prompts)}: ", end='', flush=True)
        
        start = time.time()
        image = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,
        ).images[0]
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"{elapsed:.2f}s")
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'times': times,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU performance")
    
    parser.add_argument(
        "--toe_checkpoint",
        type=str,
        default="checkpoints/toe/toe_with_pooling_best.pt",
        help="TOE checkpoint path",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/vocabulary.json",
        help="Vocabulary path",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to benchmark",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare FP32 vs INT8",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CPU Benchmark: TOE + Segmind")
    print("=" * 70)
    
    # Configure CPU
    cpu_config = configure_cpu_inference()
    
    # Test prompts
    test_prompts = [
        "1girl, solo, blue_eyes, smile, detailed, anime",
        "landscape, mountains, anime_style, high_quality",
        "2girls, sitting, talking, cafe, warm_lighting",
        "1boy, solo, black_hair, serious, uniform",
        "scenery, sunset, ocean, detailed, beautiful",
    ][:args.num_samples]
    
    if not args.compare:
        # Benchmark quantized only
        print("\n📦 Loading quantized pipeline...")
        
        pipeline = create_toe_pipeline(
            toe_checkpoint_path=args.toe_checkpoint,
            vocab_path=args.vocab,
            sdxl_model_path="segmind/SSD-1B",
            device="cpu",
        )
        
        # Quantize
        pipeline.unet = quantize_linear_layers(pipeline.unet, inplace=True)
        
        print("\n⏱️  Benchmarking INT8 pipeline...")
        results = benchmark_pipeline(pipeline, test_prompts, args.steps)
        
        print("\n" + "=" * 70)
        print("Results (INT8)")
        print("=" * 70)
        print(f"Mean time: {results['mean']:.2f}s ± {results['std']:.2f}s")
        print(f"Min: {results['min']:.2f}s")
        print(f"Max: {results['max']:.2f}s")
        print(f"Median: {results['median']:.2f}s")
        
    else:
        # Compare FP32 vs INT8
        print("\n📊 Comparison Mode: FP32 vs INT8")
        
        # FP32 baseline
        print("\n1️⃣  Loading FP32 pipeline...")
        pipeline_fp32 = create_toe_pipeline(
            toe_checkpoint_path=args.toe_checkpoint,
            vocab_path=args.vocab,
            sdxl_model_path="segmind/SSD-1B",
            device="cpu",
        )
        
        print("\n⏱️  Benchmarking FP32...")
        results_fp32 = benchmark_pipeline(pipeline_fp32, test_prompts, args.steps, warmup=0)
        
        # Clear memory
        del pipeline_fp32
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # INT8
        print("\n2️⃣  Loading INT8 pipeline...")
        pipeline_int8 = create_toe_pipeline(
            toe_checkpoint_path=args.toe_checkpoint,
            vocab_path=args.vocab,
            sdxl_model_path="segmind/SSD-1B",
            device="cpu",
        )
        
        pipeline_int8.unet = quantize_linear_layers(pipeline_int8.unet, inplace=True)
        
        print("\n⏱️  Benchmarking INT8...")
        results_int8 = benchmark_pipeline(pipeline_int8, test_prompts, args.steps, warmup=0)
        
        # Comparison
        speedup = results_fp32['mean'] / results_int8['mean']
        
        print("\n" + "=" * 70)
        print("Comparison Results")
        print("=" * 70)
        print(f"\nFP32 Baseline:")
        print(f"  Mean: {results_fp32['mean']:.2f}s ± {results_fp32['std']:.2f}s")
        print(f"  Range: {results_fp32['min']:.2f}s - {results_fp32['max']:.2f}s")
        
        print(f"\nINT8 Quantized:")
        print(f"  Mean: {results_int8['mean']:.2f}s ± {results_int8['std']:.2f}s")
        print(f"  Range: {results_int8['min']:.2f}s - {results_int8['max']:.2f}s")
        
        print(f"\n🚀 Speedup: {speedup:.2f}x")
        
        if speedup >= 1.4:
            print("   ✓ Target achieved! (≥1.4x)")
        else:
            print(f"   ⚠️  Below target (1.4-1.6x expected)")
        
        print("\nPer-sample times:")
        for i, (t_fp32, t_int8) in enumerate(zip(results_fp32['times'], results_int8['times'])):
            print(f"  Sample {i+1}: {t_fp32:.2f}s → {t_int8:.2f}s ({t_fp32/t_int8:.2f}x)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
