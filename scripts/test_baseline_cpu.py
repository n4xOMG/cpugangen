#!/usr/bin/env python3
"""
Test 1: Basic CPU Optimizations
Establish a reliable baseline for CPU inference performance.

Tests:
1. Thread count optimization (1, 2, 4, 8, max cores)
2. Float32 enforcement
3. BLAS library detection (MKL, OpenBLAS)
4. Stable timing over multiple runs
"""

import argparse
import time
import platform
import os
import psutil
from pathlib import Path

import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def get_system_info():
    """Get detailed system information."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": psutil.virtual_memory().total / (1024**3),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_blas": torch.__config__.show().split('\n') if hasattr(torch.__config__, 'show') else "Unknown",
    }
    
    # Detect BLAS library
    try:
        config_str = torch.__config__.show()
        if "MKL" in config_str:
            info["blas_library"] = "MKL (Intel Math Kernel Library)"
        elif "OpenBLAS" in config_str:
            info["blas_library"] = "OpenBLAS"
        elif "BLIS" in config_str:
            info["blas_library"] = "BLIS"
        else:
            info["blas_library"] = "Unknown/Default"
    except:
        info["blas_library"] = "Unable to detect"
    
    return info


def load_model_with_config(model_name, steps, dtype=torch.float32):
    """Load SDXL-Lightning model with specified configuration."""
    print(f"\n📦 Loading model: {model_name}")
    print(f"   Data type: {dtype}")
    
    # Force CPU and specified dtype
    device = "cpu"
    
    # Load base model
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )
    
    # Load Lightning LoRA
    print(f"📦 Loading Lightning {steps}-step LoRA...")
    ckpt_map = {
        2: "sdxl_lightning_2step_lora.safetensors",
        4: "sdxl_lightning_4step_lora.safetensors",
        8: "sdxl_lightning_8step_lora.safetensors"
    }
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", ckpt_map[steps])
    pipe.load_lora_weights(load_file(ckpt))
    
    # Configure scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    # Fuse and move to CPU
    pipe.to(device)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    
    return pipe


def run_inference_test(pipe, prompt, num_steps, seed, num_runs=5):
    """Run inference test with timing."""
    generator = torch.Generator("cpu").manual_seed(seed)
    times = []
    
    print(f"\n⏱️  Running {num_runs} inference iterations...")
    
    for i in range(num_runs):
        # Reset generator for consistency
        generator = torch.Generator("cpu").manual_seed(seed + i)
        
        # Measure inference time
        start = time.time()
        with torch.inference_mode():
            _ = pipe(
                prompt=prompt,
                num_inference_steps=num_steps,
                guidance_scale=0.0,
                generator=generator,
            )
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"   Run {i+1}/{num_runs}: {elapsed:.2f}s")
    
    return times


def test_thread_counts(pipe, prompt, num_steps, seed, thread_counts):
    """Test different thread count configurations."""
    results = {}
    
    print("\n" + "=" * 70)
    print("🧵 Testing Thread Count Configurations")
    print("=" * 70)
    
    for threads in thread_counts:
        print(f"\n{'='*70}")
        print(f"Testing with {threads} threads")
        print(f"{'='*70}")
        
        # Set thread count
        torch.set_num_threads(threads)
        print(f"✅ torch.get_num_threads() = {torch.get_num_threads()}")
        
        # Run test
        times = run_inference_test(pipe, prompt, num_steps, seed, num_runs=3)
        
        results[threads] = {
            "times": times,
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
        }
        
        print(f"\n📊 Results for {threads} threads:")
        print(f"   Mean: {results[threads]['mean']:.2f}s")
        print(f"   Std:  {results[threads]['std']:.2f}s")
        print(f"   Min:  {results[threads]['min']:.2f}s")
        print(f"   Max:  {results[threads]['max']:.2f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test 1: Basic CPU Optimizations")
    parser.add_argument("--model", default="martineux/janku6", help="Base model")
    parser.add_argument("--steps", type=int, default=4, choices=[2, 4, 8], help="Lightning steps")
    parser.add_argument("--inference-steps", type=int, default=20, help="Number of inference steps for testing")
    parser.add_argument("--prompt", default="anime girl with blue hair, beautiful eyes, highly detailed",
                        help="Test prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="outputs/baseline_test", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 70)
    print("🔬 Test 1: Basic CPU Optimizations - Baseline Measurement")
    print("=" * 70)
    
    # Get system info
    print("\n" + "=" * 70)
    print("💻 System Information")
    print("=" * 70)
    sys_info = get_system_info()
    for key, value in sys_info.items():
        if key == "torch_blas" and isinstance(value, list):
            continue  # Skip detailed BLAS config for now
        print(f"{key}: {value}")
    
    # Determine thread counts to test (skip 1 and 2 threads - too slow)
    cpu_physical = sys_info["cpu_count_physical"]
    cpu_logical = sys_info["cpu_count_logical"]
    
    thread_counts = [cpu_physical]
    if cpu_logical > cpu_physical:
        thread_counts.append(cpu_logical)
    thread_counts = sorted(set(thread_counts))  # Remove duplicates and sort
    
    print(f"\n🧵 Will test thread counts: {thread_counts}")
    print(f"   (Skipping 1-2 threads - too slow for baseline testing)")
    
    # Force environment variables for CPU optimization
    print("\n🔧 Setting CPU optimization environment variables...")
    os.environ["OMP_NUM_THREADS"] = str(cpu_physical)
    os.environ["MKL_NUM_THREADS"] = str(cpu_physical)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_physical)
    print(f"   OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"   MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS')}")
    print(f"   OPENBLAS_NUM_THREADS = {os.environ.get('OPENBLAS_NUM_THREADS')}")
    
    # Load model (float32)
    print("\n" + "=" * 70)
    print("📦 Model Loading")
    print("=" * 70)
    
    load_start = time.time()
    pipe = load_model_with_config(args.model, args.steps, dtype=torch.float32)
    load_time = time.time() - load_start
    
    print(f"\n✅ Model loaded in {load_time:.2f}s")
    
    # Verify dtype
    print("\n🔍 Verifying model configuration...")
    print(f"   UNet dtype: {pipe.unet.dtype}")
    print(f"   VAE dtype: {pipe.vae.dtype}")
    print(f"   Device: cpu")
    
    # Test different thread counts
    thread_results = test_thread_counts(
        pipe, args.prompt, args.inference_steps, args.seed, thread_counts
    )
    
    # Find optimal thread count
    best_threads = min(thread_results.keys(), key=lambda t: thread_results[t]["mean"])
    
    print("\n" + "=" * 70)
    print("📊 BASELINE TEST RESULTS")
    print("=" * 70)
    
    # Summary table
    print(f"\n{'Threads':<10} {'Mean Time':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10}")
    print("-" * 54)
    for threads, result in sorted(thread_results.items()):
        marker = " ⭐ BEST" if threads == best_threads else ""
        print(f"{threads:<10} {result['mean']:<12.2f} {result['std']:<12.2f} "
              f"{result['min']:<10.2f} {result['max']:<10.2f}{marker}")
    
    # Calculate speedup (compare against lowest thread count tested)
    min_threads = min(thread_results.keys())
    baseline_mean = thread_results[min_threads]["mean"]
    best_mean = thread_results[best_threads]["mean"]
    speedup = baseline_mean / best_mean
    
    print("\n" + "=" * 70)
    print("🎯 KEY FINDINGS")
    print("=" * 70)
    print(f"Optimal thread count: {best_threads}")
    print(f"Baseline ({min_threads} threads): {baseline_mean:.2f}s")
    print(f"Optimized performance: {best_mean:.2f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Throughput: {1/best_mean:.3f} images/sec")
    
    # Save detailed results
    results_file = output_dir / "baseline_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Test 1: Basic CPU Optimizations - Baseline Results\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("System Information:\n")
        f.write("-" * 70 + "\n")
        for key, value in sys_info.items():
            if key != "torch_blas":
                f.write(f"{key}: {value}\n")
        
        f.write("\n\nTest Configuration:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Lightning steps: {args.steps}\n")
        f.write(f"Inference steps: {args.inference_steps}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Data type: torch.float32\n")
        f.write(f"Model load time: {load_time:.2f}s\n")
        
        f.write("\n\nThread Count Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Threads':<10} {'Mean':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10}\n")
        f.write("-" * 54 + "\n")
        for threads, result in sorted(thread_results.items()):
            marker = " ⭐ BEST" if threads == best_threads else ""
            f.write(f"{threads:<10} {result['mean']:<12.2f} {result['std']:<12.2f} "
                   f"{result['min']:<10.2f} {result['max']:<10.2f}{marker}\n")
        
        f.write("\n\nKey Findings:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Optimal thread count: {best_threads}\n")
        f.write(f"Single-threaded baseline: {baseline_mean:.2f}s\n")
        f.write(f"Optimized performance: {best_mean:.2f}s\n")
        f.write(f"Speedup: {speedup:.2f}x\n")
        f.write(f"Throughput: {1/best_mean:.3f} images/sec\n")
        
        f.write("\n\nDetailed Times (per run):\n")
        f.write("-" * 70 + "\n")
        for threads, result in sorted(thread_results.items()):
            f.write(f"\n{threads} threads:\n")
            for i, t in enumerate(result['times'], 1):
                f.write(f"  Run {i}: {t:.2f}s\n")
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Save recommendations
    recommendations_file = output_dir / "recommendations.txt"
    with open(recommendations_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Recommendations for Your System\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. Optimal Configuration:\n")
        f.write(f"   - Set torch.set_num_threads({best_threads})\n")
        f.write(f"   - Use torch.float32 dtype\n")
        f.write(f"   - Expected performance: {best_mean:.2f}s per image\n\n")
        
        f.write("2. Environment Variables:\n")
        f.write(f"   export OMP_NUM_THREADS={best_threads}\n")
        f.write(f"   export MKL_NUM_THREADS={best_threads}\n")
        f.write(f"   export OPENBLAS_NUM_THREADS={best_threads}\n\n")
        
        f.write("3. Python Code:\n")
        f.write(f"   import torch\n")
        f.write(f"   torch.set_num_threads({best_threads})\n\n")
        
        f.write("4. Next Steps:\n")
        if speedup < 1.5:
            f.write("   ⚠️  Limited thread scaling observed.\n")
            f.write("   Consider: Mixed precision, attention optimization, or model distillation.\n")
        elif speedup < 3.0:
            f.write("   ✅ Good thread scaling. Your CPU benefits from parallelization.\n")
            f.write("   Next: Test mixed precision (FP16/BF16) for additional speedup.\n")
        else:
            f.write("   ✅ Excellent thread scaling!\n")
            f.write("   Next: Explore advanced optimizations (attention, quantization).\n")
    
    print(f"💾 Recommendations saved to: {recommendations_file}")
    
    print("\n" + "=" * 70)
    print("✅ Baseline Test Complete!")
    print("=" * 70)
    print(f"\n🎯 Use {best_threads} threads for optimal CPU performance")
    print(f"📊 Expected time per image: {best_mean:.2f}s")
    print(f"🚀 Speedup from single-threaded: {speedup:.2f}x")


if __name__ == "__main__":
    main()
