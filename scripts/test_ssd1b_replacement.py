#!/usr/bin/env python3
"""
Test 2: Selective Knowledge Distillation - SSD-1B UNet Replacement

This script tests the hypothesis that a smaller, specialized student UNet 
can maintain quality while being inherently more CPU-friendly.

Approach:
1. Load Illustrious pipeline (text encoders, VAE, scheduler)
2. Replace UNet with SSD-1B's distilled UNet (50% smaller)
3. Optionally apply SDXL-Lightning LoRA
4. Benchmark performance and quality

SSD-1B: Distilled SDXL with progressive layer removal
- 50% parameter reduction (1.3B vs 2.6B)
- 40 transformer blocks + 1 ResNet block removed
- Trained via knowledge distillation with layer-level losses
"""

import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    UNet2DConditionModel,
    EulerDiscreteScheduler
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image
import psutil


def get_memory_usage() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def load_baseline_pipeline(
    base_model: str = "martineux/janku6",
    lightning_steps: int = 4,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    apply_lightning: bool = True
) -> StableDiffusionXLPipeline:
    """
    Load baseline Illustrious + SDXL-Lightning pipeline.
    
    Args:
        base_model: HuggingFace model ID for Illustrious
        lightning_steps: Number of Lightning steps (2, 4, or 8)
        device: Device to use
        dtype: Data type for model
        apply_lightning: Whether to apply Lightning LoRA
        
    Returns:
        Loaded pipeline
    """
    print(f"\n{'='*60}")
    print("📦 Loading BASELINE Pipeline")
    print(f"{'='*60}")
    print(f"Base model: {base_model}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    load_start = time.time()
    
    # Load base pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )
    
    # Apply Lightning LoRA if requested
    if apply_lightning:
        print(f"📦 Loading Lightning {lightning_steps}-step LoRA...")
        ckpt_map = {
            2: "sdxl_lightning_2step_lora.safetensors",
            4: "sdxl_lightning_4step_lora.safetensors",
            8: "sdxl_lightning_8step_lora.safetensors"
        }
        ckpt = hf_hub_download("ByteDance/SDXL-Lightning", ckpt_map[lightning_steps])
        pipe.load_lora_weights(load_file(ckpt))
        
        # Configure scheduler for Lightning
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing",
            prediction_type="epsilon",
        )
    
    # Move to device
    pipe.to(device)
    
    if apply_lightning:
        pipe.fuse_lora()
        pipe.unload_lora_weights()
    
    load_time = time.time() - load_start
    
    # Get model stats
    unet_params = count_parameters(pipe.unet)
    memory_usage = get_memory_usage()
    
    print(f"\n✅ Baseline pipeline loaded in {load_time:.2f}s")
    print(f"   UNet parameters: {unet_params:,}")
    print(f"   Memory usage: {memory_usage:.2f} MB")
    
    return pipe


def load_hybrid_pipeline(
    base_model: str = "martineux/janku6",
    student_unet_model: str = "segmind/SSD-1B",
    lightning_steps: int = 4,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    apply_lightning: bool = True
) -> StableDiffusionXLPipeline:
    """
    Load hybrid pipeline: Illustrious (encoders + VAE) + SSD-1B (UNet).
    
    Args:
        base_model: HuggingFace model ID for Illustrious
        student_unet_model: HuggingFace model ID for SSD-1B
        lightning_steps: Number of Lightning steps
        device: Device to use
        dtype: Data type for model
        apply_lightning: Whether to apply Lightning LoRA
        
    Returns:
        Hybrid pipeline
    """
    print(f"\n{'='*60}")
    print("📦 Loading HYBRID Pipeline (SSD-1B UNet)")
    print(f"{'='*60}")
    print(f"Base model: {base_model}")
    print(f"Student UNet: {student_unet_model}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    
    load_start = time.time()
    
    # Load base pipeline (this gives us text encoders, VAE, scheduler)
    print("Loading base pipeline components...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )
    
    # Get baseline UNet stats before replacement
    baseline_unet_params = count_parameters(pipe.unet)
    print(f"   Baseline UNet parameters: {baseline_unet_params:,}")
    
    # Load SSD-1B UNet separately
    print(f"Loading SSD-1B UNet from {student_unet_model}...")
    try:
        ssd1b_unet = UNet2DConditionModel.from_pretrained(
            student_unet_model,
            subfolder="unet",
            torch_dtype=dtype,
        )
    except Exception as e:
        print(f"⚠️  Error loading from subfolder 'unet': {e}")
        print("Trying to load entire model and extract UNet...")
        # Fallback: load entire SSD-1B pipeline and extract UNet
        ssd1b_pipe = StableDiffusionXLPipeline.from_pretrained(
            student_unet_model,
            torch_dtype=dtype,
        )
        ssd1b_unet = ssd1b_pipe.unet
        del ssd1b_pipe  # Free memory
    
    # Get SSD-1B UNet stats
    ssd1b_unet_params = count_parameters(ssd1b_unet)
    reduction_pct = (1 - ssd1b_unet_params / baseline_unet_params) * 100
    
    print(f"   SSD-1B UNet parameters: {ssd1b_unet_params:,}")
    print(f"   Parameter reduction: {reduction_pct:.1f}%")
    
    # Replace UNet
    print("Replacing UNet with SSD-1B...")
    pipe.unet = ssd1b_unet
    
    # Apply Lightning LoRA if requested
    if apply_lightning:
        print(f"📦 Applying Lightning {lightning_steps}-step LoRA to SSD-1B UNet...")
        try:
            ckpt_map = {
                2: "sdxl_lightning_2step_lora.safetensors",
                4: "sdxl_lightning_4step_lora.safetensors",
                8: "sdxl_lightning_8step_lora.safetensors"
            }
            ckpt = hf_hub_download("ByteDance/SDXL-Lightning", ckpt_map[lightning_steps])
            pipe.load_lora_weights(load_file(ckpt))
            
            # Configure scheduler for Lightning
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="trailing",
                prediction_type="epsilon",
            )
            
            pipe.fuse_lora()
            pipe.unload_lora_weights()
            print("   ✅ Lightning LoRA applied successfully")
        except Exception as e:
            print(f"   ⚠️  Warning: Lightning LoRA application failed: {e}")
            print("   Continuing without Lightning acceleration...")
    
    # Move to device
    pipe.to(device)
    
    load_time = time.time() - load_start
    memory_usage = get_memory_usage()
    
    print(f"\n✅ Hybrid pipeline loaded in {load_time:.2f}s")
    print(f"   Memory usage: {memory_usage:.2f} MB")
    
    return pipe


def generate_image(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    num_inference_steps: int,
    seed: int,
    device: str = "cpu"
) -> tuple[Image.Image, float]:
    """
    Generate a single image and return it with generation time.
    
    Returns:
        (image, generation_time_seconds)
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    
    start = time.time()
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0,  # Lightning uses CFG=0
        generator=generator,
    )
    elapsed = time.time() - start
    
    return result.images[0], elapsed


def benchmark_pipeline(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    num_inference_steps: int,
    num_iterations: int,
    seed: int,
    device: str = "cpu",
    warmup: bool = True
) -> Dict[str, Any]:
    """
    Benchmark a pipeline with multiple iterations.
    
    Returns:
        Dictionary with benchmark statistics
    """
    times = []
    
    # Warmup run
    if warmup:
        print("🔥 Warmup run...")
        _, _ = generate_image(pipe, prompt, num_inference_steps, seed, device)
    
    # Benchmark runs
    print(f"🏃 Running {num_iterations} benchmark iterations...")
    for i in range(num_iterations):
        _, elapsed = generate_image(pipe, prompt, num_inference_steps, seed + i, device)
        times.append(elapsed)
        print(f"   Iteration {i+1}/{num_iterations}: {elapsed:.2f}s")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    return {
        "avg_time": avg_time,
        "min_time": min_time,
        "max_time": max_time,
        "times": times,
        "throughput": 1 / avg_time
    }



def main():
    parser = argparse.ArgumentParser(
        description="Test SSD-1B UNet as replacement for Illustrious UNet"
    )
    parser.add_argument(
        "--base-model", 
        default="martineux/janku6",
        help="Base Illustrious model"
    )
    parser.add_argument(
        "--student-unet",
        default="segmind/SSD-1B",
        help="Student UNet model (SSD-1B)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        choices=[2, 4, 8],
        help="Lightning steps"
    )
    parser.add_argument(
        "--no-lightning",
        action="store_true",
        help="Disable Lightning LoRA"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=5,
        help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--compare-quality",
        action="store_true",
        help="Generate quality comparison images"
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "1girl, blue_hair, beautiful_detailed_eyes, anime_style",
            "anime boy, fantasy outfit, detailed_background",
            "landscape, cherry_blossoms, anime_style, highly_detailed"
        ],
        help="Prompts for quality comparison"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        default="outputs/ssd1b_test",
        help="Output directory"
    )
    parser.add_argument(
        "--test-load-only",
        action="store_true",
        help="Only test loading, don't generate"
    )
    parser.add_argument(
        "--test-baseline-only",
        action="store_true",
        help="Only test baseline pipeline"
    )
    parser.add_argument(
        "--test-hybrid-only",
        action="store_true",
        help="Only test hybrid pipeline"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    device = "cpu"
    dtype = torch.float32
    apply_lightning = not args.no_lightning
    
    print(f"\n{'='*60}")
    print("🧪 Test 2: Selective Knowledge Distillation")
    print(f"{'='*60}")
    print(f"Base model: {args.base_model}")
    print(f"Student UNet: {args.student_unet}")
    print(f"Lightning: {'Enabled' if apply_lightning else 'Disabled'}")
    print(f"Steps: {args.steps}")
    print(f"Device: {device}")
    print(f"\n⚠️  Memory-conscious mode: Loading pipelines sequentially")
    
    # Determine which pipelines to test
    test_baseline = not args.test_hybrid_only
    test_hybrid = not args.test_baseline_only
    
    baseline_stats = None
    hybrid_stats = None
    baseline_images = {}
    hybrid_images = {}
    
    # ===== BASELINE PIPELINE =====
    if test_baseline:
        print(f"\n{'='*60}")
        print("📦 PHASE 1: Baseline Pipeline Testing")
        print(f"{'='*60}")
        
        baseline_pipe = load_baseline_pipeline(
            args.base_model,
            args.steps,
            device,
            dtype,
            apply_lightning
        )
        
        if args.test_load_only:
            print("✅ Baseline load test complete!")
            del baseline_pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import gc
            gc.collect()
            
            if not test_hybrid:
                return
        else:
            # Benchmark baseline
            if args.benchmark:
                print(f"\n⏱️  Benchmarking baseline...")
                test_prompt = args.prompts[0]
                baseline_stats = benchmark_pipeline(
                    baseline_pipe,
                    test_prompt,
                    args.steps,
                    args.num_iterations,
                    args.seed,
                    device
                )
            
            # Quality comparison - baseline
            if args.compare_quality:
                print(f"\n🎨 Generating baseline quality images...")
                for i, prompt in enumerate(args.prompts):
                    print(f"  Prompt {i+1}/{len(args.prompts)}: {prompt[:50]}...")
                    img, elapsed = generate_image(
                        baseline_pipe, prompt, args.steps, args.seed + i, device
                    )
                    baseline_images[i] = (img, elapsed, prompt)
            
            # Clean up baseline pipeline before loading hybrid
            print(f"\n🧹 Cleaning up baseline pipeline to free memory...")
            del baseline_pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import gc
            gc.collect()
            print(f"   Memory freed. Current usage: {get_memory_usage():.2f} MB")
    
    # ===== HYBRID PIPELINE =====
    if test_hybrid:
        print(f"\n{'='*60}")
        print("📦 PHASE 2: Hybrid Pipeline Testing")
        print(f"{'='*60}")
        
        hybrid_pipe = load_hybrid_pipeline(
            args.base_model,
            args.student_unet,
            args.steps,
            device,
            dtype,
            apply_lightning
        )
        
        if args.test_load_only:
            print("✅ Hybrid load test complete!")
            del hybrid_pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import gc
            gc.collect()
            return
        else:
            # Benchmark hybrid
            if args.benchmark:
                print(f"\n⏱️  Benchmarking hybrid...")
                test_prompt = args.prompts[0]
                hybrid_stats = benchmark_pipeline(
                    hybrid_pipe,
                    test_prompt,
                    args.steps,
                    args.num_iterations,
                    args.seed,
                    device
                )
            
            # Quality comparison - hybrid
            if args.compare_quality:
                print(f"\n🎨 Generating hybrid quality images...")
                for i, prompt in enumerate(args.prompts):
                    print(f"  Prompt {i+1}/{len(args.prompts)}: {prompt[:50]}...")
                    img, elapsed = generate_image(
                        hybrid_pipe, prompt, args.steps, args.seed + i, device
                    )
                    hybrid_images[i] = (img, elapsed, prompt)
            
            # Clean up hybrid pipeline
            print(f"\n🧹 Cleaning up hybrid pipeline...")
            del hybrid_pipe
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            import gc
            gc.collect()
    
    # ===== RESULTS =====
    if args.benchmark and baseline_stats and hybrid_stats:
        # Calculate speedup
        speedup = baseline_stats["avg_time"] / hybrid_stats["avg_time"]
        
        # Print results
        print(f"\n{'='*60}")
        print("📊 BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"\nBaseline (Illustrious UNet):")
        print(f"  Average: {baseline_stats['avg_time']:.2f}s")
        print(f"  Min:     {baseline_stats['min_time']:.2f}s")
        print(f"  Max:     {baseline_stats['max_time']:.2f}s")
        
        print(f"\nHybrid (SSD-1B UNet):")
        print(f"  Average: {hybrid_stats['avg_time']:.2f}s")
        print(f"  Min:     {hybrid_stats['min_time']:.2f}s")
        print(f"  Max:     {hybrid_stats['max_time']:.2f}s")
        
        print(f"\n🚀 Speedup: {speedup:.2f}x")
        
        # Save results
        results = {
            "base_model": args.base_model,
            "student_unet": args.student_unet,
            "lightning_steps": args.steps,
            "lightning_enabled": apply_lightning,
            "device": device,
            "baseline": baseline_stats,
            "hybrid": hybrid_stats,
            "speedup": speedup
        }
        
        results_path = output_dir / "benchmark_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_path}")
    
    # Save quality comparison images
    if args.compare_quality and baseline_images and hybrid_images:
        print(f"\n{'='*60}")
        print("🎨 Saving Quality Comparison Images")
        print(f"{'='*60}")
        
        comparison_dir = output_dir / "quality_comparison"
        comparison_dir.mkdir(exist_ok=True, parents=True)
        
        for i in baseline_images.keys():
            baseline_img, baseline_time, prompt = baseline_images[i]
            hybrid_img, hybrid_time, _ = hybrid_images[i]
            
            # Save individual images
            baseline_path = comparison_dir / f"prompt_{i+1}_baseline.png"
            hybrid_path = comparison_dir / f"prompt_{i+1}_ssd1b.png"
            
            baseline_img.save(baseline_path)
            hybrid_img.save(hybrid_path)
            
            # Create side-by-side comparison
            side_by_side = Image.new('RGB', (baseline_img.width * 2, baseline_img.height))
            side_by_side.paste(baseline_img, (0, 0))
            side_by_side.paste(hybrid_img, (baseline_img.width, 0))
            
            comparison_path = comparison_dir / f"prompt_{i+1}_comparison.png"
            side_by_side.save(comparison_path)
            
            print(f"\nPrompt {i+1}: {prompt[:50]}...")
            print(f"  Baseline: {baseline_time:.2f}s → {baseline_path.name}")
            print(f"  SSD-1B:   {hybrid_time:.2f}s → {hybrid_path.name}")
            print(f"  Speedup:  {baseline_time/hybrid_time:.2f}x")
            print(f"  Comparison: {comparison_path.name}")
    
    print(f"\n{'='*60}")
    print("✅ Test Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
