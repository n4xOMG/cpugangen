#!/usr/bin/env python3
"""
Profile Inference Pipeline

Deep profiling tool to identify cacheable operations and optimization opportunities.

Usage:
    python scripts/profile_inference.py \\
        --model segmind/SSD-1B \\
        --steps 25 \\
        --output outputs/profiling/
"""

import argparse
import cProfile
import pstats
import io
import sys
import time
from pathlib import Path
from typing import Dict, Any

import torch
from diffusers import StableDiffusionXLPipeline

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def profile_inference(
    pipeline,
    prompt: str,
    num_steps: int,
    seed: int = 42,
    apply_lightning: bool = True
) -> tuple[Any, Dict[str, Any]]:
    """
    Profile a single inference run with cProfile.
    
    Returns:
        (image, profile_stats)
    """
    profiler = cProfile.Profile()
    
    # Profile inference
    profiler.enable()
    
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipeline(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=0.0 if apply_lightning else 7.5,
        generator=generator,
    )
    
    profiler.disable()
    
    # Extract statistics
    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream)
    
    # Sort by cumulative time
    stats.sort_stats('cumulative')
    
    return result.images[0], stats


def analyze_profile(stats: pstats.Stats, top_n: int = 30) -> Dict[str, Any]:
    """
    Analyze profile statistics to identify optimization opportunities.
    
    Returns:
        Dictionary with analysis results
    """
    stats_stream = io.StringIO()
    stats.stream = stats_stream
    
    # Get top functions by cumulative time
    stats.print_stats(top_n)
    output = stats_stream.getvalue()
    
    # Parse statistics
    stats_dict = stats.stats
    
    # Find frequently called functions
    frequent_funcs = []
    for func, (cc, nc, tt, ct, callers) in stats_dict.items():
        if nc > 100:  # Called more than 100 times
            func_name = f"{func[0]}:{func[1]}({func[2]})"
            frequent_funcs.append({
                "function": func_name,
                "calls": nc,
                "total_time": tt,
                "cumulative_time": ct,
                "time_per_call": ct / nc if nc > 0 else 0
            })
    
    # Sort by total impact (calls * time_per_call)
    frequent_funcs.sort(key=lambda x: x["calls"] * x["time_per_call"], reverse=True)
    
    return {
        "full_output": output,
        "frequent_functions": frequent_funcs[:20],
        "total_functions": len(stats_dict)
    }


def identify_cacheable_operations(analysis: Dict[str, Any]) -> list:
    """
    Identify operations that are good candidates for caching.
    
    Looks for:
    - Functions called many times (>50)
    - With consistent inputs (deterministic)
    - Taking non-trivial time
    """
    candidates = []
    
    for func_info in analysis["frequent_functions"]:
        func_name = func_info["function"]
        calls = func_info["calls"]
        time_per_call = func_info["time_per_call"]
        
        # Heuristics for cacheable operations
        is_cacheable = False
        reason = ""
        
        # Check for timestep embedding
        if "timestep" in func_name.lower() and "emb" in func_name.lower():
            is_cacheable = True
            reason = "Timestep embedding computation"
        
        # Check for sinusoidal operations
        elif "sin" in func_name.lower() or "cos" in func_name.lower():
            if calls > 50:
                is_cacheable = True
                reason = "Frequent trigonometric operation"
        
        # Check for linear projections
        elif "linear" in func_name.lower() or "matmul" in func_name.lower():
            if calls > 100 and time_per_call > 0.0001:
                is_cacheable = True
                reason = "Frequent linear transformation"
        
        # Check for layer norm
        elif "norm" in func_name.lower():
            if calls > 100:
                is_cacheable = True
                reason = "Frequent normalization"
        
        if is_cacheable:
            candidates.append({
                **func_info,
                "cache_reason": reason,
                "estimated_savings_sec": calls * time_per_call * 0.9  # 90% hit rate assumption
            })
    
    # Sort by estimated savings
    candidates.sort(key=lambda x: x["estimated_savings_sec"], reverse=True)
    
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Profile diffusion inference pipeline")
    
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
        "--prompt",
        type=str,
        default="1girl, solo, blue_hair, detailed, anime",
        help="Test prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/profiling",
        help="Output directory for profile data",
    )
    parser.add_argument(
        "--enable-caching",
        action="store_true",
        help="Enable deterministic caching",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=128,
        help="Cache size",
    )
    
    args = parser.parse_args()
    
    # Set default steps to match lightning if not specified
    if args.steps is None:
        args.steps = args.lightning_steps
    
    apply_lightning = not args.no_lightning
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("Inference Pipeline Profiling")
    print("="*70)
    print(f"Base model: {args.base_model}")
    if args.use_hybrid:
        print(f"Student UNet: {args.student_unet}")
    print(f"Lightning: {'Enabled' if apply_lightning else 'Disabled'} ({args.lightning_steps}-step)")
    print(f"Steps: {args.steps}")
    print(f"Prompt: {args.prompt}")
    print(f"Caching: {'Enabled' if args.enable_caching else 'Disabled'}")
    print()
    
    # Initialize cache if requested
    if args.enable_caching:
        from hqpd.optimization import get_global_cache
        cache = get_global_cache(max_size=args.cache_size, enabled=True)
        print(f"✓ Caching enabled (cache size: {args.cache_size})")
    
    # Load pipeline
    print("\nLoading pipeline...")
    
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
    print("✓ Pipeline loaded")
    
    # Profile inference
    print("\nProfiling inference (this may take a few minutes)...")
    image, stats = profile_inference(
        pipeline,
        args.prompt,
        args.steps,
        args.seed,
        apply_lightning
    )
    
    # Analyze profile
    print("\nAnalyzing profile...")
    analysis = analyze_profile(stats, top_n=50)
    
    # Identify cacheable operations
    print("\nIdentifying cacheable operations...")
    cacheable = identify_cacheable_operations(analysis)
    
    # Print results
    print("\n" + "="*70)
    print("PROFILING RESULTS")
    print("="*70)
    
    print("\n1. TOP 30 FUNCTIONS BY CUMULATIVE TIME:")
    print("-"*70)
    print(analysis["full_output"])
    
    print("\n2. FREQUENTLY CALLED FUNCTIONS:")
    print("-"*70)
    print(f"{'Function':<50} {'Calls':>8} {'Time/Call':>12}")
    print("-"*70)
    for func in analysis["frequent_functions"][:15]:
        func_short = func["function"][-50:]
        print(f"{func_short:<50} {func['calls']:>8} {func['time_per_call']:>12.6f}s")
    
    print("\n3. CACHEABLE OPERATION CANDIDATES:")
    print("-"*70)
    print(f"{'Function':<40} {'Calls':>8} {'Est. Savings':>12}")
    print("-"*70)
    
    if cacheable:
        for cand in cacheable[:10]:
            func_short = cand["function"].split(":")[-1][:40]
            print(f"{func_short:<40} {cand['calls']:>8} {cand['estimated_savings_sec']:>11.3f}s")
            print(f"  → {cand['cache_reason']}")
    else:
        print("No obvious cacheable operations identified.")
        print("Consider lowering thresholds or analyzing specific functions.")
    
    # Save detailed report
    report_path = output_dir / "profile_report.txt"
    with open(report_path, "w") as f:
        f.write("="*70 + "\n")
        f.write("INFERENCE PROFILING REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Base model: {args.base_model}\n")
        if args.use_hybrid:
            f.write(f"Student UNet: {args.student_unet}\n")
        f.write(f"Lightning: {'Enabled' if apply_lightning else 'Disabled'} ({args.lightning_steps}-step)\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Prompt: {args.prompt}\n\n")
        
        f.write("TOP FUNCTIONS:\n")
        f.write("-"*70 + "\n")
        f.write(analysis["full_output"])
        
        f.write("\n\nFREQUENTLY CALLED FUNCTIONS:\n")
        f.write("-"*70 + "\n")
        for func in analysis["frequent_functions"]:
            f.write(f"{func['function']}\n")
            f.write(f"  Calls: {func['calls']}, Time/Call: {func['time_per_call']:.6f}s\n")
        
        f.write("\n\nCACHEABLE OPERATIONS:\n")
        f.write("-"*70 + "\n")
        for cand in cacheable:
            f.write(f"{cand['function']}\n")
            f.write(f"  Reason: {cand['cache_reason']}\n")
            f.write(f"  Calls: {cand['calls']}, Est. Savings: {cand['estimated_savings_sec']:.3f}s\n\n")
    
    print(f"\n💾 Detailed report saved to: {report_path}")
    
    # Save binary profile for external analysis
    profile_path = output_dir / "inference.prof"
    stats.dump_stats(str(profile_path))
    print(f"💾 Binary profile saved to: {profile_path}")
    print(f"   View with: python -m pstats {profile_path}")
    
    # Save image
    image_path = output_dir / "profile_test_image.png"
    image.save(image_path)
    print(f"💾 Test image saved to: {image_path}")
    
    print("\n" + "="*70)
    print("✓ Profiling Complete")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
