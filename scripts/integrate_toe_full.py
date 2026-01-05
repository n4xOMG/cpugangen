"""
Full TOE Integration Script - Generate images using TOE embeddings!

This script uses the custom StableDiffusionXLTOEPipeline to generate images
with TOE (Tag-Optimized Encoder) instead of CLIP text encoders.

Usage:
    # Hybrid mode (TOE context + CLIP pooled) - RECOMMENDED for testing
    python scripts/integrate_toe_full.py

    # Custom prompt
    python scripts/integrate_toe_full.py --prompt "1girl, solo, blue_eyes, smile"

    # Batch generation with comparison
    python scripts/integrate_toe_full.py --compare --num_images 5
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.sdxl_toe_pipeline import create_toe_pipeline


def generate_with_toe(
    prompt: str,
    pipeline,
    output_path: str,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    negative_prompt: str = None,
) -> Dict:
    """
    Generate a single image using TOE pipeline.
    
    Args:
        prompt: Tag-based prompt (comma-separated)
        pipeline: StableDiffusionXLTOEPipeline instance
        output_path: Where to save the image
        num_inference_steps: Denoising steps
        guidance_scale: CFG scale
        seed: Random seed
        negative_prompt: Negative prompt
        
    Returns:
        Dictionary with generation metadata
    """
    device = pipeline.device
    
    # Set seed for reproducibility
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"\n🎨 Generating with TOE...")
    print(f"   Prompt: {prompt}")
    print(f"   Steps: {num_inference_steps}, CFG: {guidance_scale}, Seed: {seed}")
    
    # Time the generation
    start_time = time.perf_counter()
    
    # Generate!
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt or "lowres, bad anatomy, bad hands, text, error",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    
    total_time = time.perf_counter() - start_time
    
    # Save image
    image = result.images[0]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    
    print(f"   ✓ Saved: {output_path}")
    print(f"   ✓ Time: {total_time:.2f}s")
    
    # Return metadata
    return {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
        'seed': seed,
        'total_time_s': round(total_time, 3),
        'output_path': output_path,
    }


def compare_toe_vs_clip(
    prompts: List[str],
    toe_pipeline,
    clip_pipeline,
    output_dir: str,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
):
    """
    Generate side-by-side comparisons of TOE vs CLIP.
    
    Args:
        prompts: List of test prompts
        toe_pipeline: TOE-based pipeline
        clip_pipeline: Standard CLIP-based pipeline
        output_dir: Output directory
        num_inference_steps: Steps
        guidance_scale: CFG scale
    """
    print("\n" + "=" * 70)
    print("TOE vs CLIP Comparison")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n📊 Test {i+1}/{len(prompts)}: {prompt[:50]}...")
        seed = 42 + i
        
        # Generate with TOE
        toe_output = os.path.join(output_dir, f"toe_{seed}.png")
        toe_meta = generate_with_toe(
            prompt, toe_pipeline, toe_output,
            num_inference_steps, guidance_scale, seed
        )
        
        # Generate with CLIP
        print(f"\n🎨 Generating with CLIP...")
        start_time = time.perf_counter()
        
        generator = torch.Generator(device=clip_pipeline.device).manual_seed(seed)
        clip_result = clip_pipeline(
            prompt=prompt,
            negative_prompt="lowres, bad anatomy, bad hands, text, error",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        clip_time = time.perf_counter() - start_time
        clip_output = os.path.join(output_dir, f"clip_{seed}.png")
        clip_result.images[0].save(clip_output)
        
        print(f"   ✓ Saved: {clip_output}")
        print(f"   ✓ Time: {clip_time:.2f}s")
        
        # Record comparison
        results.append({
            'prompt': prompt,
            'seed': seed,
            'toe': {
                'time_s': toe_meta['total_time_s'],
                'path': toe_output,
            },
            'clip': {
                'time_s': round(clip_time, 3),
                'path': clip_output,
            },
            'speedup': round(clip_time / toe_meta['total_time_s'], 3),
        })
        
        print(f"\n   Speedup: {results[-1]['speedup']:.2f}x")
    
    # Save comparison report
    report_path = os.path.join(output_dir, 'comparison_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'total_tests': len(results),
            'avg_toe_time_s': sum(r['toe']['time_s'] for r in results) / len(results),
            'avg_clip_time_s': sum(r['clip']['time_s'] for r in results) / len(results),
            'avg_speedup': sum(r['speedup'] for r in results) / len(results),
            'results': results,
        }, f, indent=2)
    
    print(f"\n✓ Comparison report saved: {report_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using TOE-SDXL integration"
    )
    
    # Model paths
    parser.add_argument(
        '--toe_checkpoint',
        type=str,
        default='checkpoints/toe/toe_best.pt',
        help='Path to TOE checkpoint'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default='data/vocabulary.json',
        help='Path to vocabulary file'
    )
    parser.add_argument(
        '--sdxl_model',
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='SDXL model name or path to .safetensors'
    )
    
    # Generation settings
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Single prompt to generate (comma-separated tags)'
    )
    parser.add_argument(
        '--negative_prompt',
        type=str,
        default=None,
        help='Negative prompt'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=25,
        help='Number of inference steps'
    )
    parser.add_argument(
        '--guidance',
        type=float,
        default=7.5,
        help='Guidance scale'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/toe_generation.png',
        help='Output image path'
    )
    
    # Comparison mode
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Run comparison with CLIP baseline'
    )
    parser.add_argument(
        '--num_images',
        type=int,
        default=3,
        help='Number of images for comparison'
    )
    
    # Mode
    parser.add_argument(
        '--hybrid',
        action='store_true',
        default=True,
        help='Use hybrid mode (TOE context + CLIP pooled)'
    )
    
    args = parser.parse_args()
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  WARNING: Running on CPU. This will be slow!")
        print("   For faster generation, use a GPU.")
    
    print("\n" + "=" * 70)
    print("TOE-SDXL Integration Test")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Mode: {'Hybrid (TOE context + CLIP pooled)' if args.hybrid else 'Full TOE'}")
    
    # Create TOE pipeline
    toe_pipeline = create_toe_pipeline(
        toe_checkpoint_path=args.toe_checkpoint,
        vocab_path=args.vocab,
        sdxl_model_path=args.sdxl_model,
        device=device,
        use_hybrid=args.hybrid,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # Comparison mode
    if args.compare:
        print("\n📊 Running TOE vs CLIP comparison...")
        
        # Load standard CLIP pipeline
        from diffusers import StableDiffusionXLPipeline
        
        print("\nLoading CLIP baseline pipeline...")
        if args.sdxl_model.endswith('.safetensors'):
            clip_pipeline = StableDiffusionXLPipeline.from_single_file(
                args.sdxl_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
        else:
            clip_pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.sdxl_model,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                variant="fp16" if device == "cuda" else None,
            )
        
        clip_pipeline = clip_pipeline.to(device)
        print("✓ CLIP pipeline loaded")
        
        # Test prompts
        test_prompts = [
            "1girl, solo, long_hair, blue_eyes, smile, looking_at_viewer, outdoors, cherry_blossoms",
            "2girls, yuri, kissing, romantic, sunset, detailed_background",
            "scenery, landscape, mountains, river, anime_style, high_quality",
            "1boy, male_focus, detailed_face, urban_background, modern_clothes",
            "still_life, food, plate, table, aesthetic, warm_lighting",
        ][:args.num_images]
        
        # Run comparison
        compare_toe_vs_clip(
            prompts=test_prompts,
            toe_pipeline=toe_pipeline,
            clip_pipeline=clip_pipeline,
            output_dir="outputs/comparison",
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
        )
        
        print("\n" + "=" * 70)
        print("✓ Comparison Complete!")
        print("=" * 70)
        print("\nCheck outputs/comparison/ for:")
        print("  - toe_*.png (TOE-generated images)")
        print("  - clip_*.png (CLIP-generated images)")
        print("  - comparison_report.json (detailed metrics)")
    
    # Single generation mode
    else:
        # Use default prompt if none provided
        if args.prompt is None:
            args.prompt = "1girl, solo, long_hair, blue_eyes, smile, looking_at_viewer, outdoors, cherry_blossoms"
            print(f"\n💡 Using default prompt: {args.prompt}")
        
        # Generate
        metadata = generate_with_toe(
            prompt=args.prompt,
            pipeline=toe_pipeline,
            output_path=args.output,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
        )
        
        # Save metadata
        meta_path = args.output.replace('.png', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Metadata saved: {meta_path}")
        
        print("\n" + "=" * 70)
        print("✓ Generation Complete!")
        print("=" * 70)
        print(f"\n🎉 Your TOE-generated image is ready: {args.output}")
        print(f"\nThis image was created using TOE embeddings,")
        print(f"not CLIP - you're seeing the result of your trained model!")


if __name__ == "__main__":
    main()
