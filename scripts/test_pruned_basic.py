#!/usr/bin/env python3
"""
Simple Pruned Model Test

Just generates one image with the pruned model to verify it works.
No schedule optimization needed - just basic functionality test.

Usage:
    python scripts/test_pruned_basic.py \
        --pruned-model checkpoints/pruned_illustrious_safe \
        --prompt "1girl, blue_hair, anime_style" \
        --output outputs/test_basic.png
"""

import argparse
import sys
import os
import time
import torch
from diffusers import StableDiffusionXLPipeline

def main():
    parser = argparse.ArgumentParser(description="Basic Pruned Model Test")
    parser.add_argument("--pruned-model", type=str, required=True, help="Path to pruned model")
    parser.add_argument("--prompt", type=str, default="1girl, blue_hair, anime_style, detailed, masterpiece")
    parser.add_argument("--num-steps", type=int, default=20, help="Inference steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="outputs/test_basic.png")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Basic Pruned Model Test")
    print(f"{'='*60}\n")
    print(f"Model: {args.pruned_model}")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.num_steps}")
    print(f"Device: {args.device}\n")
    
    # Load model
    print("Loading pruned model...")
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pruned_model,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        pipeline = pipeline.to(args.device)
        print("✅ Model loaded successfully\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1
    
    # Generate image
    print("Generating image...")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    
    start_time = time.time()
    try:
        output = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.num_steps,
            generator=generator,
            guidance_scale=7.5,
        )
        elapsed = time.time() - start_time
        
        # Save
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
        output.images[0].save(args.output)
        
        print(f"\n✅ Generation successful!")
        print(f"   Time: {elapsed:.2f}s ({elapsed/args.num_steps:.2f}s/step)")
        print(f"   Saved to: {args.output}\n")
        
        print("💡 Next steps:")
        print("   1. Open the image and check quality")
        print("   2. If image looks good → proceed with AYS schedule optimization")
        print("   3. If image is garbage → prune with even lower sparsity\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
