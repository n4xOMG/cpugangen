#!/usr/bin/env python3
"""
Generate with TOE + Segmind - using proven TOE integration.

This uses the monkey-patching approach from integrate_toe_full.py
but with Segmind SSD-1B UNet instead of full SDXL UNet.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.sdxl_toe_pipeline import create_toe_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--toe_checkpoint", type=str, default="checkpoints/toe/toe_with_pooling_best.pt")
    parser.add_argument("--vocab", type=str, default="data/vocabulary.json")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TOE + Segmind SSD-1B Generation")
    print("=" * 70)
    
    # Create pipeline with TOE (uses proven monkey-patching approach)
    print("\n1. Creating TOE pipeline...")
    pipeline = create_toe_pipeline(
        toe_checkpoint_path=args.toe_checkpoint,
        vocab_path=args.vocab,  # Fixed: was vocabulary_path
        sdxl_model_path="segmind/SSD-1B",  # Fixed: was base_model_id
        device=args.device,
    )
    
    print("\n2. Generating image...")
    
    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device == "cuda":
            torch.cuda.manual_seed(args.seed)
    
    # Generate
    image = pipeline(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg_scale,
    ).images[0]
    
    # Save
    image.save(args.output)
    print(f"\n✓ Saved to: {args.output}")


if __name__ == "__main__":
    main()
