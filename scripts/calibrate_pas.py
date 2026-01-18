#!/usr/bin/env python3
"""
PAS Calibration Script

Runs shift score analysis on prompts to find optimal phase division point D*.

Usage:
    python scripts/calibrate_pas.py \
        --model martineux/janku6 \
        --prompts data/prompts.txt \
        --num-steps 20 \
        --max-prompts 100 \
        --output configs/pas_calibration.json
"""

import argparse
import os
import sys
import json
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import StableDiffusionXLPipeline
from algorithms.pas_calibration import ShiftScoreAnalyzer


def load_prompts(filepath: str, max_prompts: int = 100) -> list:
    """Load prompts from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    # Random sample if too many
    if len(prompts) > max_prompts:
        prompts = random.sample(prompts, max_prompts)
    
    return prompts


def main():
    parser = argparse.ArgumentParser(description="PAS Calibration")
    parser.add_argument("--model", type=str, default="martineux/janku6")
    parser.add_argument("--prompts", type=str, required=True, help="Path to prompts file (one per line)")
    parser.add_argument("--num-steps", type=int, default=20, help="Denoising steps for calibration")
    parser.add_argument("--max-prompts", type=int, default=100, help="Maximum prompts to use")
    parser.add_argument("--output", type=str, default="configs/pas_calibration.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--visualize", action="store_true", help="Create shift score plot")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("PAS Calibration - Finding Optimal Phase Division")
    print(f"{'='*60}\n")
    print(f"Model: {args.model}")
    print(f"Prompts file: {args.prompts}")
    print(f"Max prompts: {args.max_prompts}")
    print(f"Steps: {args.num_steps}")
    print(f"Device: {args.device}\n")
    
    # Load prompts
    print("Loading prompts...")
    prompts = load_prompts(args.prompts, args.max_prompts)
    print(f"✅ Loaded {len(prompts)} prompts\n")
    
    # Load model
    print("Loading model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipeline = pipeline.to(args.device)
    print("✅ Model loaded\n")
    
    # Run calibration
    analyzer = ShiftScoreAnalyzer(device=args.device, verbose=True)
    
    results = analyzer.calibrate(
        pipeline=pipeline,
        prompts=prompts,
        num_steps=args.num_steps,
        max_prompts=args.max_prompts
    )
    
    # Suggest PAS parameters based on D*
    D_star = results['D_star']
    
    # Default parameter suggestions (based on SD-Acc paper)
    suggested_params = {
        'T_sketch': max(D_star, args.num_steps * 3 // 4),  # Extend slightly beyond D*
        'T_complete': min(3, args.num_steps // 10),  # First 10% full
        'T_sparse': 5,  # Run full every 5 steps in sketching
        'L_sketch': 8,  # Keep 8 blocks in sparse steps
        'L_refine': 4,  # Only 4 top blocks in refinement
    }
    
    results['suggested_params'] = suggested_params
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Calibration results saved to: {args.output}\n")
    
    # Print recommendations
    print(f"{'='*60}")
    print("Recommended PAS Parameters")
    print(f"{'='*60}\n")
    for param, value in suggested_params.items():
        print(f"  {param}: {value}")
    
    print(f"\nPhase division:")
    print(f"  Sketching: steps 0-{suggested_params['T_sketch']-1}")
    print(f"  Refinement: steps {suggested_params['T_sketch']}-{args.num_steps-1}\n")
    
    # Visualize
    if args.visualize:
        print("Creating visualization...")
        viz_path = args.output.replace('.json', '_visualization.png')
        analyzer.visualize_shift_scores(
            results['shift_scores'],
            D_star,
            viz_path
        )
        print()
    
    print("✅ Calibration complete!")
    print(f"\n💡 Next step: Use these parameters with PAS pipeline\n")


if __name__ == "__main__":
    main()
