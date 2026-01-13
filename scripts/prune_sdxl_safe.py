#!/usr/bin/env python3
"""
Safe SDXL Pruning Script

Uses conservative pruning settings to avoid destroying the model.

Usage:
    python scripts/prune_sdxl_safe.py \
        --model martineux/janku6 \
        --text-encoder-sparsity 0.25 \
        --unet-sparsity 0.15 \
        --output checkpoints/pruned_illustrious_safe
"""

import argparse
import os
import sys
import json
import torch
from diffusers import StableDiffusionXLPipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.safe_pruning import prune_sdxl_safe


def main():
    parser = argparse.ArgumentParser(description="Safe SDXL Pruning")
    parser.add_argument("--model", type=str, default="martineux/janku6")
    parser.add_argument("--text-encoder-sparsity", type=float, default=0.25, help="25% (safe)")
    parser.add_argument("--unet-sparsity", type=float, default=0.15, help="15% (safe)")
    parser.add_argument("--output", type=str, default="checkpoints/pruned_illustrious_safe")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("SAFE SDXL Pruning")
    print(f"{'='*60}\n")
    print(f"⚠️  Using CONSERVATIVE settings:")
    print(f"   Text Encoder: {args.text_encoder_sparsity*100:.1f}% (vs 47.5% in paper)")
    print(f"   UNet: {args.unet_sparsity*100:.1f}% (vs 35% in paper)")
    print(f"\n   This is safer and less likely to break the model!\n")
    
    # Load model
    print("Loading model...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipeline = pipeline.to(args.device)
    print("✅ Loaded\n")
    
    # Prune safely
    stats = prune_sdxl_safe(
        pipeline=pipeline,
        text_encoder_sparsity=args.text_encoder_sparsity,
        unet_sparsity=args.unet_sparsity,
        verbose=True
    )
    
    # Save
    print(f"\nSaving to {args.output}...")
    os.makedirs(args.output, exist_ok=True)
    
    # Save components
    if pipeline.text_encoder is not None:
        pipeline.text_encoder.save_pretrained(os.path.join(args.output, "text_encoder"))
    if pipeline.text_encoder_2 is not None:
        pipeline.text_encoder_2.save_pretrained(os.path.join(args.output, "text_encoder_2"))
    if pipeline.unet is not None:
        pipeline.unet.save_pretrained(os.path.join(args.output, "unet"))
    if pipeline.vae is not None:
        pipeline.vae.save_pretrained(os.path.join(args.output, "vae"))
    if pipeline.scheduler is not None:
        pipeline.scheduler.save_pretrained(os.path.join(args.output, "scheduler"))
    if hasattr(pipeline, 'tokenizer') and pipeline.tokenizer is not None:
        pipeline.tokenizer.save_pretrained(os.path.join(args.output, "tokenizer"))
    if hasattr(pipeline, 'tokenizer_2') and pipeline.tokenizer_2 is not None:
        pipeline.tokenizer_2.save_pretrained(os.path.join(args.output, "tokenizer_2"))
    
    # Save model_index.json
    model_index = {
        "_class_name": "StableDiffusionXLPipeline",
        "_diffusers_version": "0.21.0",
        "text_encoder": ["transformers", "CLIPTextModel"],
        "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "tokenizer_2": ["transformers", "CLIPTokenizer"],
        "unet": ["diffusers", "UNet2DConditionModel"],
        "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        "vae": ["diffusers", "AutoencoderKL"]
    }
    with open(os.path.join(args.output, "model_index.json"), 'w') as f:
        json.dump(model_index, f, indent=2)
    
    # Save metadata
    metadata = {
        'base_model': args.model,
        'text_encoder_sparsity': args.text_encoder_sparsity,
        'unet_sparsity': args.unet_sparsity,
        'pruning_method': 'safe_conservative',
        'stats': stats
    }
    with open(os.path.join(args.output, "pruning_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Done! Safely pruned model saved to {args.output}")
    print(f"\n⚠️  NEXT: Test the model BEFORE optimizing schedule!")
    print(f"   python scripts/test_pruned_with_schedule.py \\")
    print(f"       --pruned-model {args.output} \\")
    print(f"       --skip-baseline")


if __name__ == "__main__":
    main()
