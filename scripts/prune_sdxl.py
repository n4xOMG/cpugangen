#!/usr/bin/env python3
"""
Prune SDXL Model using Magnitude Pruning

Applies magnitude-based pruning to SDXL components:
- Text Encoder 1 & 2: 47.5% sparsity
- UNet: 35% sparsity

Usage:
    python scripts/prune_sdxl.py \\
        --model martineux/janku6 \\
        --text-encoder-sparsity 0.475 \\
        --unet-sparsity 0.35 \\
        --output checkpoints/pruned_sdxl

Reference: "Efficient Pruning of Text-to-Image Models" (arxiv:2411.15113)
"""

import argparse
import os
import sys
import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import save_file

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.magnitude_pruning import MagnitudePruner


def prune_sdxl(
    model_id: str,
    output_path: str,
    text_encoder_sparsity: float = 0.475,
    unet_sparsity: float = 0.35,
    device: str = "cpu",
    torch_dtype: torch.dtype = torch.float32
):
    """
    Prune SDXL model using magnitude pruning.
    
    Args:
        model_id: HuggingFace model ID or local path
        output_path: Output directory
        text_encoder_sparsity: Sparsity for text encoders (default: 47.5%)
        unet_sparsity: Sparsity for UNet (default: 35%)
        device: Device to load model on
        torch_dtype: Data type for model
    """
    print(f"\n{'='*60}")
    print("SDXL Magnitude Pruning")
    print(f"{'='*60}\n")
    print(f"Model: {model_id}")
    print(f"Text Encoder Sparsity: {text_encoder_sparsity*100:.1f}%")
    print(f"UNet Sparsity: {unet_sparsity*100:.1f}%")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print()
    
    # Load model
    print("Loading SDXL Pipeline...")
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        use_safetensors=True
    )
    pipeline = pipeline.to(device)
    print("✅ Model loaded\n")
    
    # Count initial parameters
    total_params_before = sum(
        p.numel() for component in [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.unet]
        if component is not None
        for p in component.parameters()
    )
    print(f"Total parameters before pruning: {total_params_before:,}\n")
    
    # Create pruner
    pruner = MagnitudePruner(verbose=True)
    
    # Prune components
    stats = pruner.prune_sdxl_components(
        pipeline=pipeline,
        text_encoder_sparsity=text_encoder_sparsity,
        unet_sparsity=unet_sparsity
    )
    
    # Save pruned model
    print(f"Saving pruned model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    # Save each component
    if pipeline.text_encoder is not None:
        te1_path = os.path.join(output_path, "text_encoder")
        pipeline.text_encoder.save_pretrained(te1_path)
        print(f"  ✅ Text Encoder 1 saved to {te1_path}")
    
    if pipeline.text_encoder_2 is not None:
        te2_path = os.path.join(output_path, "text_encoder_2")
        pipeline.text_encoder_2.save_pretrained(te2_path)
        print(f"  ✅ Text Encoder 2 saved to {te2_path}")
    
    if pipeline.unet is not None:
        unet_path = os.path.join(output_path, "unet")
        pipeline.unet.save_pretrained(unet_path)
        print(f"  ✅ UNet saved to {unet_path}")
    
    # Save other components (VAE, scheduler, tokenizers)
    if pipeline.vae is not None:
        vae_path = os.path.join(output_path, "vae")
        pipeline.vae.save_pretrained(vae_path)
    
    if pipeline.scheduler is not None:
        scheduler_path = os.path.join(output_path, "scheduler")
        pipeline.scheduler.save_pretrained(scheduler_path)
    
    if hasattr(pipeline, 'tokenizer') and pipeline.tokenizer is not None:
        tokenizer_path = os.path.join(output_path, "tokenizer")
        pipeline.tokenizer.save_pretrained(tokenizer_path)
    
    if hasattr(pipeline, 'tokenizer_2') and pipeline.tokenizer_2 is not None:
        tokenizer2_path = os.path.join(output_path, "tokenizer_2")
        pipeline.tokenizer_2.save_pretrained(tokenizer2_path)
    
    # Save pruning metadata
    import json
    metadata = {
        'base_model': model_id,
        'text_encoder_sparsity': text_encoder_sparsity,
        'unet_sparsity': unet_sparsity,
        'pruning_stats': {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv 
                              for kk, vv in v.items()} 
                         for k, v in stats.items()},
        'total_params_before': total_params_before,
    }
    
    metadata_path = os.path.join(output_path, "pruning_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Pruning complete! Model saved to {output_path}")
    print(f"📋 Metadata saved to {metadata_path}")
    
    return pipeline, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune SDXL using Magnitude Pruning")
    parser.add_argument(
        "--model",
        type=str,
        default="martineux/janku6",
        help="Model ID or path (default: martineux/janku6 - Illustrious)"
    )
    parser.add_argument(
        "--text-encoder-sparsity",
        type=float,
        default=0.475,
        help="Text encoder sparsity (default: 0.475 = 47.5%%)"
    )
    parser.add_argument(
        "--unet-sparsity",
        type=float,
        default=0.35,
        help="UNet sparsity (default: 0.35 = 35%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/pruned_sdxl",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16"],
        help="Data type"
    )
    
    args = parser.parse_args()
    
    # Convert dtype
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    
    # Run pruning
    prune_sdxl(
        model_id=args.model,
        output_path=args.output,
        text_encoder_sparsity=args.text_encoder_sparsity,
        unet_sparsity=args.unet_sparsity,
        device=args.device,
        torch_dtype=dtype
    )
