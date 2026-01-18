#!/usr/bin/env python3
"""
Quantize TOE + Segmind pipeline for CPU inference.

This script applies dynamic INT8 quantization to the UNet,
targeting ~1.5x speedup on CPU.

Usage:
    python scripts/quantize_pipeline.py \
        --output_dir checkpoints/quantized
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.quantization import (
    quantize_linear_layers,
    save_quantized_model,
    configure_cpu_inference,
)
from hqpd.models.sdxl_toe_pipeline import create_toe_pipeline


def main():
    parser = argparse.ArgumentParser(description="Quantize pipeline for CPU")
    
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
        "--output_dir",
        type=str,
        default="checkpoints/quantized",
        help="Output directory for quantized models",
    )
    parser.add_argument(
        "--quantize_vae",
        action="store_true",
        help="Also quantize VAE decoder",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Quantizing TOE + Segmind Pipeline for CPU")
    print("=" * 70)
    
    # Configure CPU
    print("\n🔧 Step 1: Configure CPU...")
    cpu_config = configure_cpu_inference()
    
    # Load pipeline
    print("\n📦 Step 2: Load pipeline...")
    print("   (This may take a few minutes on first run)")
    
    pipeline = create_toe_pipeline(
        toe_checkpoint_path=args.toe_checkpoint,
        vocab_path=args.vocab,
        sdxl_model_path="segmind/SSD-1B",
        device="cpu",  # Load directly on CPU
    )
    
    print("   ✓ Pipeline loaded")
    
    # Quantize UNet
    print("\n🔧 Step 3: Quantize UNet...")
    print("   Target: Linear layers only (dynamic INT8)")
    print("   Expected speedup: ~1.5x on CPU")
    
    original_unet = pipeline.unet
    quantized_unet = quantize_linear_layers(original_unet, inplace=True)
    pipeline.unet = quantized_unet
    
    # Optionally quantize VAE
    if args.quantize_vae:
        print("\n🔧 Step 4: Quantize VAE decoder...")
        original_vae = pipeline.vae
        quantized_vae = quantize_linear_layers(original_vae, inplace=True)
        pipeline.vae = quantized_vae
    
    # Save quantized models
    print("\n💾 Step 5: Save quantized models...")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save UNet
    save_quantized_model(
        pipeline.unet,
        output_dir / "unet_int8.pt"
    )
    
    # Save VAE if quantized
    if args.quantize_vae:
        save_quantized_model(
            pipeline.vae,
            output_dir / "vae_int8.pt"
        )
    
    # Save configuration
    config_file = output_dir / "quantization_config.txt"
    with open(config_file, 'w') as f:
        f.write("Quantization Configuration\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"UNet: Quantized (Dynamic INT8, Linear layers)\n")
        f.write(f"VAE: {'Quantized' if args.quantize_vae else 'Original (FP32)'}\n")
        f.write(f"TOE: Original (FP32)\n\n")
        f.write(f"CPU Config:\n")
        for key, value in cpu_config.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"   ✓ Config saved to: {config_file}")
    
    # Test generation
    print("\n🎨 Step 6: Test quantized pipeline...")
    test_prompt = "1girl, solo, smile, anime, high_quality"
    
    print(f"   Generating test image with prompt: '{test_prompt}'")
    print("   (This tests correctness, not speed)")
    
    try:
        image = pipeline(
            prompt=test_prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
        ).images[0]
        
        # Save test image
        test_output = output_dir / "test_quantized.png"
        image.save(test_output)
        print(f"   ✓ Test successful! Image saved to: {test_output}")
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        print("   This may indicate quantization issues")
        return 1
    
    print("\n" + "=" * 70)
    print("✓ Quantization Complete!")
    print("=" * 70)
    print(f"\nQuantized models saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Benchmark on CPU: python scripts/benchmark_cpu.py")
    print("  2. Validate quality: python scripts/validate_quality.py")
    print("\nExpected CPU performance:")
    print("  • Speedup: ~1.5x (vs FP32 baseline)")
    print("  • Time: ~90-120s per image (from ~180s baseline)")
    print("  • Model size: ~2GB (from ~6GB)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
