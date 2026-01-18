#!/usr/bin/env python3
"""
Debug test: Compare Segmind alone vs with TOE to isolate quality issues.
"""

import sys
from pathlib import Path
import torch
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_segmind_alone():
    """Test Segmind without TOE modification."""
    print("=" * 70)
    print("Test 1: Pure Segmind (no TOE)")
    print("=" * 70)
    
    from diffusers import StableDiffusionXLPipeline
    
    # Load Segmind with FP32 (CPU compatible)
    print("Loading Segmind SSD-1B...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float32,  # FP32 for CPU
    )
    pipe.to("cpu")
    
    # Quick generation (5 steps only for speed)
    print("Generating with 5 steps...")
    start = time.time()
    image = pipe(
        prompt="1girl, solo, blue_eyes, smile, anime",
        num_inference_steps=5,
        guidance_scale=7.5,
    ).images[0]
    elapsed = time.time() - start
    
    image.save("debug_segmind_alone.png")
    print(f"✓ Saved: debug_segmind_alone.png")
    print(f"✓ Time: {elapsed:.1f}s for 5 steps ({elapsed/5:.1f}s/step)")
    
    return elapsed

def test_segmind_quantized():
    """Test Segmind with INT8 quantization."""
    print("\n" + "=" * 70)
    print("Test 2: Segmind + INT8 Quantization (no TOE)")
    print("=" * 70)
    
    from diffusers import StableDiffusionXLPipeline
    from hqpd.quantization import quantize_linear_layers
    
    # Load Segmind
    print("Loading Segmind SSD-1B...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float32,
    )
    pipe.to("cpu")
    
    # Quantize UNet only
    print("Applying INT8 quantization...")
    pipe.unet = quantize_linear_layers(pipe.unet)
    
    # Quick generation
    print("Generating with 5 steps...")
    start = time.time()
    image = pipe(
        prompt="1girl, solo, blue_eyes, smile, anime",
        num_inference_steps=5,
        guidance_scale=7.5,
    ).images[0]
    elapsed = time.time() - start
    
    image.save("debug_segmind_int8.png")
    print(f"✓ Saved: debug_segmind_int8.png")
    print(f"✓ Time: {elapsed:.1f}s for 5 steps ({elapsed/5:.1f}s/step)")
    
    return elapsed

if __name__ == "__main__":
    print("\n🔍 Debugging Quantization Issues\n")
    
    # Test 1: Pure Segmind
    t1 = test_segmind_alone()
    
    # Test 2: Quantized
    t2 = test_segmind_quantized()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Pure Segmind: {t1:.1f}s")
    print(f"INT8 Quantized: {t2:.1f}s")
    print(f"Speedup: {t1/t2:.2f}x")
    print("\nCheck images for quality comparison:")
    print("  - debug_segmind_alone.png")
    print("  - debug_segmind_int8.png")
