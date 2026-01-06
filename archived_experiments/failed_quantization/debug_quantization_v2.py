#!/usr/bin/env python3
"""
Comprehensive Quantization Test Suite (v2)

Tests multiple quantization approaches across different SDXL models:
- Option A: Per-Channel INT8
- Option B: Selective INT8 (FFN only)
- Option C: Weights-Only INT8 (W8A32)

Supports multiple models:
- Segmind SSD-1B (baseline)
- Illustrious (anime-focused)
- Custom SDXL checkpoints
"""

import sys
import argparse
from pathlib import Path
import torch
import time
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# Model configurations
MODEL_CONFIGS = {
    'segmind': {
        'repo_id': 'segmind/SSD-1B',
        'prompt': '1girl, solo, blue_eyes, smile, anime',
        'name': 'Segmind SSD-1B'
    },
    'illustrious': {
        'repo_id': 'martineux/janku6',
        'prompt': '1girl, solo, blue_eyes, smile, anime style, masterpiece, best quality',
        'name': 'Illustrious Janku6'
    }
}


def load_model(model_key):
    """Load SDXL model by key."""
    from diffusers import StableDiffusionXLPipeline
    
    config = MODEL_CONFIGS.get(model_key)
    if not config:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    print(f"Loading {config['name']}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config['repo_id'],
        torch_dtype=torch.float32,
    )
    pipe.to("cpu")
    
    return pipe, config


def test_baseline(model_key, steps=5):
    """Test baseline FP32 model (no quantization)."""
    print("=" * 70)
    print(f"Baseline: FP32 (no quantization)")
    print("=" * 70)
    
    pipe, config = load_model(model_key)
    
    print(f"Generating with {steps} steps...")
    start = time.time()
    image = pipe(
        prompt=config['prompt'],
        num_inference_steps=steps,
        guidance_scale=7.5,
    ).images[0]
    elapsed = time.time() - start
    
    output_path = f"debug_{model_key}_baseline.png"
    image.save(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"✓ Time: {elapsed:.1f}s for {steps} steps ({elapsed/steps:.1f}s/step)")
    
    return elapsed, image


def test_perchannel(model_key, steps=5):
    """Test per-channel INT8 quantization."""
    print("\n" + "=" * 70)
    print(f"Option A: Per-Channel INT8")
    print("=" * 70)
    
    from diffusers import StableDiffusionXLPipeline
    from hqpd.quantization import quantize_perchannel
    
    pipe, config = load_model(model_key)
    
    print("Applying per-channel quantization...")
    pipe.unet = quantize_perchannel(pipe.unet)
    
    print(f"Generating with {steps} steps...")
    start = time.time()
    image = pipe(
        prompt=config['prompt'],
        num_inference_steps=steps,
        guidance_scale=7.5,
    ).images[0]
    elapsed = time.time() - start
    
    output_path = f"debug_{model_key}_perchannel.png"
    image.save(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"✓ Time: {elapsed:.1f}s for {steps} steps ({elapsed/steps:.1f}s/step)")
    
    return elapsed, image


def test_selective(model_key, steps=5):
    """Test selective INT8 quantization (FFN only)."""
    print("\n" + "=" * 70)
    print(f"Option B: Selective INT8 (FFN only)")
    print("=" * 70)
    
    from diffusers import StableDiffusionXLPipeline
    from hqpd.quantization import quantize_selective
    
    pipe, config = load_model(model_key)
    
    print("Applying selective quantization...")
    pipe.unet = quantize_selective(pipe.unet, verbose=True)
    
    print(f"Generating with {steps} steps...")
    start = time.time()
    image = pipe(
        prompt=config['prompt'],
        num_inference_steps=steps,
        guidance_scale=7.5,
    ).images[0]
    elapsed = time.time() - start
    
    output_path = f"debug_{model_key}_selective.png"
    image.save(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"✓ Time: {elapsed:.1f}s for {steps} steps ({elapsed/steps:.1f}s/step)")
    
    return elapsed, image


def test_weights_only(model_key, steps=5):
    """Test weights-only INT8 quantization."""
    print("\n" + "=" * 70)
    print(f"Option C: Weights-Only INT8 (W8A32)")
    print("=" * 70)
    
    from diffusers import StableDiffusionXLPipeline
    from hqpd.quantization import quantize_weights_only
    
    pipe, config = load_model(model_key)
    
    print("Applying weights-only quantization...")
    pipe.unet = quantize_weights_only(pipe.unet)
    
    print(f"Generating with {steps} steps...")
    start = time.time()
    image = pipe(
        prompt=config['prompt'],
        num_inference_steps=steps,
        guidance_scale=7.5,
    ).images[0]
    elapsed = time.time() - start
    
    output_path = f"debug_{model_key}_weightsonly.png"
    image.save(output_path)
    print(f"✓ Saved: {output_path}")
    print(f"✓ Time: {elapsed:.1f}s for {steps} steps ({elapsed/steps:.1f}s/step)")
    
    return elapsed, image


def create_comparison_grid(model_key, images, titles):
    """Create side-by-side comparison grid."""
    # Resize all images to same size (smallest dimension)
    sizes = [img.size for img in images]
    min_width = min(s[0] for s in sizes)
    min_height = min(s[1] for s in sizes)
    
    resized = [img.resize((min_width, min_height), Image.LANCZOS) for img in images]
    
    # Create grid (2x2 or 2x3 depending on count)
    n_imgs = len(images)
    if n_imgs <= 2:
        grid_size = (n_imgs, 1)
    elif n_imgs <= 4:
        grid_size = (2, 2)
    else:
        grid_size = (2, 3)
    
    grid_width = grid_size[0] * min_width
    grid_height = grid_size[1] * min_height
    
    grid = Image.new('RGB', (grid_width, grid_height))
    
    for idx, img in enumerate(resized):
        x = (idx % grid_size[0]) * min_width
        y = (idx // grid_size[0]) * min_height
        grid.paste(img, (x, y))
    
    grid_path = f"comparison_grid_{model_key}.png"
    grid.save(grid_path)
    print(f"\n✓ Saved comparison grid: {grid_path}")
    
    return grid


def print_summary(model_key, results):
    """Print summary report."""
    print("\n" + "=" * 70)
    print(f"SUMMARY - {MODEL_CONFIGS[model_key]['name']}")
    print("=" * 70)
    
    baseline_time = results.get('baseline', {}).get('time')
    
    if baseline_time:
        print(f"\nBaseline (FP32):        {baseline_time:.1f}s  (1.00x)")
        
        for method, data in results.items():
            if method == 'baseline':
                continue
            
            method_time = data['time']
            speedup = baseline_time / method_time
            
            method_names = {
                'perchannel': 'Per-Channel INT8:',
                'selective': 'Selective INT8:   ',
                'weightsonly': 'Weights-Only INT8:'
            }
            
            name = method_names.get(method, method)
            print(f"{name}    {method_time:.1f}s  ({speedup:.2f}x speedup)")
    
    print("\n" + "=" * 70)
    print("Visual Quality Comparison:")
    print("  → Check generated images side-by-side")
    print(f"  → See comparison_grid_{model_key}.png")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Test quantization methods on SDXL models')
    parser.add_argument('--model', default='segmind', 
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Model to test')
    parser.add_argument('--methods', default='all',
                       choices=['all', 'quick', 'perchannel', 'selective', 'weightsonly'],
                       help='Which methods to test')
    parser.add_argument('--steps', type=int, default=5,
                       help='Number of inference steps')
    
    args = parser.parse_args()
    
    print("\n🔬 Quantization Test Suite v2\n")
    print(f"Model: {MODEL_CONFIGS[args.model]['name']}")
    print(f"Methods: {args.methods}")
    print(f"Steps: {args.steps}\n")
    
    results = {}
    images = []
    titles = []
    
    # Always run baseline
    print("Running baseline test...")
    baseline_time, baseline_img = test_baseline(args.model, args.steps)
    results['baseline'] = {'time': baseline_time}
    images.append(baseline_img)
    titles.append('Baseline (FP32)')
    
    # Run selected methods
    methods_to_run = []
    if args.methods == 'all':
        methods_to_run = ['perchannel', 'selective', 'weightsonly']
    elif args.methods == 'quick':
        methods_to_run = ['perchannel', 'selective']
    else:
        methods_to_run = [args.methods]
    
    test_functions = {
        'perchannel': test_perchannel,
        'selective': test_selective,
        'weightsonly': test_weights_only
    }
    
    for method in methods_to_run:
        test_func = test_functions[method]
        method_time, method_img = test_func(args.model, args.steps)
        results[method] = {'time': method_time}
        images.append(method_img)
        titles.append(method.capitalize())
    
    # Create comparison grid
    create_comparison_grid(args.model, images, titles)
    
    # Print summary
    print_summary(args.model, results)
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    main()
