#!/usr/bin/env python3
"""
Validate LCM-LoRA and SDXL-Lightning with Illustrious model.

Tests both pre-trained acceleration methods to find the best option
for fast CPU inference without custom training.

Compares:
1. Baseline (20 steps)
2. LCM-LoRA (4 steps)
3. SDXL-Lightning (4 steps)
"""

import argparse
import sys
from pathlib import Path
import torch
import time
from PIL import Image
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_baseline(model_id, prompt, device="cpu", num_steps=20):
    """Test baseline generation."""
    from diffusers import StableDiffusionXLPipeline
    
    print("\n" + "=" * 70)
    print("📦 BASELINE (Standard Generation)")
    print("=" * 70)
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )
    pipe.to(device)
    
    print(f"✓ Pipeline loaded on {device}")
    print(f"Testing with {num_steps} steps...\n")
    
    start = time.time()
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=7.5,
    ).images[0]
    elapsed = time.time() - start
    
    result = {
        'method': 'baseline',
        'steps': num_steps,
        'time': elapsed,
        'seconds_per_step': elapsed / num_steps,
        'image': image
    }
    
    print(f"Time: {elapsed:.1f}s ({elapsed/num_steps:.1f}s/step)")
    
    # Clean up
    del pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return result


def test_lcm_lora(model_id, prompt, device="cpu", num_steps=4):
    """Test LCM-LoRA acceleration."""
    from diffusers import DiffusionPipeline, LCMScheduler
    
    print("\n" + "=" * 70)
    print("⚡ LCM-LoRA (Latent Consistency Model)")
    print("=" * 70)
    
    try:
        # Load base model
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        )
        
        # Load LCM-LoRA weights
        print("Loading LCM-LoRA weights...")
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
        
        # Switch to LCM scheduler
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
        pipe.to(device)
        
        print(f"✓ LCM-LoRA loaded on {device}")
        print(f"Testing with {num_steps} steps...\n")
        
        start = time.time()
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=1.0,  # LCM uses low guidance
        ).images[0]
        elapsed = time.time() - start
        
        result = {
            'method': 'lcm-lora',
            'steps': num_steps,
            'time': elapsed,
            'seconds_per_step': elapsed / num_steps,
            'image': image,
            'compatible': True
        }
        
        print(f"Time: {elapsed:.1f}s ({elapsed/num_steps:.1f}s/step)")
        
        # Clean up
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
        
    except Exception as e:
        print(f"❌ LCM-LoRA failed: {e}")
        return {
            'method': 'lcm-lora',
            'compatible': False,
            'error': str(e)
        }


def test_sdxl_lightning(model_id, prompt, device="cpu", num_steps=4):
    """Test SDXL-Lightning acceleration."""
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
    print("\n" + "=" * 70)
    print("⚡ SDXL-Lightning (ByteDance)")
    print("=" * 70)
    
    try:
        # Load base model
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        )
        
        # Download Lightning checkpoint for 4-step
        print("Downloading SDXL-Lightning checkpoint...")
        ckpt = hf_hub_download(
            "ByteDance/SDXL-Lightning",
            "sdxl_lightning_4step_lora.safetensors"
        )
        
        # Load Lightning weights
        print("Loading Lightning weights...")
        state_dict = load_file(ckpt)
        pipe.load_lora_weights(state_dict)
        
        # Use Lightning scheduler
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing"
        )
        
        pipe.to(device)
        
        print(f"✓ SDXL-Lightning loaded on {device}")
        print(f"Testing with {num_steps} steps...\n")
        
        start = time.time()
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=0.0,  # Lightning uses CFG-free mode
        ).images[0]
        elapsed = time.time() - start
        
        result = {
            'method': 'sdxl-lightning',
            'steps': num_steps,
            'time': elapsed,
            'seconds_per_step': elapsed / num_steps,
            'image': image,
            'compatible': True
        }
        
        print(f"Time: {elapsed:.1f}s ({elapsed/num_steps:.1f}s/step)")
        
        # Clean up
        del pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return result
        
    except Exception as e:
        print(f"❌ SDXL-Lightning failed: {e}")
        return {
            'method': 'sdxl-lightning',
            'compatible': False,
            'error': str(e)
        }


def create_comparison_grid(results, output_path):
    """Create visual comparison grid."""
    
    # Filter only successful results with images
    valid_results = [r for r in results if 'image' in r]
    
    if not valid_results:
        print("⚠️  No valid images to create grid")
        return
    
    n_methods = len(valid_results)
    
    # Get image size
    img_width, img_height = valid_results[0]['image'].size
    
    # Create grid: n_methods columns × 1 row
    grid_width = img_width * n_methods
    grid_height = img_height
    
    grid = Image.new('RGB', (grid_width, grid_height))
    
    for i, result in enumerate(valid_results):
        img = result['image'].resize((img_width, img_height), Image.LANCZOS)
        grid.paste(img, (i * img_width, 0))
    
    grid.save(output_path)
    print(f"\n✓ Comparison grid saved: {output_path}")


def print_comparison(results):
    """Print detailed comparison."""
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'Steps':<8} {'Total Time':<12} {'Per Step':<12} {'Speedup'}")
    print("-" * 70)
    
    baseline_time = next((r['time'] for r in results if r['method'] == 'baseline'), None)
    
    for result in results:
        if not result.get('compatible', True):
            print(f"{result['method']:<20} {'N/A':<8} {'FAILED':<12} {'N/A':<12} N/A")
            continue
        
        method = result['method']
        steps = result['steps']
        total_time = result['time']
        per_step = result['seconds_per_step']
        
        if baseline_time:
            speedup = baseline_time / total_time
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{method:<20} {steps:<8} {total_time:>8.1f}s    {per_step:>8.1f}s    {speedup_str}")
    
    print("-" * 70)
    
    # Print recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    compatible_results = [r for r in results if r.get('compatible', True) and 'time' in r]
    
    if len(compatible_results) > 1:
        # Find fastest
        fastest = min(compatible_results, key=lambda x: x['time'])
        
        print(f"\n✅ FASTEST METHOD: {fastest['method'].upper()}")
        print(f"   Steps: {fastest['steps']}")
        print(f"   Time: {fastest['time']:.1f}s")
        
        if baseline_time:
            speedup = baseline_time / fastest['time']
            print(f"   Speedup: {speedup:.2f}x")
            print(f"\n   With ONNX (3.4x): {fastest['time'] / 3.4:.1f}s")
            print(f"   Total speedup: {speedup * 3.4:.2f}x 🚀")
    
    print("\n💾 Next steps:")
    print("1. Check visual quality in comparison grid")
    print("2. Choose best method based on speed + quality")
    print("3. Export to ONNX for additional 3.4x speedup")


def save_results_json(results, output_path):
    """Save results to JSON."""
    data = []
    
    for r in results:
        entry = {
            'method': r['method'],
            'compatible': r.get('compatible', True),
        }
        
        if 'time' in r:
            entry.update({
                'steps': r['steps'],
                'time': r['time'],
                'seconds_per_step': r['seconds_per_step']
            })
        
        if 'error' in r:
            entry['error'] = r['error']
        
        data.append(entry)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate LCM and Lightning with Illustrious'
    )
    parser.add_argument('--model', default='martineux/janku6',
                       help='SDXL model to test')
    parser.add_argument('--prompt', default='1girl, solo, blue_eyes, smile, anime style, masterpiece',
                       help='Test prompt')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device for testing')
    parser.add_argument('--baseline-steps', type=int, default=20,
                       help='Steps for baseline test')
    parser.add_argument('--fast-steps', type=int, default=4,
                       help='Steps for LCM/Lightning')
    parser.add_argument('--output-dir', default='accelerator_validation',
                       help='Output directory')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline test (faster testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 70)
    print("🔬 ACCELERATOR VALIDATION")
    print("=" * 70)
    print(f"\nModel: {args.model}")
    print(f"Device: {args.device}")
    print(f"Prompt: {args.prompt}")
    
    results = []
    
    # Test baseline
    if not args.skip_baseline:
        baseline = test_baseline(args.model, args.prompt, args.device, args.baseline_steps)
        results.append(baseline)
        baseline['image'].save(output_dir / "baseline.png")
    
    # Test LCM-LoRA
    lcm = test_lcm_lora(args.model, args.prompt, args.device, args.fast_steps)
    results.append(lcm)
    if lcm.get('compatible'):
        lcm['image'].save(output_dir / "lcm_lora.png")
    
    # Test SDXL-Lightning
    lightning = test_sdxl_lightning(args.model, args.prompt, args.device, args.fast_steps)
    results.append(lightning)
    if lightning.get('compatible'):
        lightning['image'].save(output_dir / "sdxl_lightning.png")
    
    # Create comparison grid
    create_comparison_grid(results, output_dir / "comparison.png")
    
    # Print comparison
    print_comparison(results)
    
    # Save JSON
    save_results_json(results, output_dir / "results.json")
    
    print("\n" + "=" * 70)
    print(f"✅ VALIDATION COMPLETE - Results in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
