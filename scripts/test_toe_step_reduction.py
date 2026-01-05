#!/usr/bin/env python3
"""
TOE Progressive Step Reduction Experiment

Hypothesis: TOE's cleaner tag-based conditioning may allow fewer denoising 
steps compared to CLIP-based models while maintaining quality.

This script tests quality/speed tradeoff across different step counts to find
the optimal configuration for TOE-based generation.
"""

import sys
import argparse
from pathlib import Path
import torch
import time
from PIL import Image
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

# Model configuration
MODEL_CONFIGS = {
    'segmind': {
        'repo_id': 'segmind/SSD-1B',
        'name': 'Segmind SSD-1B'
    },
    'illustrious': {
        'repo_id': 'martineux/janku6',
        'name': 'Illustrious Janku6'
    }
}

TEST_PROMPTS = [
    "1girl, solo, blue_eyes, smile, anime style, masterpiece",
    "1girl, long_hair, red_eyes, white_hair, fantasy, detailed",
    "1boy, short_hair, serious, school_uniform, anime",
    "2girls, friends, outdoors, cherry_blossoms, spring",
]


def test_baseline_model(model_key, steps_list, prompt, output_dir):
    """Test baseline SDXL model (no TOE) at different step counts."""
    from diffusers import StableDiffusionXLPipeline
    
    config = MODEL_CONFIGS[model_key]
    print("\n" + "=" * 70)
    print(f"BASELINE: {config['name']} (Standard CLIP)")
    print("=" * 70)
    
    # Load model
    print(f"Loading {config['name']}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config['repo_id'],
        torch_dtype=torch.float32,
    )
    pipe.to("cpu")
    print("✓ Model loaded\n")
    
    results = {}
    images = {}
    
    for steps in steps_list:
        print(f"Testing with {steps} steps...")
        start = time.time()
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
        ).images[0]
        
        elapsed = time.time() - start
        
        # Save image
        img_path = output_dir / f"baseline_{model_key}_steps{steps}.png"
        image.save(img_path)
        
        results[steps] = {
            'time': elapsed,
            'seconds_per_step': elapsed / steps,
            'image_path': str(img_path)
        }
        images[steps] = image
        
        print(f"  ✓ Time: {elapsed:.1f}s ({elapsed/steps:.1f}s/step)\n")
    
    return results, images


def test_toe_model(model_key, steps_list, prompt, toe_checkpoint, vocab_path, output_dir):
    """Test TOE-based model at different step counts."""
    from hqpd.models.sdxl_toe_pipeline import create_toe_pipeline
    
    config = MODEL_CONFIGS[model_key]
    print("\n" + "=" * 70)
    print(f"TOE: {config['name']} (TOE Encoder)")
    print("=" * 70)
    
    # Load TOE pipeline
    print(f"Loading TOE pipeline...")
    pipe = create_toe_pipeline(
        toe_checkpoint_path=toe_checkpoint,
        vocab_path=vocab_path,
        sdxl_model_path=config['repo_id'],
        device='cpu',
        use_hybrid=True,
        torch_dtype=torch.float32,
    )
    print("✓ TOE pipeline loaded\n")
    
    results = {}
    images = {}
    
    for steps in steps_list:
        print(f"Testing with {steps} steps...")
        start = time.time()
        
        image = pipe(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=7.5,
        ).images[0]
        
        elapsed = time.time() - start
        
        # Save image
        img_path = output_dir / f"toe_{model_key}_steps{steps}.png"
        image.save(img_path)
        
        results[steps] = {
            'time': elapsed,
            'seconds_per_step': elapsed / steps,
            'image_path': str(img_path)
        }
        images[steps] = image
        
        print(f"  ✓ Time: {elapsed:.1f}s ({elapsed/steps:.1f}s/step)\n")
    
    return results, images


def create_comparison_grid(baseline_images, toe_images, steps_list, output_path):
    """Create side-by-side comparison grid."""
    n_steps = len(steps_list)
    
    # Get image size (assume all same)
    img_width, img_height = list(baseline_images.values())[0].size
    
    # Create grid: 2 rows (baseline, TOE) x n_steps columns
    grid_width = img_width * n_steps
    grid_height = img_height * 2
    
    grid = Image.new('RGB', (grid_width, grid_height))
    
    # Add baseline images (top row)
    for i, steps in enumerate(steps_list):
        img = baseline_images[steps].resize((img_width, img_height), Image.LANCZOS)
        grid.paste(img, (i * img_width, 0))
    
    # Add TOE images (bottom row)
    for i, steps in enumerate(steps_list):
        img = toe_images[steps].resize((img_width, img_height), Image.LANCZOS)
        grid.paste(img, (i * img_width, img_height))
    
    grid.save(output_path)
    print(f"\n✓ Saved comparison grid: {output_path}")
    
    return grid


def print_analysis(baseline_results, toe_results, model_key):
    """Print detailed analysis and recommendations."""
    print("\n" + "=" * 70)
    print(f"ANALYSIS - {MODEL_CONFIGS[model_key]['name']}")
    print("=" * 70)
    
    print("\n📊 Performance Comparison:\n")
    print(f"{'Steps':<8} {'Baseline':<12} {'TOE':<12} {'Speedup':<10} {'Recommendation'}")
    print("-" * 70)
    
    recommendations = []
    
    for steps in sorted(baseline_results.keys()):
        baseline_time = baseline_results[steps]['time']
        toe_time = toe_results[steps]['time']
        speedup = baseline_time / toe_time
        
        # Determine if this is a good trade-off
        if speedup > 1.3:
            rec = "✅ EXCELLENT"
        elif speedup > 1.15:
            rec = "✅ Good"
        elif speedup > 1.05:
            rec = "⚠️  Marginal"
        else:
            rec = "❌ No benefit"
        
        print(f"{steps:<8} {baseline_time:>8.1f}s    {toe_time:>8.1f}s    {speedup:>6.2f}x     {rec}")
        
        if speedup > 1.2:
            recommendations.append((steps, speedup))
    
    # Find optimal configuration
    print("\n" + "=" * 70)
    print("💡 Recommendations:")
    print("=" * 70)
    
    if recommendations:
        best_steps, best_speedup = max(recommendations, key=lambda x: x[1])
        print(f"\n✅ BEST CONFIGURATION:")
        print(f"   Steps: {best_steps}")
        print(f"   Speedup: {best_speedup:.2f}x")
        print(f"   Time: {toe_results[best_steps]['time']:.1f}s")
        print(f"\n   → Visual quality check required!")
        print(f"   → Compare: baseline_*_steps{best_steps}.png vs toe_*_steps{best_steps}.png")
    else:
        print("\n⚠️  No significant speedup found at tested step counts.")
        print("   → Try testing with more steps (30-50 range)")
        print("   → Or focus on other optimization approaches")
    
    print("\n" + "=" * 70)


def save_results_json(baseline_results, toe_results, model_key, output_path):
    """Save results to JSON for further analysis."""
    data = {
        'model': model_key,
        'model_name': MODEL_CONFIGS[model_key]['name'],
        'baseline': baseline_results,
        'toe': toe_results,
        'comparison': {}
    }
    
    # Calculate speedups
    for steps in baseline_results.keys():
        baseline_time = baseline_results[steps]['time']
        toe_time = toe_results[steps]['time']
        data['comparison'][steps] = {
            'speedup': baseline_time / toe_time,
            'time_saved': baseline_time - toe_time
        }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved detailed results: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Test TOE progressive step reduction hypothesis'
    )
    parser.add_argument('--model', default='illustrious',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='SDXL model to test')
    parser.add_argument('--steps', default='5,10,15,20,30',
                       help='Comma-separated list of step counts to test')
    parser.add_argument('--toe-checkpoint', required=True,
                       help='Path to trained TOE checkpoint')
    parser.add_argument('--vocab', required=True,
                       help='Path to vocabulary.json')
    parser.add_argument('--prompt', default=None,
                       help='Custom prompt (uses default test prompts if not specified)')
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Compare with baseline CLIP model')
    parser.add_argument('--output-dir', default='toe_step_reduction_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse step counts
    steps_list = [int(s.strip()) for s in args.steps.split(',')]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Select prompt
    prompt = args.prompt if args.prompt else TEST_PROMPTS[0]
    
    print("\n" + "=" * 70)
    print("🔬 TOE PROGRESSIVE STEP REDUCTION EXPERIMENT")
    print("=" * 70)
    print(f"\nModel: {MODEL_CONFIGS[args.model]['name']}")
    print(f"Step counts: {steps_list}")
    print(f"Prompt: {prompt}")
    print(f"Output: {output_dir}\n")
    
    # Test TOE model
    toe_results, toe_images = test_toe_model(
        args.model, 
        steps_list, 
        prompt,
        args.toe_checkpoint,
        args.vocab,
        output_dir
    )
    
    # Test baseline if requested
    if args.compare_baseline:
        baseline_results, baseline_images = test_baseline_model(
            args.model,
            steps_list,
            prompt,
            output_dir
        )
        
        # Create comparison grid
        create_comparison_grid(
            baseline_images,
            toe_images,
            steps_list,
            output_dir / f"comparison_grid_{args.model}.png"
        )
        
        # Print analysis
        print_analysis(baseline_results, toe_results, args.model)
        
        # Save JSON results
        save_results_json(
            baseline_results,
            toe_results,
            args.model,
            output_dir / f"results_{args.model}.json"
        )
    else:
        print("\n" + "=" * 70)
        print("TOE RESULTS ONLY (no baseline comparison)")
        print("=" * 70)
        
        for steps in sorted(toe_results.keys()):
            result = toe_results[steps]
            print(f"Steps: {steps:<3}  Time: {result['time']:>6.1f}s  ({result['seconds_per_step']:.1f}s/step)")
        
        print("\n💡 Run with --compare-baseline to see speedup analysis")
    
    print("\n" + "=" * 70)
    print("✅ EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Visually inspect generated images for quality")
    print("2. Identify optimal step count (best speed/quality balance)")
    print("3. If successful, integrate into production pipeline")


if __name__ == "__main__":
    main()
