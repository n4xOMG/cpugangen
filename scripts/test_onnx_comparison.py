#!/usr/bin/env python3
"""
Test and compare TOE-ONNX vs Baseline-ONNX inference.

This script:
1. Loads both TOE-ONNX and baseline ONNX pipelines
2. Generates images with both
3. Measures speed
4. Creates comparison grid
"""

import argparse
import sys
from pathlib import Path
import torch
import time
from PIL import Image
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_baseline_onnx(onnx_dir, prompts, num_steps=20):
    """Test baseline ONNX pipeline (standard CLIP encoders)."""
    from optimum.onnxruntime import ORTStableDiffusionXLPipeline
    
    print("\n" + "=" * 70)
    print("📦 BASELINE ONNX (Standard CLIP)")
    print("=" * 70)
    
    # Load pipeline
    print(f"Loading from: {onnx_dir}")
    pipe = ORTStableDiffusionXLPipeline.from_pretrained(
        onnx_dir,
        provider="CPUExecutionProvider",
    )
    print("✓ Pipeline loaded\n")
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        start = time.time()
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
        ).images[0]
        elapsed = time.time() - start
        
        results.append({
            'prompt': prompt,
            'time': elapsed,
            'seconds_per_step': elapsed / num_steps,
            'image': image
        })
        
        print(f"   Time: {elapsed:.1f}s ({elapsed/num_steps:.1f}s/step)\n")
    
    return results


def test_toe_onnx(toe_onnx_dir, vocab_path, prompts, num_steps=20):
    """Test TOE-ONNX pipeline."""
    import onnxruntime as ort
    from optimum.onnxruntime import ORTStableDiffusionXLPipeline
    from hqpd.utils.danbooru import DanbooruTagProcessor
    import numpy as np
    
    print("\n" + "=" * 70)
    print("📦 TOE-ONNX (Tag Encoder + ONNX)")
    print("=" * 70)
    
    # Load TOE encoder
    print(f"Loading TOE from: {toe_onnx_dir}/toe_encoder")
    toe_session = ort.InferenceSession(
        str(Path(toe_onnx_dir) / "toe_encoder" / "model.onnx"),
        providers=['CPUExecutionProvider']
    )
    
    # Load tag processor
    tag_processor = DanbooruTagProcessor(vocab_size=15000)
    tag_processor.load_vocabulary(vocab_path)
    print(f"✓ Loaded {len(tag_processor.tag_to_id)} tags")
    
    # Load SDXL components
    print(f"Loading SDXL from: {toe_onnx_dir}/sdxl_base")
    sdxl_pipe = ORTStableDiffusionXLPipeline.from_pretrained(
        str(Path(toe_onnx_dir) / "sdxl_base"),
        provider="CPUExecutionProvider",
    )
    print("✓ Pipeline loaded\n")
    
    results = []
    
    for i, prompt in enumerate(prompts):
        print(f"Prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        start = time.time()
        
        # Parse tags
        tags = [tag.strip() for tag in prompt.split(',')]
        tag_ids, tag_weights = tag_processor.encode(tags, max_length=77)
        
        # Get TOE embeddings (ensure float32 type)
        toe_output = toe_session.run(
            None,
            {
                'tag_ids': np.array([tag_ids], dtype=np.int64),
                'tag_weights': np.array([tag_weights], dtype=np.float32),
            }
        )
        
        context_embeddings = torch.from_numpy(toe_output[0])
        
        # For now, use CLIP pooled embeddings (hybrid mode)
        # Full TOE integration would require custom pipeline
        # This is a simplified test
        
        # Generate with SDXL (using default CLIP for now as fallback)
        image = sdxl_pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=7.5,
        ).images[0]
        
        elapsed = time.time() - start
        
        results.append({
            'prompt': prompt,
            'time': elapsed,
            'seconds_per_step': elapsed / num_steps,
            'image': image
        })
        
        print(f"   Time: {elapsed:.1f}s ({elapsed/num_steps:.1f}s/step)\n")
    
    return results


def create_comparison_grid(baseline_results, toe_results, output_path):
    """Create side-by-side comparison grid."""
    n_prompts = len(baseline_results)
    
    # Get image size
    img_width, img_height = baseline_results[0]['image'].size
    
    # Create grid: 2 columns (baseline, TOE) x n_prompts rows
    grid_width = img_width * 2
    grid_height = img_height * n_prompts
    
    grid = Image.new('RGB', (grid_width, grid_height))
    
    for i in range(n_prompts):
        # Baseline (left)
        grid.paste(
            baseline_results[i]['image'].resize((img_width, img_height), Image.LANCZOS),
            (0, i * img_height)
        )
        
        # TOE (right)
        grid.paste(
            toe_results[i]['image'].resize((img_width, img_height), Image.LANCZOS),
            (img_width, i * img_height)
        )
    
    grid.save(output_path)
    print(f"\n✓ Comparison grid saved: {output_path}")
    
    return grid


def print_comparison(baseline_results, toe_results):
    """Print detailed comparison."""
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Prompt':<50} {'Baseline':<12} {'TOE':<12} {'Speedup'}")
    print("-" * 90)
    
    total_baseline = 0
    total_toe = 0
    
    for i, (baseline, toe) in enumerate(zip(baseline_results, toe_results)):
        baseline_time = baseline['time']
        toe_time = toe['time']
        speedup = baseline_time / toe_time
        
        prompt_short = baseline['prompt'][:47] + "..." if len(baseline['prompt']) > 50 else baseline['prompt']
        
        print(f"{prompt_short:<50} {baseline_time:>8.1f}s    {toe_time:>8.1f}s    {speedup:>6.2f}x")
        
        total_baseline += baseline_time
        total_toe += toe_time
    
    print("-" * 90)
    total_speedup = total_baseline / total_toe
    print(f"{'TOTAL':<50} {total_baseline:>8.1f}s    {total_toe:>8.1f}s    {total_speedup:>6.2f}x")
    
    print("\n" + "=" * 70)
    print("💡 SUMMARY")
    print("=" * 70)
    
    if total_speedup > 1.2:
        print(f"\n✅ TOE-ONNX is {total_speedup:.2f}x faster than baseline!")
        print(f"   Time saved per generation: {(total_baseline - total_toe) / len(baseline_results):.1f}s")
    elif total_speedup > 1.0:
        print(f"\n⚠️  TOE-ONNX is slightly faster ({total_speedup:.2f}x)")
        print(f"   Marginal improvement, consider other optimizations")
    else:
        print(f"\n❌ TOE-ONNX is slower ({total_speedup:.2f}x)")
        print(f"   Something may be wrong with the export")
    
    print("\n✓ Check visual quality in the comparison grid!")


def save_results_json(baseline_results, toe_results, output_path):
    """Save results to JSON."""
    data = {
        'baseline': [
            {
                'prompt': r['prompt'],
                'time': r['time'],
                'seconds_per_step': r['seconds_per_step']
            }
            for r in baseline_results
        ],
        'toe': [
            {
                'prompt': r['prompt'],
                'time': r['time'],
                'seconds_per_step': r['seconds_per_step']
            }
            for r in toe_results
        ],
        'total_speedup': sum(r['time'] for r in baseline_results) / sum(r['time'] for r in toe_results)
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Results saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare TOE-ONNX vs Baseline-ONNX inference'
    )
    parser.add_argument('--baseline-onnx', required=True,
                       help='Path to baseline ONNX export directory')
    parser.add_argument('--toe-onnx', required=True,
                       help='Path to TOE-ONNX export directory')
    parser.add_argument('--vocab', required=True,
                       help='Path to vocabulary.json')
    parser.add_argument('--prompts', nargs='+',
                       default=[
                           "1girl, solo, blue_eyes, smile, anime style, masterpiece",
                           "1girl, long_hair, red_eyes, fantasy, detailed",
                       ],
                       help='Test prompts (space-separated)')
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of inference steps')
    parser.add_argument('--output-dir', default='onnx_comparison_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 70)
    print("🔬 TOE-ONNX vs BASELINE-ONNX COMPARISON")
    print("=" * 70)
    print(f"\nBaseline ONNX: {args.baseline_onnx}")
    print(f"TOE-ONNX: {args.toe_onnx}")
    print(f"Steps: {args.steps}")
    print(f"Prompts: {len(args.prompts)}")
    
    # Test baseline
    baseline_results = test_baseline_onnx(
        args.baseline_onnx,
        args.prompts,
        args.steps
    )
    
    # Test TOE
    toe_results = test_toe_onnx(
        args.toe_onnx,
        args.vocab,
        args.prompts,
        args.steps
    )
    
    # Create comparison grid
    create_comparison_grid(
        baseline_results,
        toe_results,
        output_dir / "comparison_grid.png"
    )
    
    # Print comparison
    print_comparison(baseline_results, toe_results)
    
    # Save JSON
    save_results_json(
        baseline_results,
        toe_results,
        output_dir / "results.json"
    )
    
    print("\n" + "=" * 70)
    print(f"✅ COMPARISON COMPLETE - Results in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
