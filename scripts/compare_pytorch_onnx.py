#!/usr/bin/env python3
"""
Compare PyTorch vs ONNX SDXL-Lightning side-by-side.
Generates images with both, compares quality and performance.
"""

import argparse
import time
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def calculate_image_diff(img1_path, img2_path):
    """Calculate pixel-wise difference between two images."""
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    
    # Ensure same size
    if img1.size != img2.size:
        print(f"⚠️  Warning: Images have different sizes: {img1.size} vs {img2.size}")
        img2 = img2.resize(img1.size)
    
    # Convert to numpy
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    
    # Calculate differences
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Calculate percentage of pixels with differences
    threshold = 1.0  # Ignore tiny differences
    changed_pixels = np.sum(diff > threshold) / diff.size * 100
    
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "changed_pixels_pct": changed_pixels,
    }


def create_comparison_image(pytorch_path, onnx_path, output_path, stats):
    """Create a side-by-side comparison image."""
    img1 = Image.open(pytorch_path)
    img2 = Image.open(onnx_path)
    
    # Create side-by-side canvas
    width = img1.width + img2.width + 60  # 60px for labels
    height = max(img1.height, img2.height) + 100
    
    canvas = Image.new("RGB", (width, height), color=(20, 20, 20))
    
    # Paste images
    canvas.paste(img1, (30, 80))
    canvas.paste(img2, (img1.width + 60, 80))
    
    # Add labels
    draw = ImageDraw.Draw(canvas)
    try:
        font_large = ImageFont.truetype("arial.ttf", 32)
        font_small = ImageFont.truetype("arial.ttf", 20)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Title
    draw.text((width // 2 - 200, 20), "PyTorch vs ONNX Comparison", 
              fill=(255, 255, 255), font=font_large)
    
    # Labels
    draw.text((30 + img1.width // 2 - 50, 50), "PyTorch", 
              fill=(100, 200, 255), font=font_small)
    draw.text((img1.width + 60 + img2.width // 2 - 40, 50), "ONNX", 
              fill=(255, 200, 100), font=font_small)
    
    # Stats
    stats_y = 80 + max(img1.height, img2.height) + 10
    stats_text = (
        f"Max pixel diff: {stats['max_diff']:.2f} | "
        f"Mean diff: {stats['mean_diff']:.2f} | "
        f"Changed pixels: {stats['changed_pixels_pct']:.2f}%"
    )
    draw.text((30, stats_y), stats_text, fill=(200, 200, 200), font=font_small)
    
    canvas.save(output_path)
    print(f"💾 Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch vs ONNX SDXL-Lightning")
    parser.add_argument("--pytorch-model", default="martineux/janku6", help="PyTorch base model")
    parser.add_argument("--onnx-model", default="onnx_models/sdxl_lightning_optimum/onnx", 
                        help="ONNX model directory")
    parser.add_argument("--steps", type=int, default=4, help="Inference steps")
    parser.add_argument("--prompt", default="anime girl with blue hair, beautiful eyes, highly detailed",
                        help="Generation prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--benchmark-runs", type=int, default=5, 
                        help="Number of runs for benchmarking")
    parser.add_argument("--output", default="outputs/comparison", help="Output directory")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 70)
    print("🔬 PyTorch vs ONNX Comparison & Benchmark")
    print("=" * 70)
    
    # Run PyTorch generation
    print("\n" + "=" * 70)
    print("🔥 Running PyTorch inference...")
    print("=" * 70)
    
    pytorch_output = output_dir / "pytorch"
    pytorch_cmd = [
        sys.executable, "scripts/test_pytorch.py",
        "--model", args.pytorch_model,
        "--steps", str(args.steps),
        "--prompt", args.prompt,
        "--seed", str(args.seed),
        "--output", str(pytorch_output),
        "--num-images", str(args.benchmark_runs),
        "--benchmark",
    ]
    
    pytorch_start = time.time()
    subprocess.run(pytorch_cmd, check=True)
    pytorch_total = time.time() - pytorch_start
    
    # Run ONNX generation
    print("\n" + "=" * 70)
    print("⚡ Running ONNX inference...")
    print("=" * 70)
    
    onnx_output = output_dir / "onnx"
    onnx_cmd = [
        sys.executable, "scripts/test_onnx.py",
        "--model", args.onnx_model,
        "--steps", str(args.steps),
        "--prompt", args.prompt,
        "--seed", str(args.seed),
        "--output", str(onnx_output),
        "--num-images", str(args.benchmark_runs),
        "--benchmark",
    ]
    
    onnx_start = time.time()
    subprocess.run(onnx_cmd, check=True)
    onnx_total = time.time() - onnx_start
    
    # Read benchmark results
    print("\n" + "=" * 70)
    print("📊 Comparing Results...")
    print("=" * 70)
    
    def parse_benchmark(path):
        with open(path) as f:
            lines = f.readlines()
            for line in lines:
                if "Average time:" in line:
                    avg_time = float(line.split(":")[1].strip().replace("s", ""))
                    return avg_time
        return None
    
    pytorch_avg = parse_benchmark(pytorch_output / "pytorch_benchmark.txt")
    onnx_avg = parse_benchmark(onnx_output / "onnx_benchmark.txt")
    
    # Image quality comparison (first image only)
    print("\n📸 Image Quality Analysis (same seed)...")
    pytorch_img = pytorch_output / f"pytorch_{args.seed}.png"
    onnx_img = onnx_output / f"onnx_{args.seed}.png"
    
    if pytorch_img.exists() and onnx_img.exists():
        stats = calculate_image_diff(pytorch_img, onnx_img)
        
        print(f"\n   Max pixel difference:  {stats['max_diff']:.2f} / 255")
        print(f"   Mean pixel difference: {stats['mean_diff']:.2f} / 255")
        print(f"   Changed pixels:        {stats['changed_pixels_pct']:.2f}%")
        
        # Create comparison image
        comparison_path = output_dir / "comparison.png"
        create_comparison_image(pytorch_img, onnx_img, comparison_path, stats)
    else:
        print("\n⚠️  Could not find images for comparison")
        stats = None
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 FINAL COMPARISON SUMMARY")
    print("=" * 70)
    
    summary = []
    summary.append(f"\n{'Metric':<30} {'PyTorch':<20} {'ONNX':<20} {'Diff':<15}")
    summary.append("=" * 85)
    
    if pytorch_avg and onnx_avg:
        speedup = pytorch_avg / onnx_avg
        summary.append(f"{'Average inference time':<30} {pytorch_avg:.2f}s{'':<14} {onnx_avg:.2f}s{'':<14} {speedup:.2f}x")
        summary.append(f"{'Throughput':<30} {1/pytorch_avg:.2f} img/s{'':<10} {1/onnx_avg:.2f} img/s{'':<10}")
    
    if stats:
        summary.append(f"{'Max pixel difference':<30} {'-':<20} {'-':<20} {stats['max_diff']:.2f}/255")
        summary.append(f"{'Mean pixel difference':<30} {'-':<20} {'-':<20} {stats['mean_diff']:.2f}/255")
        summary.append(f"{'Changed pixels':<30} {'-':<20} {'-':<20} {stats['changed_pixels_pct']:.2f}%")
    
    for line in summary:
        print(line)
    
    # Verdict
    print("\n" + "=" * 70)
    print("🎯 VERDICT")
    print("=" * 70)
    
    if stats and stats['mean_diff'] < 5.0:
        print("✅ Image quality: IDENTICAL (mean diff < 5/255)")
    elif stats and stats['mean_diff'] < 10.0:
        print("✅ Image quality: VIRTUALLY IDENTICAL (mean diff < 10/255)")
    elif stats:
        print(f"⚠️  Image quality: NOTICEABLE DIFFERENCE (mean diff = {stats['mean_diff']:.2f}/255)")
    
    if pytorch_avg and onnx_avg:
        if onnx_avg < pytorch_avg:
            speedup = pytorch_avg / onnx_avg
            print(f"⚡ Performance: ONNX is {speedup:.2f}x FASTER")
        else:
            slowdown = onnx_avg / pytorch_avg
            print(f"🐌 Performance: ONNX is {slowdown:.2f}x SLOWER")
    
    # Save summary
    summary_path = output_dir / "comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write("PyTorch vs ONNX Comparison Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write(f"Steps: {args.steps}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Benchmark runs: {args.benchmark_runs}\n\n")
        for line in summary:
            f.write(line + "\n")
    
    print(f"\n💾 Summary saved to: {summary_path}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
