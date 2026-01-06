#!/usr/bin/env python3
"""
Test SDXL-Lightning with PyTorch (baseline for comparison).
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Generate images with PyTorch SDXL-Lightning")
    parser.add_argument("--model", default="martineux/janku6", help="Base model to use")
    parser.add_argument("--steps", type=int, default=4, choices=[2, 4, 8], help="Lightning steps")
    parser.add_argument("--prompt", default="anime girl with blue hair, beautiful eyes, highly detailed", 
                        help="Generation prompt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", default="outputs/pytorch", help="Output directory")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark (multiple iterations)")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 60)
    print("🔥 PyTorch SDXL-Lightning Inference")
    print("=" * 60)
    
    # Load model
    print(f"\n📦 Loading model: {args.model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    load_start = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
    )
    
    # Load Lightning LoRA
    print(f"📦 Loading Lightning {args.steps}-step LoRA...")
    ckpt_map = {
        2: "sdxl_lightning_2step_lora.safetensors",
        4: "sdxl_lightning_4step_lora.safetensors",
        8: "sdxl_lightning_8step_lora.safetensors"
    }
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", ckpt_map[args.steps])
    pipe.load_lora_weights(load_file(ckpt))
    
    # Configure scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    # Fuse and move to device
    pipe.to(device)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f}s")
    print(f"🎮 Using device: {device}")
    
    # Generate images
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    if args.benchmark:
        print(f"\n⏱️  Running benchmark ({args.num_images} iterations)...")
        times = []
        
        # Warmup
        print("🔥 Warmup run...")
        _ = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=generator,
        )
        
        # Benchmark runs
        for i in range(args.num_images):
            print(f"\n🎨 Generation {i+1}/{args.num_images}")
            generator = torch.Generator(device=device).manual_seed(args.seed + i)
            
            start = time.time()
            result = pipe(
                prompt=args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=0.0,
                generator=generator,
            )
            elapsed = time.time() - start
            times.append(elapsed)
            
            # Save image
            image = result.images[0]
            output_path = output_dir / f"pytorch_bench_{i+1}.png"
            image.save(output_path)
            print(f"   Time: {elapsed:.2f}s")
            print(f"   Saved: {output_path}")
        
        # Statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Average time: {avg_time:.2f}s")
        print(f"Min time:     {min_time:.2f}s")
        print(f"Max time:     {max_time:.2f}s")
        print(f"Throughput:   {1/avg_time:.2f} images/sec")
        
        # Save benchmark results
        results_path = output_dir / "pytorch_benchmark.txt"
        with open(results_path, "w") as f:
            f.write("PyTorch SDXL-Lightning Benchmark\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Steps: {args.steps}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Dtype: {dtype}\n")
            f.write(f"Iterations: {args.num_images}\n\n")
            f.write(f"Load time:    {load_time:.2f}s\n")
            f.write(f"Average time: {avg_time:.2f}s\n")
            f.write(f"Min time:     {min_time:.2f}s\n")
            f.write(f"Max time:     {max_time:.2f}s\n")
            f.write(f"Throughput:   {1/avg_time:.2f} images/sec\n\n")
            f.write("Individual times:\n")
            for i, t in enumerate(times, 1):
                f.write(f"  {i}: {t:.2f}s\n")
        
        print(f"\n💾 Results saved to: {results_path}")
        
    else:
        # Single generation
        print(f"\n🎨 Generating {args.num_images} image(s)...")
        print(f"   Prompt: {args.prompt}")
        print(f"   Steps: {args.steps}")
        print(f"   Seed: {args.seed}")
        
        for i in range(args.num_images):
            generator = torch.Generator(device=device).manual_seed(args.seed + i)
            
            start = time.time()
            result = pipe(
                prompt=args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=0.0,
                generator=generator,
            )
            elapsed = time.time() - start
            
            image = result.images[0]
            output_path = output_dir / f"pytorch_{args.seed + i}.png"
            image.save(output_path)
            
            print(f"\n✅ Image {i+1} generated in {elapsed:.2f}s")
            print(f"   Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
