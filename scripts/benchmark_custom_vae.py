#!/usr/bin/env python3
"""
Custom TinyVAE Benchmark Script

Tests ONLY the custom trained TinyVAE configurations:
- Baseline G: Hybrid + Custom TinyVAE
- Baseline H: Full Optimization + Custom TinyVAE

Usage:
    python scripts/benchmark_custom_vae.py --num-prompts 2 --output outputs/custom_vae_benchmark
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import statistics

import torch
import psutil
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.optimization import CacheContext, get_global_cache
from hqpd.models.tiny_vae import TinyVAEDecoder


class TinyVAEWrapper:
    """
    Wrapper to make TinyVAEDecoder compatible with diffusers pipeline.
    
    The pipeline expects vae.decode(latents).sample interface.
    """
    
    def __init__(self, decoder: TinyVAEDecoder, scaling_factor: float = 0.13025):
        self.decoder = decoder
        self.scaling_factor = scaling_factor
        # Match expected config interface
        self.config = type('Config', (), {'scaling_factor': scaling_factor})()
    
    def decode(self, latents, return_dict=True):
        """Decode latents to image, matching diffusers VAE interface."""
        # Ensure latents are 4D (B, C, H, W)
        if latents.dim() == 3:
            latents = latents.unsqueeze(0)  # Add batch dimension
        
        # Unscale latents (pipeline passes scaled latents)
        latents_unscaled = latents / self.scaling_factor
        
        # Decode
        decoded = self.decoder(latents_unscaled)
        
        # Return in expected format
        if return_dict:
            return type('VAEOutput', (), {'sample': decoded})()
        return (decoded,)
    
    def to(self, device):
        self.decoder = self.decoder.to(device)
        return self
    
    def eval(self):
        self.decoder.eval()
        return self
    
    @property
    def dtype(self):
        return next(self.decoder.parameters()).dtype
    
    @property
    def device(self):
        return next(self.decoder.parameters()).device


def load_custom_tinyvae(checkpoint_path: str = "checkpoints/tiny_vae/tiny_vae_decoder_epoch10.pt") -> TinyVAEWrapper:
    """
    Load custom trained TinyVAE decoder.
    """
    from pathlib import Path
    
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"TinyVAE checkpoint not found: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Handle both full training checkpoints and raw state dicts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  ✓ Loading from training checkpoint (epoch {checkpoint.get('epoch', '?')})")
    else:
        state_dict = checkpoint
    
    # Create decoder with same config as training
    decoder = TinyVAEDecoder(
        latent_channels=4,
        base_channels=64,
        max_channels=256,
        num_upsample_blocks=3
    )
    decoder.load_state_dict(state_dict)
    decoder.eval()
    
    print(f"  ✓ Loaded custom TinyVAE from: {ckpt_path}")
    
    return TinyVAEWrapper(decoder)


def clear_memory():
    """Aggressively clear memory between benchmark runs."""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    time.sleep(2)


def get_memory_usage() -> float:
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3


def benchmark_configuration(
    config_name: str,
    load_fn: callable,
    prompts: List[str],
    num_steps: int,
    guidance_scale: float,
    warmup: int = 1,
    output_dir: Path = None,
    save_images: bool = True,
) -> Dict[str, Any]:
    """Benchmark a single configuration."""
    print(f"\\n{'='*70}")
    print(f"CONFIGURATION: {config_name}")
    print(f"{'='*70}")
    
    memory_before = get_memory_usage()
    print(f"Memory before loading: {memory_before:.2f} GB")
    
    # Load pipeline
    print("\\nLoading pipeline...")
    start_load = time.time()
    pipeline = load_fn()
    load_time = time.time() - start_load
    
    memory_after_load = get_memory_usage()
    print(f"✓ Pipeline loaded in {load_time:.1f}s")
    print(f"Memory after loading: {memory_after_load:.2f} GB (+{memory_after_load-memory_before:.2f} GB)")
    
    # Warmup
    if warmup > 0:
        print(f"\\n🔥 Warmup ({warmup} runs)...")
        for i in range(warmup):
            _ = pipeline(
                prompt=prompts[0],
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            )
            print(f"  Warmup {i+1}/{warmup} complete")
    
    # Reset cache
    if hasattr(pipeline, '_cache_context'):
        ctx = pipeline._cache_context
        if hasattr(ctx, 'reset_statistics'):
            ctx.reset_statistics()
        elif hasattr(ctx, 'clear'):
            ctx.clear()
    
    # Benchmark runs
    times = []
    memory_peaks = []
    
    # Create config-specific output dir for images
    if save_images and output_dir:
        config_safe_name = config_name.replace(" ", "_").replace(":", "").replace("+", "_")
        images_dir = output_dir / "images" / config_safe_name
        images_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\\n🏃 Running {len(prompts)} benchmark iterations...")
    for i, prompt in enumerate(prompts):
        start = time.time()
        result = pipeline(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        elapsed = time.time() - start
        
        memory_peak = get_memory_usage()
        times.append(elapsed)
        memory_peaks.append(memory_peak)
        
        # Save generated image
        if save_images and output_dir:
            image = result.images[0]
            image_path = images_dir / f"prompt_{i+1}.png"
            image.save(image_path)
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.2f}s | Peak: {memory_peak:.2f} GB | Saved: {image_path.name}")
        else:
            print(f"  [{i+1}/{len(prompts)}] {elapsed:.2f}s | Peak: {memory_peak:.2f} GB")
    
    # Clean up
    del pipeline
    clear_memory()
    
    results = {
        "config_name": config_name,
        "mean_time": statistics.mean(times),
        "memory_peak_avg": statistics.mean(memory_peaks),
    }
    
    print(f"\\nStats: {results['mean_time']:.2f}s avg | {results['memory_peak_avg']:.2f} GB peak")
    return results


def load_baseline_g() -> StableDiffusionXLPipeline:
    """Baseline G: Hybrid + Custom TinyVAE"""
    print("Loading Baseline G: Hybrid + Custom TinyVAE...")
    
    print("  [DEBUG] Loading base pipeline from martineux/janku6...")
    pipeline = StableDiffusionXLPipeline.from_pretrained("martineux/janku6", torch_dtype=torch.float32)
    print("  [DEBUG] Base pipeline loaded.")
    
    # SSD-1B UNet
    print("  [DEBUG] Replacing UNet with SSD-1B...")
    try:
        pipeline.unet = UNet2DConditionModel.from_pretrained("segmind/SSD-1B", subfolder="unet", torch_dtype=torch.float32)
        print("  [DEBUG] SSD-1B UNet loaded directly.")
    except:
        print("  [DEBUG] SSD-1B direct load failed, trying pipeline fallback...")
        temp = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float32)
        pipeline.unet = temp.unet
        del temp
        clear_memory()
        print("  [DEBUG] SSD-1B UNet loaded via fallback.")
    
    # Custom TinyVAE
    print("  [DEBUG] Loading Custom TinyVAE...")
    pipeline.vae = load_custom_tinyvae()
    print("  [DEBUG] Custom TinyVAE loaded.")
    
    # Lightning LoRA
    print("  [DEBUG] Downloading/Loading LoRA...")
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.safetensors")
    pipeline.load_lora_weights(load_file(ckpt))
    print("  [DEBUG] LoRA loaded.")
    
    print("  [DEBUG] Configuring scheduler...")
    pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing", prediction_type="epsilon")
    
    # NO FUSION because it causes crash with custom VAE
    # pipeline.fuse_lora()  <-- DISABLED
    # pipeline.unload_lora_weights()
    
    print("  [DEBUG] Moving pipeline to CPU...")
    return pipeline.to("cpu")


def load_baseline_h() -> StableDiffusionXLPipeline:
    """Baseline H: Full Optimization + Custom TinyVAE"""
    print("Loading Baseline H: Full Optimization + Custom TinyVAE...")
    
    cache = get_global_cache(max_size=256, enabled=True)
    pipeline = load_baseline_g() # Reuse G loading logic
    
    # Attach cache context
    cache_ctx = CacheContext(enabled=True, max_size=256, clear_on_enter=True, print_stats_on_exit=False)
    cache_ctx.__enter__()
    pipeline._cache_context = cache_ctx
    
    print("  ✓ Full optimization + Custom TinyVAE pipeline loaded")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Custom TinyVAE Benchmark")
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--output", type=str, default="outputs/custom_vae_benchmark")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    prompts = [
        "1girl, blue_hair, anime_style, detailed, beautiful_eyes",
        "anime landscape, mountains, sunset, highly_detailed",
        "1boy, fantasy_outfit, action_pose, dynamic, detailed",
        "anime_portrait, close_up, detailed_face, beautiful_lighting",
        "2girls, cafe, sitting, talking, warm_atmosphere, detailed",
    ][:args.num_prompts]
    
    print("="*60)
    print("CUSTOM TINYVAE BENCHMARK (G & H)")
    print("="*60)
    
    results = []
    
    # Baseline G
    try:
        results.append(benchmark_configuration(
            "Baseline G: Hybrid + Custom TinyVAE",
            load_baseline_g, prompts, 4, 0.0, 1, output_dir
        ))
    except Exception as e:
        print(f"Baseline G failed: {e}")
        import traceback
        traceback.print_exc()

    # Baseline H
    try:
        results.append(benchmark_configuration(
            "Baseline H: Full Opt + Custom TinyVAE",
            load_baseline_h, prompts, 4, 0.0, 1, output_dir
        ))
    except Exception as e:
        print(f"Baseline H failed: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\\nAll done. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
