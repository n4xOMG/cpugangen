#!/usr/bin/env python3
"""
Export SDXL-Lightning to ONNX using optimum-cli (the correct way).

This script prepares the Lightning-fused model and then uses optimum-cli
to export it properly with all the correct output names.
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="martineux/janku6")
    parser.add_argument("--output-dir", default="onnx_models/sdxl_lightning_optimum")
    parser.add_argument("--steps", type=int, default=4, choices=[2, 4, 8])
    args = parser.parse_args()
    
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    
    output_dir = Path(args.output_dir)
    temp_dir = output_dir / "temp_pytorch"
    onnx_dir = output_dir / "onnx"
    
    print("\n" + "=" * 60)
    print("🚀 SDXL-Lightning → ONNX (via optimum-cli)")
    print("=" * 60)
    
    # Step 1: Load and fuse Lightning
    print("\n📦 Step 1: Loading base model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    print(f"\n📦 Step 2: Loading Lightning {args.steps}-step...")
    ckpt_map = {2: "sdxl_lightning_2step_lora.safetensors",
                4: "sdxl_lightning_4step_lora.safetensors",
                8: "sdxl_lightning_8step_lora.safetensors"}
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", ckpt_map[args.steps])
    pipe.load_lora_weights(load_file(ckpt))
    
    print("\n🔧 Step 3: Configuring scheduler...")
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )
    
    print("\n🔧 Step 4: Fusing LoRA weights...")
    pipe.to(device)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    
    # Save to temp directory
    print("\n💾 Step 5: Saving fused model...")
    temp_dir.mkdir(exist_ok=True, parents=True)
    pipe.to("cpu")
    pipe.save_pretrained(temp_dir, safe_serialization=True)
    print(f"   Saved to: {temp_dir}")
    
    # Use optimum-cli to export
    print("\n🔧 Step 6: Exporting with optimum-cli...")
    print("   This may take 10-20 minutes...")
    
    onnx_dir.mkdir(exist_ok=True, parents=True)
    
    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", str(temp_dir),
        "--task", "stable-diffusion-xl",
        "--no-post-process",  # Skip validation to avoid tolerance errors
        str(onnx_dir),
    ]
    
    print(f"   Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("\n❌ Export failed!")
        print("   Try running manually:")
        print(f"   optimum-cli export onnx --model {temp_dir} --task stable-diffusion-xl {onnx_dir}")
        return
    
    # Cleanup
    print("\n🧹 Step 7: Cleaning up temp files...")
    shutil.rmtree(temp_dir)
    
    print("\n" + "=" * 60)
    print("✅ EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nONNX models: {onnx_dir}")
    print("\nTest with:")
    print(f"  python scripts/test_onnx.py --model {onnx_dir}")


if __name__ == "__main__":
    main()
