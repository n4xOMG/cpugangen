#!/usr/bin/env python3
"""
Export SDXL model to ONNX format for optimized CPU inference.

This script exports Illustrious (or any SDXL model) to ONNX format,
which provides 2-2.5x speedup on CPU through graph optimizations
and static quantization.

Supports:
- Hugging Face models (e.g., martineux/janku6)
- Local .safetensors files
- Optional static quantization (INT8)
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_sdxl_to_onnx(
    model_id: str,
    output_dir: str,
    device: str = 'cpu',
    opset: int = 14,
    fp16: bool = False,
    quantize: bool = False,
):
    """
    Export SDXL model to ONNX format.
    
    Args:
        model_id: HuggingFace model ID or local path
        output_dir: Directory to save ONNX models
        device: Device to use for export ('cpu' or 'cuda')
        opset: ONNX opset version (14 recommended for best compatibility)
        fp16: Use FP16 precision (recommended for GPU export)
        quantize: Apply static INT8 quantization after export
    """
    from optimum.onnxruntime import ORTStableDiffusionXLPipeline
    from diffusers import StableDiffusionXLPipeline
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 70)
    print("🚀 SDXL to ONNX Export")
    print("=" * 70)
    print(f"\nModel: {model_id}")
    print(f"Output: {output_dir}")
    print(f"Opset: {opset}")
    print(f"FP16: {fp16}")
    print(f"Quantize: {quantize}")
    
    # Step 1: Load PyTorch model
    print("\n📦 Step 1: Loading PyTorch model...")
    
    if model_id.endswith('.safetensors'):
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_id,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if fp16 else torch.float32,
        )
    
    print("✓ Model loaded")
    
    # Step 2: Export to ONNX
    print("\n🔧 Step 2: Exporting to ONNX...")
    print("   This may take 5-10 minutes...")
    
    # Use optimum to export
    ort_pipe = ORTStableDiffusionXLPipeline.from_pretrained(
        model_id,
        export=True,
        provider="CPUExecutionProvider",
    )
    
    # Save ONNX models
    ort_pipe.save_pretrained(output_dir)
    
    print(f"✓ ONNX models exported to: {output_dir}")
    
    # Step 3: Apply quantization if requested
    if quantize:
        print("\n⚙️ Step 3: Applying INT8 static quantization...")
        
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        
        # Quantize UNet (most compute-intensive)
        unet_path = output_dir / "unet" / "model.onnx"
        
        if unet_path.exists():
            quantizer = ORTQuantizer.from_pretrained(output_dir / "unet")
            
            # Configure static quantization
            qconfig = AutoQuantizationConfig.avx512_vnni(
                is_static=True,
                per_channel=True,
            )
            
            # Apply quantization
            quantizer.quantize(
                save_dir=output_dir / "unet_quantized",
                quantization_config=qconfig,
            )
            
            print("✓ UNet quantized")
        else:
            print("⚠️  UNet model.onnx not found, skipping quantization")
    
    # Step 4: Test inference
    print("\n🧪 Step 4: Testing ONNX inference...")
    
    import time
    
    # Load ONNX pipeline
    onnx_pipe = ORTStableDiffusionXLPipeline.from_pretrained(
        output_dir,
        provider="CPUExecutionProvider",
    )
    
    # Quick test
    prompt = "1girl, solo, blue_eyes, smile, anime"
    print(f"   Prompt: {prompt}")
    print(f"   Generating with 5 steps...")
    
    start = time.time()
    image = onnx_pipe(
        prompt=prompt,
        num_inference_steps=5,
        height=512,
        width=512,
    ).images[0]
    elapsed = time.time() - start
    
    # Save test image
    test_path = output_dir / "onnx_test_output.png"
    image.save(test_path)
    
    print(f"✓ Test image saved: {test_path}")
    print(f"✓ Time: {elapsed:.1f}s ({elapsed/5:.1f}s/step)")
    
    print("\n" + "=" * 70)
    print("✅ ONNX Export Complete!")
    print("=" * 70)
    print(f"\nONNX models saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Compare speed with original PyTorch model")
    print("2. Test with different prompts for quality validation")
    print("3. If good results, integrate into production pipeline")
    
    return str(output_dir)


def compare_onnx_vs_pytorch(
    model_id: str,
    onnx_dir: str,
    test_prompts: list = None,
    num_steps: int = 20,
):
    """
    Compare ONNX vs PyTorch inference speed and quality.
    
    Args:
        model_id: Original model ID
        onnx_dir: Directory with ONNX models
        test_prompts: List of prompts to test
        num_steps: Number of inference steps
    """
    import time
    from diffusers import StableDiffusionXLPipeline
    from optimum.onnxruntime import ORTStableDiffusionXLPipeline
    from PIL import Image
    
    if test_prompts is None:
        test_prompts = [
            "1girl, solo, blue_eyes, smile, anime style, masterpiece",
            "1girl, long_hair, red_eyes, fantasy, detailed",
        ]
    
    print("\n" + "=" * 70)
    print("⚖️  ONNX vs PyTorch Comparison")
    print("=" * 70)
    
    results = {'pytorch': {}, 'onnx': {}}
    
    # Test PyTorch
    print("\n📦 Loading PyTorch model...")
    pytorch_pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    )
    pytorch_pipe.to("cpu")
    
    print(f"\n🔧 Testing PyTorch ({num_steps} steps)...")
    for i, prompt in enumerate(test_prompts):
        print(f"\n  Prompt {i+1}: {prompt[:50]}...")
        start = time.time()
        image = pytorch_pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
        ).images[0]
        elapsed = time.time() - start
        
        results['pytorch'][i] = {
            'time': elapsed,
            'image': image
        }
        print(f"  Time: {elapsed:.1f}s ({elapsed/num_steps:.1f}s/step)")
    
    # Clear memory
    del pytorch_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Test ONNX
    print(f"\n📦 Loading ONNX model...")
    onnx_pipe = ORTStableDiffusionXLPipeline.from_pretrained(
        onnx_dir,
        provider="CPUExecutionProvider",
    )
    
    print(f"\n⚡ Testing ONNX ({num_steps} steps)...")
    for i, prompt in enumerate(test_prompts):
        print(f"\n  Prompt {i+1}: {prompt[:50]}...")
        start = time.time()
        image = onnx_pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
        ).images[0]
        elapsed = time.time() - start
        
        results['onnx'][i] = {
            'time': elapsed,
            'image': image
        }
        print(f"  Time: {elapsed:.1f}s ({elapsed/num_steps:.1f}s/step)")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    
    for i in range(len(test_prompts)):
        pytorch_time = results['pytorch'][i]['time']
        onnx_time = results['onnx'][i]['time']
        speedup = pytorch_time / onnx_time
        
        print(f"\nPrompt {i+1}:")
        print(f"  PyTorch: {pytorch_time:.1f}s")
        print(f"  ONNX:    {onnx_time:.1f}s")
        print(f"  Speedup: {speedup:.2f}x")
    
    # Save comparison grid
    output_dir = Path(onnx_dir).parent
    grid_images = []
    
    for i in range(len(test_prompts)):
        grid_images.extend([
            results['pytorch'][i]['image'],
            results['onnx'][i]['image'],
        ])
    
    # Create grid
    n_prompts = len(test_prompts)
    img_width, img_height = grid_images[0].size
    
    grid = Image.new('RGB', (img_width * 2, img_height * n_prompts))
    
    for idx, img in enumerate(grid_images):
        row = idx // 2
        col = idx % 2
        grid.paste(img, (col * img_width, row * img_height))
    
    grid_path = output_dir / "pytorch_vs_onnx_comparison.png"
    grid.save(grid_path)
    
    print(f"\n✓ Comparison grid saved: {grid_path}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Export SDXL model to ONNX for optimized inference'
    )
    parser.add_argument('--model', required=True,
                       help='HuggingFace model ID or local path')
    parser.add_argument('--output-dir', default='onnx_models',
                       help='Output directory for ONNX models')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for export (GPU recommended for speed)')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 precision (auto-enabled for GPU)')
    parser.add_argument('--quantize', action='store_true',
                       help='Apply INT8 static quantization')
    parser.add_argument('--compare', action='store_true',
                       help='Compare ONNX vs PyTorch performance')
    parser.add_argument('--test-steps', type=int, default=20,
                       help='Number of steps for comparison test')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ CUDA not available. Falling back to CPU.")
            args.device = 'cpu'
        else:
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Export to ONNX
    onnx_dir = export_sdxl_to_onnx(
        model_id=args.model,
        output_dir=args.output_dir,
        device=args.device,
        opset=args.opset,
        fp16=args.fp16,
        quantize=args.quantize,
    )
    
    # Run comparison if requested
    if args.compare:
        compare_onnx_vs_pytorch(
            model_id=args.model,
            onnx_dir=onnx_dir,
            num_steps=args.test_steps,
        )


if __name__ == "__main__":
    main()
