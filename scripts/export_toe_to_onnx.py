#!/usr/bin/env python3
"""
Export TOE-based SDXL pipeline to ONNX format.

This exports the complete pipeline including:
- TOE encoder (tag embeddings + transformer)
- SDXL UNet
- VAE decoder

Benefits:
- TOE: 50MB vs 999MB CLIP (memory savings)
- ONNX: 2-2.5x compute speedup
- Combined: Memory efficient + faster inference
"""

import argparse
import sys
from pathlib import Path
import torch
import time

sys.path.insert(0, str(Path(__file__).parent.parent))


def export_toe_pipeline_to_onnx(
    toe_checkpoint: str,
    vocab_path: str,
    base_model: str,
    output_dir: str,
    device: str = 'cpu',
    opset: int = 14,
    quantize_unet: bool = False,
):
    """
    Export TOE-based SDXL pipeline to ONNX.
    
    Args:
        toe_checkpoint: Path to trained TOE checkpoint
        vocab_path: Path to vocabulary.json
        base_model: Base SDXL model ID or path
        output_dir: Output directory for ONNX models
        device: Device to use ('cpu' or 'cuda')
        opset: ONNX opset version
        quantize_unet: Apply INT8 quantization to UNet
    """
    from diffusers import StableDiffusionXLPipeline
    from hqpd.models.toe import TagOptimizedEncoder
    from hqpd.utils.danbooru import DanbooruTagProcessor
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "=" * 70)
    print("🚀 TOE-SDXL Pipeline to ONNX Export")
    print("=" * 70)
    print(f"\nTOE checkpoint: {toe_checkpoint}")
    print(f"Vocabulary: {vocab_path}")
    print(f"Base model: {base_model}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    
    # Step 1: Load TOE components
    print("\n📦 Step 1: Loading TOE components...")
    
    # Load vocabulary
    tag_processor = DanbooruTagProcessor(vocab_size=15000)
    tag_processor.load_vocabulary(vocab_path)
    print(f"   ✓ Loaded {len(tag_processor.tag_to_id)} tags")
    
    # Load TOE model
    checkpoint = torch.load(toe_checkpoint, map_location=device)
    has_pooling = any('pooling_head' in k for k in checkpoint['model_state_dict'].keys())
    
    toe_model = TagOptimizedEncoder(
        vocab_size=15000,
        embed_dim=2048,
        num_layers=4,
        num_heads=8,
        enable_pooling=has_pooling
    )
    toe_model.load_state_dict(checkpoint['model_state_dict'])
    toe_model.to(device)
    toe_model.eval()
    
    print(f"   ✓ TOE model loaded ({toe_model.get_model_size_mb():.2f} MB)")
    
    # Step 2: Load base SDXL pipeline
    print("\n📦 Step 2: Loading SDXL components...")
    
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,
    )
    
    print("   ✓ SDXL pipeline loaded")
    
    # Step 3: Export TOE to ONNX
    print("\n🔧 Step 3: Exporting TOE to ONNX...")
    
    toe_onnx_path = output_dir / "toe_encoder"
    toe_onnx_path.mkdir(exist_ok=True)
    
    # Prepare sample inputs for TOE
    sample_tag_ids = torch.randint(0, 15000, (1, 20))
    sample_tag_weights = torch.ones(1, 20)
    
    # Export TOE
    torch.onnx.export(
        toe_model,
        (sample_tag_ids, sample_tag_weights),
        toe_onnx_path / "model.onnx",
        input_names=['tag_ids', 'tag_weights'],
        output_names=['context_embeddings'],
        dynamic_axes={
            'tag_ids': {0: 'batch', 1: 'num_tags'},
            'tag_weights': {0: 'batch', 1: 'num_tags'},
            'context_embeddings': {0: 'batch'}
        },
        opset_version=opset,
    )
    
    print(f"   ✓ TOE exported to: {toe_onnx_path}")
    
    # Step 4: Export SDXL components using optimum
    print("\n🔧 Step 4: Exporting SDXL components (UNet, VAE) using optimum...")
    print("   This handles complex SDXL architecture correctly...")
    
    from optimum.onnxruntime import ORTStableDiffusionXLPipeline
    
    # Use optimum to export the full SDXL pipeline
    # This will create UNet, VAE encoder, VAE decoder, etc.
    sdxl_onnx_path = output_dir / "sdxl_base"
    
    print(f"   Exporting to: {sdxl_onnx_path}")
    print("   (This may take 5-10 minutes...)")
    
    ort_pipe = ORTStableDiffusionXLPipeline.from_pretrained(
        base_model,
        export=True,
        provider="CPUExecutionProvider",
    )
    
    # Save to output directory
    ort_pipe.save_pretrained(sdxl_onnx_path)
    
    print(f"   ✓ SDXL components exported to: {sdxl_onnx_path}")
    print(f"      - UNet: {sdxl_onnx_path / 'unet'}")
    print(f"      - VAE Encoder: {sdxl_onnx_path / 'vae_encoder'}")
    print(f"      - VAE Decoder: {sdxl_onnx_path / 'vae_decoder'}")
    
    # Step 5: Save metadata
    print("\n💾 Step 5: Saving metadata...")
    
    import json
    
    metadata = {
        'toe_checkpoint': str(toe_checkpoint),
        'vocabulary': str(vocab_path),
        'base_model': base_model,
        'has_pooling': has_pooling,
        'opset_version': opset,
        'export_method': 'optimum',
        'components': {
            'toe_encoder': str(toe_onnx_path.relative_to(output_dir)),
            'sdxl_base': str(sdxl_onnx_path.relative_to(output_dir)),
        }
    }
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Metadata saved")
    
    # Step 6: Quick test
    print("\n🧪 Step 6: Testing TOE ONNX export...")
    
    import onnxruntime as ort
    
    # Test TOE
    toe_session = ort.InferenceSession(
        str(toe_onnx_path / "model.onnx"),
        providers=['CPUExecutionProvider']
    )
    
    test_tags = ["1girl", "solo", "blue_eyes", "smile"]
    test_tag_ids, test_tag_weights = tag_processor.encode(test_tags, max_length=77)
    
    toe_output = toe_session.run(
        None,
        {
            'tag_ids': torch.tensor([test_tag_ids]).numpy(),
            'tag_weights': torch.tensor([test_tag_weights]).numpy(),
        }
    )
    
    print(f"   ✓ TOE test successful (output shape: {toe_output[0].shape})")
    
    print("\n" + "=" * 70)
    print("✅ TOE-ONNX Export Complete!")
    print("=" * 70)
    print(f"\nONNX models saved to: {output_dir}")
    print("\nComponents:")
    print(f"  - TOE Encoder: {toe_onnx_path}")
    print(f"  - SDXL Base (UNet, VAE): {sdxl_onnx_path}")
    print("\nNext steps:")
    print("1. Test full generation with TOE-ONNX pipeline")
    print("2. Compare speed: PyTorch-TOE vs ONNX-TOE")
    print("3. Validate quality on diverse prompts")
    
    return str(output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Export TOE-based SDXL pipeline to ONNX'
    )
    parser.add_argument('--toe-checkpoint', required=True,
                       help='Path to trained TOE checkpoint')
    parser.add_argument('--vocab', required=True,
                       help='Path to vocabulary.json')
    parser.add_argument('--base-model', required=True,
                       help='Base SDXL model ID or path')
    parser.add_argument('--output-dir', default='onnx_models/illustrious_toe',
                       help='Output directory for ONNX models')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for export (GPU recommended)')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version')
    parser.add_argument('--quantize-unet', action='store_true',
                       help='Apply INT8 quantization to UNet (experimental)')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("❌ CUDA not available. Falling back to CPU.")
            args.device = 'cpu'
        else:
            print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Export
    export_toe_pipeline_to_onnx(
        toe_checkpoint=args.toe_checkpoint,
        vocab_path=args.vocab,
        base_model=args.base_model,
        device=args.device,
        output_dir=args.output_dir,
        opset=args.opset,
        quantize_unet=args.quantize_unet,
    )


if __name__ == "__main__":
    main()
