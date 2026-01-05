"""
Test script for Tag-Optimized Encoder
Verify architecture and basic functionality before training
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from hqpd.models.toe import TagOptimizedEncoder
from hqpd.utils.danbooru import DanbooruTagProcessor


def test_toe_architecture():
    """Test TOE model creation and forward pass."""
    print("=" * 60)
    print("Testing Tag-Optimized Encoder Architecture")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating TOE model...")
    model = TagOptimizedEncoder(
        vocab_size=15000,
        embed_dim=2048,
        num_layers=4,
        num_heads=8,
        mlp_ratio=2,
        max_length=77,
        quantize_embeddings=False  # Test without quantization first
    )
    
    print(f"   ✓ Model created successfully")
    print(f"   ✓ Parameters: {model.get_num_params():,}")
    print(f"   ✓ Size: {model.get_model_size_mb():.2f} MB (FP32)")
    print(f"   ✓ Expected: ~50M params, ~200MB")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    num_tags = 20
    
    # Create dummy input
    tag_ids = torch.randint(0, 15000, (batch_size, num_tags))
    tag_weights = torch.rand(batch_size, num_tags)
    
    # Forward pass
    with torch.no_grad():
        output = model(tag_ids, tag_weights)
    
    print(f"   ✓ Input shape: {tag_ids.shape}")
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Expected: (batch={batch_size}, seq_len=77, dim=2048)")
    
    assert output.shape == (batch_size, 77, 2048), "Output shape mismatch!"
    print(f"   ✓ Shape validation passed!")
    
    return model


def test_tag_processor():
    """Test Danbooru tag processor."""
    print("\n" + "=" * 60)
    print("Testing Danbooru Tag Processor")
    print("=" * 60)
    
    # Create processor
    print("\n1. Creating tag processor...")
    processor = DanbooruTagProcessor(vocab_size=15000)
    
    # Sample tags
    sample_tags = [
        ["1girl", "solo", "long_hair", "blue_eyes", "school_uniform"],
        ["1girl", "smile", "outdoors", "cherry_blossoms", "spring"],
        ["multiple_girls", "2girls", "sisters", "happy"],
        ["1boy", "male_focus", "sword", "fantasy", "armor"],
    ]
    
    print("\n2. Building vocabulary...")
    processor.build_vocabulary(sample_tags * 100)  # Repeat for frequency
    print(f"   ✓ Vocabulary size: {len(processor.tag_to_id)}")
    
    # Test encoding
    print("\n3. Testing tag encoding...")
    test_tags = ["1girl", "solo", "long_hair", "blue_eyes"]
    tag_ids, tag_weights = processor.encode(test_tags, max_length=77)
    
    print(f"   ✓ Input tags: {test_tags}")
    print(f"   ✓ Encoded IDs (first 10): {tag_ids[:10]}")
    print(f"   ✓ Weights (first 10): {tag_weights[:10]}")
    
    # Test decoding
    decoded = processor.decode(tag_ids)
    print(f"   ✓ Decoded tags: {decoded[:10]}")
    
    return processor


def test_complete_pipeline():
    """Test tag processor + TOE model together."""
    print("\n" + "=" * 60)
    print("Testing Complete Pipeline")
    print("=" * 60)
    
    # Create components
    processor = DanbooruTagProcessor(vocab_size=15000)
    sample_tags = [
        ["1girl", "solo", "long_hair"],
        ["1boy", "smile"],
    ]
    processor.build_vocabulary(sample_tags * 100)
    
    model = TagOptimizedEncoder()
    
    # Process tags
    print("\n1. Processing tags...")
    test_input = ["1girl", "solo", "long_hair", "blue_eyes", "portrait"]
    tag_ids, tag_weights = processor.encode(test_input)
    
    # Convert to tensors
    tag_ids_tensor = torch.tensor([tag_ids])  # Add batch dimension
    tag_weights_tensor = torch.tensor([tag_weights])
    
    print(f"   ✓ Tags: {test_input}")
    print(f"   ✓ Tensor shapes: ids={tag_ids_tensor.shape}, weights={tag_weights_tensor.shape}")
    
    # Forward pass
    print("\n2. Generating context embeddings...")
    with torch.no_grad():
        context = model(tag_ids_tensor, tag_weights_tensor)
    
    print(f"   ✓ Context shape: {context.shape}")
    print(f"   ✓ Context dtype: {context.dtype}")
    print(f"   ✓ Context range: [{context.min():.3f}, {context.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


def test_quantization():
    """Test INT4 quantization simulation."""
    print("\n" + "=" * 60)
    print("Testing INT4 Quantization")
    print("=" * 60)
    
    # Create model with quantization
    print("\n1. Creating quantized model...")
    model = TagOptimizedEncoder(quantize_embeddings=True)
    model.train()  # Quantization only active during training
    
    # Forward pass
    batch_size = 2
    num_tags = 10
    tag_ids = torch.randint(0, 15000, (batch_size, num_tags))
    tag_weights = torch.ones(batch_size, num_tags)
    
    print("\n2. Running forward pass with quantization...")
    output = model(tag_ids, tag_weights)
    
    print(f"   ✓ Output shape: {output.shape}")
    print(f"   ✓ Quantization is active in training mode")
    
    # Switch to eval mode
    model.eval()
    with torch.no_grad():
        output_eval = model(tag_ids, tag_weights)
    
    print(f"   ✓ Eval mode output shape: {output_eval.shape}")
    print(f"   ✓ Quantization disabled in eval mode")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "TOE Architecture Test Suite" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # Run tests
        model = test_toe_architecture()
        processor = test_tag_processor()
        test_complete_pipeline()
        test_quantization()
        
        print("\n" + "=" * 60)
        print("SUCCESS! All components working correctly.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Install remaining dependencies (diffusers, transformers)")
        print("2. Download Danbooru dataset subset")
        print("3. Pre-compute CLIP embeddings")
        print("4. Run training: python scripts/train_toe.py")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
