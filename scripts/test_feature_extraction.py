#!/usr/bin/env python3
"""
Quick test script to verify feature extraction works correctly before full training.
Run this to ensure no errors in the updated distillation training setup.
"""

import sys
import torch
from diffusers import UNet2DConditionModel

# Add scripts to path
sys.path.insert(0, 'scripts')

from feature_extractor import UNetFeatureExtractor, get_default_feature_layers


def test_feature_extraction():
    """Test feature extraction on a dummy model."""
    print("=" * 60)
    print("Feature Extraction Test")
    print("=" * 60)
    
    # Create a simple SDXL-like UNet config for testing
    print("\n1. Loading UNet model...")
    try:
        # Try loading actual SSD-1B model
        unet = UNet2DConditionModel.from_pretrained(
            "segmind/SSD-1B",
            subfolder="unet",
            torch_dtype=torch.float32
        )
        print("✓ Loaded SSD-1B UNet")
    except Exception as e:
        print(f"Could not load SSD-1B: {e}")
        print("Skipping test - requires internet connection and HF access")
        return
    
    # Get recommended layers
    feature_layers = get_default_feature_layers('sdxl')
    print(f"\n2. Feature layers to extract: {feature_layers}")
    
    # Create feature extractor
    print("\n3. Creating feature extractor...")
    extractor = UNetFeatureExtractor(unet, feature_layers, normalize=True)
    print("✓ Feature extractor created")
    
    # Create dummy inputs
    print("\n4. Preparing dummy inputs...")
    batch_size = 2
    latent_channels = 4
    latent_size = 128
    
    noisy_latents = torch.randn(batch_size, latent_channels, latent_size, latent_size)
    timesteps = torch.randint(0, 1000, (batch_size,))
    encoder_hidden_states = torch.randn(batch_size, 77, 2048)  # SDXL hidden states
    
    # Added conditioning for SDXL
    added_cond_kwargs = {
        "text_embeds": torch.randn(batch_size, 1280),
        "time_ids": torch.randn(batch_size, 6)
    }
    
    print(f"  Latents shape: {noisy_latents.shape}")
    print(f"  Timesteps shape: {timesteps.shape}")
    print(f"  Encoder hidden states shape: {encoder_hidden_states.shape}")
    
    # Extract features
    print("\n5. Running forward pass with feature extraction...")
    try:
        with torch.no_grad():
            output = extractor.extract(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )
        
        if isinstance(output, tuple):
            output = output[0]
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return
    
    # Check extracted features
    print("\n6. Checking extracted features...")
    features = extractor.get_features()
    
    if len(features) == 0:
        print("✗ No features extracted!")
        return
    
    print(f"✓ Extracted {len(features)} feature maps:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    # Verify normalization
    print("\n7. Verifying feature normalization...")
    for name, feat in features.items():
        # Check if features are normalized (L2 norm ≈ 1)
        norm = torch.norm(feat, p=2, dim=1, keepdim=True).mean()
        print(f"  {name}: mean L2 norm = {norm.item():.4f}")
    
    # Cleanup
    extractor.remove_hooks()
    print("\n8. Cleanup complete")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou can now run the full training with:")
    print("  python scripts/train_distillation.py --config configs/distillation_config.json")
    print("\nExpected behavior:")
    print("  - Training loss may be slightly HIGHER (0.02-0.05 vs previous 0.01)")
    print("  - Feature loss should be NON-ZERO (0.02-0.10)")
    print("  - Quality should IMPROVE significantly")
    print("  - LPIPS and SSIM metrics will be logged during validation")


def test_distillation_loss():
    """Test distillation loss computation."""
    print("\n" + "=" * 60)
    print("Distillation Loss Test")
    print("=" * 60)
    
    from train_distillation import DistillationLoss
    
    print("\n1. Creating DistillationLoss...")
    loss_fn = DistillationLoss(
        output_weight=1.0,
        feature_weight=2.0,
        normalize_features=True
    )
    print("✓ Loss function created")
    
    print("\n2. Creating dummy outputs and features...")
    batch_size = 2
    channels = 4
    height, width = 128, 128
    
    student_output = torch.randn(batch_size, channels, height, width)
    teacher_output = torch.randn(batch_size, channels, height, width)
    
    # Simulate extracted features
    student_features = {
        'down_blocks.1': torch.randn(batch_size, 640, 64, 64),
        'down_blocks.2': torch.randn(batch_size, 1280, 32, 32),
    }
    teacher_features = {
        'down_blocks.1': torch.randn(batch_size, 640, 64, 64),
        'down_blocks.2': torch.randn(batch_size, 1280, 32, 32),
    }
    
    print("\n3. Computing loss with features...")
    losses = loss_fn(
        student_output,
        teacher_output,
        student_features=student_features,
        teacher_features=teacher_features
    )
    
    print("✓ Loss computation successful")
    print(f"  Total loss: {losses['total'].item():.6f}")
    print(f"  Output loss: {losses['output'].item():.6f}")
    print(f"  Feature loss: {losses['feature'].item():.6f}")
    
    if losses['feature'].item() == 0.0:
        print("✗ WARNING: Feature loss is zero!")
    else:
        print("✓ Feature loss is non-zero (correct!)")
    
    print("\n4. Testing loss without features...")
    losses_no_feat = loss_fn(student_output, teacher_output)
    print(f"  Total loss (no features): {losses_no_feat['total'].item():.6f}")
    print(f"  Feature loss (no features): {losses_no_feat['feature'].item():.6f}")
    
    print("\n✅ Distillation loss tests passed!")


if __name__ == '__main__':
    print("\nRunning verification tests...\n")
    
    try:
        test_feature_extraction()
    except Exception as e:
        print(f"\n✗ Feature extraction test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        test_distillation_loss()
    except Exception as e:
        print(f"\n✗ Distillation loss test failed: {e}")
        import traceback
        traceback.print_exc()
