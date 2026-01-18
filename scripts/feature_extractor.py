#!/usr/bin/env python3
"""
UNet Feature Extraction Module for Knowledge Distillation

Captures intermediate activations from specified UNet blocks using forward hooks.
Designed for SDXL UNet2DConditionModel feature matching during distillation.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetFeatureExtractor:
    """
    Extract intermediate features from SDXL UNet blocks using forward hooks.
    
    Example usage:
        extractor = UNetFeatureExtractor(unet, ['down_blocks.1', 'down_blocks.2'])
        output = extractor.extract(noisy_latents, timesteps, encoder_hidden_states, ...)
        features = extractor.features  # Dict of extracted features
    """
    
    def __init__(
        self, 
        unet: nn.Module,
        layer_names: List[str],
        normalize: bool = True
    ):
        """
        Args:
            unet: UNet2DConditionModel to extract features from
            layer_names: List of layer names to hook (e.g., ['down_blocks.1', 'up_blocks.0'])
            normalize: Whether to L2-normalize features before returning
        """
        self.unet = unet
        self.layer_names = layer_names
        self.normalize = normalize
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        for name in self.layer_names:
            # Navigate to the module by name
            module = self._get_module_by_name(name)
            if module is not None:
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)
            else:
                print(f"Warning: Layer {name} not found in UNet")
    
    def _get_module_by_name(self, name: str) -> Optional[nn.Module]:
        """Get module by dotted name (e.g., 'down_blocks.1')."""
        parts = name.split('.')
        module = self.unet
        
        for part in parts:
            if part.isdigit():
                # Handle list indexing (e.g., down_blocks.1)
                module = module[int(part)]
            else:
                # Handle attribute access (e.g., down_blocks)
                if hasattr(module, part):
                    module = getattr(module, part)
                else:
                    return None
        
        return module
    
    def _make_hook(self, name: str):
        """Create a hook function that stores activations."""
        def hook(module, input, output):
            # Handle tuple outputs (some blocks return (hidden_states, ...)
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            # Store activation (detach to save memory)
            self.features[name] = activation.detach()
        
        return hook
    
    def extract(self, *args, **kwargs) -> torch.Tensor:
        """
        Run UNet forward pass and extract features.
        
        Args:
            *args, **kwargs: Arguments to pass to UNet forward
            
        Returns:
            UNet output (same as calling unet(*args, **kwargs))
        """
        # Clear previous features
        self.features.clear()
        
        # Forward pass (hooks will populate self.features)
        output = self.unet(*args, **kwargs)
        
        # Normalize features if requested
        if self.normalize:
            self._normalize_features()
        
        return output
    
    def _normalize_features(self):
        """L2-normalize all extracted features channel-wise."""
        for name, feat in self.features.items():
            # Normalize along channel dimension (dim=1 for NCHW format)
            # Add small epsilon to avoid division by zero
            norm = torch.norm(feat, p=2, dim=1, keepdim=True) + 1e-8
            self.features[name] = feat / norm
    
    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get extracted features as dictionary."""
        return self.features
    
    def clear(self):
        """Clear stored features to free memory."""
        self.features.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


class FeatureAligner(nn.Module):
    """
    Align student and teacher features with different spatial/channel dimensions.
    
    Uses learnable 1x1 convolutions to project student features to match teacher dimensions.
    This is useful when student UNet has different channel counts than teacher.
    """
    
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleDict()
    
    def forward(
        self, 
        student_feat: torch.Tensor, 
        teacher_feat: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """
        Align student feature to match teacher feature dimensions.
        
        Args:
            student_feat: Student feature [B, C_s, H, W]
            teacher_feat: Teacher feature [B, C_t, H, W]
            layer_name: Layer identifier for creating/reusing projection
            
        Returns:
            Aligned student feature [B, C_t, H, W]
        """
        # Check if dimensions match
        if student_feat.shape == teacher_feat.shape:
            return student_feat
        
        aligned_feat = student_feat
        
        # 1. Align spatial dimensions
        if student_feat.shape[2:] != teacher_feat.shape[2:]:
            aligned_feat = F.interpolate(
                aligned_feat,
                size=teacher_feat.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 2. Align channel dimensions (if needed)
        if student_feat.shape[1] != teacher_feat.shape[1]:
            # Create projection layer if not exists
            if layer_name not in self.projections:
                self.projections[layer_name] = nn.Conv2d(
                    student_feat.shape[1],
                    teacher_feat.shape[1],
                    kernel_size=1,
                    bias=False
                ).to(student_feat.device)
                
                # Initialize with identity-like mapping
                nn.init.xavier_uniform_(self.projections[layer_name].weight)
            
            # Project channels
            aligned_feat = self.projections[layer_name](aligned_feat)
        
        return aligned_feat


def get_default_feature_layers(unet_type: str = 'sdxl') -> List[str]:
    """
    Get recommended feature extraction layers for different UNet types.
    
    Args:
        unet_type: Type of UNet ('sdxl', 'ssd1b', etc.)
        
    Returns:
        List of layer names to extract features from
    """
    if unet_type.lower() == 'sdxl':
        # Extract from mid-level semantic blocks
        # Skip shallow (low-level) and deep (too specific) layers
        return [
            'down_blocks.1',  # Downsampling block 2
            'down_blocks.2',  # Downsampling block 3
            'up_blocks.0',    # Upsampling block 1
            'up_blocks.1'     # Upsampling block 2
        ]
    else:
        # Default: same as SDXL
        return [
            'down_blocks.1',
            'down_blocks.2',
            'up_blocks.0',
            'up_blocks.1'
        ]


if __name__ == '__main__':
    """Test feature extraction on dummy UNet."""
    print("Feature Extractor Module - Test Mode")
    print("=" * 60)
    
    # This is a placeholder test - actual testing requires loading a real UNet
    print("\n✅ Module loaded successfully")
    print(f"Default SDXL feature layers: {get_default_feature_layers('sdxl')}")
    print("\nTo use:")
    print("  extractor = UNetFeatureExtractor(unet, get_default_feature_layers())")
    print("  output = extractor.extract(noisy_latents, timesteps, ...)")
    print("  features = extractor.features  # Dict[str, Tensor]")
