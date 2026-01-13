"""
Conservative Magnitude Pruning for SDXL

This is a SAFER version that:
1. Skips more critical layers
2. Uses gradual sparsity (less aggressive)
3. Tests with lower sparsity first

Reference: "Efficient Pruning of Text-to-Image Models" (arxiv:2411.15113)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm


def should_skip_layer(name: str) -> bool:
    """
    Determine if a layer should be skipped during pruning.
    
    Skip:
    - Embeddings (position, token, etc.)
    - Normalization layers (LayerNorm, GroupNorm)
    - Input/output projections
    - Time embeddings
    - Attention projection layers (critical for structure)
    """
    skip_patterns = [
        'embedding',
        'norm',  # All normalization
        'ln',    # LayerNorm abbreviations
        'gn',    # GroupNorm abbreviations
        'time_embed',
        'time_proj',
        'conv_in',    # Input convolution
        'conv_out',   # Output convolution
        'proj_in',    # Input projection
        'proj_out',   # Output projection
        'to_out',     # Attention output projection (CRITICAL!)
        'to_k',       # Attention key (keep for now)
        'to_q',       # Attention query (keep for now)
        'to_v',       # Attention value (keep for now)
    ]
    
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in skip_patterns)


def get_layer_sparsity(name: str, base_sparsity: float) -> float:
    """
    Get sparsity for a specific layer based on its role.
    
    Strategy:
    - Down blocks (early): Lower sparsity (more important for structure)
    - Mid blocks: Medium sparsity
    - Up blocks (late): Higher sparsity (less critical)
    - FF layers: Can handle more pruning
    """
    name_lower = name.lower()
    
    # Down blocks (structure builders) - be conservative
    if 'down_block' in name_lower or 'down.0' in name_lower:
        return base_sparsity * 0.5  # Half the sparsity
    
    # Mid blocks - moderate
    elif 'mid_block' in name_lower:
        return base_sparsity * 0.75
    
    # Up blocks - can tolerate more
    elif 'up_block' in name_lower or 'up.0' in name_lower:
        return base_sparsity * 1.0
    
    # Feed-forward layers - can tolerate more
    elif 'ff' in name_lower or 'mlp' in name_lower:
        return base_sparsity * 1.2
    
    # Default
    else:
        return base_sparsity


def prune_sdxl_safe(
    pipeline,
    text_encoder_sparsity: float = 0.25,  # Conservative: 25% instead of 47.5%
    unet_sparsity: float = 0.15,          # Conservative: 15% instead of 35%
    verbose: bool = True
):
    """
    Safely prune SDXL with conservative settings.
    
    Args:
        pipeline: StableDiffusionXLPipeline
        text_encoder_sparsity: Target sparsity for text encoders (default: 25%)
        unet_sparsity: Target sparsity for UNet (default: 15%)
        verbose: Print progress
    """
    stats = {}
    
    # Prune Text Encoders (conservative)
    for encoder_name in ['text_encoder', 'text_encoder_2']:
        if not hasattr(pipeline, encoder_name):
            continue
        
        encoder = getattr(pipeline, encoder_name)
        if encoder is None:
            continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Pruning {encoder_name} (target: {text_encoder_sparsity*100:.1f}%)")
            print(f"{'='*60}\n")
        
        pruned = 0
        skipped = 0
        total = 0
        
        for name, module in encoder.named_modules():
            if not isinstance(module, (nn.Linear, nn.Conv2d)):
                continue
            
            total += 1
            
            # Skip critical layers
            if should_skip_layer(name):
                if verbose:
                    print(f"  ⏭️  Skipping {name} (critical layer)")
                skipped += 1
                continue
            
            # Prune this layer
            if hasattr(module, 'weight'):
                weight = module.weight.data
                weight_flat = torch.abs(weight).view(-1)
                
                k = max(1, int(text_encoder_sparsity * weight_flat.numel()))
                threshold = torch.kthvalue(weight_flat, k).values
                
                mask = torch.abs(weight) > threshold
                weight.mul_(mask.float())
                
                actual_sparsity = (weight == 0).float().mean().item()
                pruned += 1
                
                if verbose:
                    print(f"  ✂️  {name}: {actual_sparsity*100:.1f}% sparse")
        
        stats[encoder_name] = {
            'total_layers': total,
            'pruned': pruned,
            'skipped': skipped
        }
    
    # Prune UNet (very conservative, layer-specific sparsity)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Pruning UNet (base target: {unet_sparsity*100:.1f}%)")
        print(f"{'='*60}\n")
    
    pruned = 0
    skipped = 0
    total = 0
    
    for name, module in pipeline.unet.named_modules():
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        
        total += 1
        
        # Skip critical layers
        if should_skip_layer(name):
            if verbose:
                print(f"  ⏭️  Skipping {name} (critical)")
            skipped += 1
            continue
        
        # Get layer-specific sparsity
        layer_sparsity = get_layer_sparsity(name, unet_sparsity)
        layer_sparsity = min(layer_sparsity, 0.3)  # Cap at 30% maximum
        
        # Prune
        if hasattr(module, 'weight'):
            weight = module.weight.data
            weight_flat = torch.abs(weight).view(-1)
            
            k = max(1, int(layer_sparsity * weight_flat.numel()))
            
            try:
                threshold = torch.kthvalue(weight_flat, k).values
            except RuntimeError:
                # Fallback for very large tensors
                weight_np = weight_flat.cpu().numpy()
                threshold = torch.tensor(np.percentile(weight_np, layer_sparsity * 100))
            
            mask = torch.abs(weight) > threshold
            weight.mul_(mask.float())
            
            actual_sparsity = (weight == 0).float().mean().item()
            pruned += 1
            
            if verbose and actual_sparsity > 0.01:  # Only print if actually pruned
                print(f"  ✂️  {name}: {actual_sparsity*100:.1f}% sparse (target: {layer_sparsity*100:.1f}%)")
    
    stats['unet'] = {
        'total_layers': total,
        'pruned': pruned,
        'skipped': skipped
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("Pruning Summary")
        print(f"{'='*60}\n")
        for component, data in stats.items():
            print(f"{component}:")
            print(f"  Total layers: {data['total_layers']}")
            print(f"  Pruned: {data['pruned']}")
            print(f"  Skipped: {data['skipped']}")
            print()
    
    return stats


if __name__ == "__main__":
    print("Conservative Pruning Module")
    print("Use this for safer pruning with lower sparsity targets")
