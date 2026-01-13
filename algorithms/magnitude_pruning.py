"""
Magnitude Pruning for Diffusion Models

Implements simple magnitude-based pruning as described in:
"Efficient Pruning of Text-to-Image Models" (arxiv:2411.15113)

Key findings from paper:
- Simple magnitude pruning outperforms advanced methods (Wanda, SparseGPT) for diffusion models
- Optimal configuration: Text encoder (47.5% sparsity), UNet (35% sparsity)
- Beyond these thresholds, sudden performance drops occur
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import copy


class MagnitudePruner:
    """
    Simple magnitude-based pruning for SDXL components.
    
    Prunes weights with smallest absolute magnitudes, as these typically
    carry less semantic information in diffusion models.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize pruner.
        
        Args:
            verbose: Print progress information
        """
        self.verbose = verbose
        self.pruning_masks = {}
        self.pruning_stats = {}
    
    def compute_sparsity(self, module: nn.Module) -> float:
        """
        Compute current sparsity of a module.
        
        Args:
            module: PyTorch module
            
        Returns:
            Sparsity (fraction of zero weights)
        """
        total_params = 0
        zero_params = 0
        
        for param in module.parameters():
            if param.requires_grad:
                total_params += param.numel()
                zero_params += (param.data == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def prune_layer(
        self,
        layer: nn.Module,
        sparsity: float,
        structured: bool = False
    ) -> None:
        """
        Prune a single layer by magnitude.
        
        Args:
            layer: Layer to prune (Linear, Conv2d, etc.)
            sparsity: Target sparsity (0.0-1.0)
            structured: If True, prune entire channels/filters (not implemented yet)
        """
        if not hasattr(layer, 'weight'):
            if self.verbose:
                print(f"  ⚠️  Layer {layer.__class__.__name__} has no weight, skipping")
            return
        
        weight = layer.weight.data
        
        # Compute magnitude
        weight_magnitude = torch.abs(weight)
        
        # Flatten for threshold computation
        weight_flat = weight_magnitude.view(-1)
        
        # Find threshold using kthvalue (more memory efficient than quantile)
        # kthvalue finds the k-th smallest element
        k = max(1, int(sparsity * weight_flat.numel()))
        
        # For very large tensors, use numpy as fallback
        try:
            threshold = torch.kthvalue(weight_flat, k).values
        except RuntimeError:
            # Fallback to numpy for extremely large tensors
            import numpy as np
            weight_np = weight_flat.cpu().numpy()
            threshold = torch.tensor(np.percentile(weight_np, sparsity * 100))
            if weight.is_cuda:
                threshold = threshold.cuda()
        
        # Create mask (keep weights above threshold)
        mask = weight_magnitude > threshold
        
        # Apply mask
        weight.mul_(mask.float())
        
        # Store mask for later (in case we need to fine-tune)
        layer_id = id(layer)
        self.pruning_masks[layer_id] = mask
        
        # Compute actual sparsity achieved
        actual_sparsity = (weight == 0).float().mean().item()
        
        if self.verbose:
            layer_name = layer.__class__.__name__
            shape = tuple(weight.shape)
            print(f"  ✂️  {layer_name}{shape}: Target {sparsity*100:.1f}%, "
                  f"Actual {actual_sparsity*100:.1f}%")
    
    def prune_model(
        self,
        model: nn.Module,
        sparsity: float,
        layer_types: Optional[List[type]] = None,
        skip_layers: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Prune entire model uniformly.
        
        Args:
            model: Model to prune
            sparsity: Target sparsity
            layer_types: Types of layers to prune (default: Linear, Conv2d)
            skip_layers: Layer name patterns to skip
            
        Returns:
            Dictionary of pruning statistics
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Pruning Model: {model.__class__.__name__}")
            print(f"Target Sparsity: {sparsity*100:.1f}%")
            print(f"{'='*60}\n")
        
        # Default layer types
        if layer_types is None:
            layer_types = [nn.Linear, nn.Conv2d]
        
        skip_layers = skip_layers or []
        
        # Count initial parameters
        initial_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        initial_sparsity = self.compute_sparsity(model)
        
        # Prune each matching layer
        pruned_layers = 0
        for name, module in model.named_modules():
            # Check if module type matches
            if not any(isinstance(module, layer_type) for layer_type in layer_types):
                continue
            
            # Check skip list
            if any(skip_pattern in name for skip_pattern in skip_layers):
                if self.verbose:
                    print(f"⏭️  Skipping {name} (in skip list)")
                continue
            
            self.prune_layer(module, sparsity=sparsity)
            pruned_layers += 1
        
        # Compute final stats
        final_sparsity = self.compute_sparsity(model)
        
        stats = {
            'initial_params': initial_params,
            'initial_sparsity': initial_sparsity,
            'final_sparsity': final_sparsity,
            'target_sparsity': sparsity,
            'pruned_layers': pruned_layers,
        }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Pruning Complete")
            print(f"{'='*60}")
            print(f"Initial sparsity: {initial_sparsity*100:.2f}%")
            print(f"Final sparsity: {final_sparsity*100:.2f}%")
            print(f"Target sparsity: {sparsity*100:.2f}%")
            print(f"Pruned layers: {pruned_layers}")
        
        self.pruning_stats = stats
        return stats
    
    def prune_sdxl_components(
        self,
        pipeline,
        text_encoder_sparsity: float = 0.475,
        unet_sparsity: float = 0.35
    ) -> Dict[str, Dict]:
        """
        Prune SDXL components with paper-recommended sparsity levels.
        
        From "Efficient Pruning..." paper:
        - Text Encoder: 47.5% sparsity
        - UNet: 35% sparsity
        
        Args:
            pipeline: StableDiffusionXLPipeline
            text_encoder_sparsity: Sparsity for text encoders
            unet_sparsity: Sparsity for UNet
            
        Returns:
            Dictionary of pruning statistics per component
        """
        all_stats = {}
        
        # Prune Text Encoder 1 (OpenCLIP)
        if hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
            if self.verbose:
                print("\n📝 Pruning Text Encoder 1 (OpenCLIP ViT-bigG/14)...")
            
            stats = self.prune_model(
                model=pipeline.text_encoder,
                sparsity=text_encoder_sparsity,
                skip_layers=['embeddings', 'layernorm', 'final_layer_norm']
            )
            all_stats['text_encoder_1'] = stats
        
        # Prune Text Encoder 2 (CLIP)
        if hasattr(pipeline, 'text_encoder_2') and pipeline.text_encoder_2 is not None:
            if self.verbose:
                print("\n📝 Pruning Text Encoder 2 (CLIP ViT-L/14)...")
            
            stats = self.prune_model(
                model=pipeline.text_encoder_2,
                sparsity=text_encoder_sparsity,
                skip_layers=['embeddings', 'layernorm', 'final_layer_norm']
            )
            all_stats['text_encoder_2'] = stats
        
        # Prune UNet
        if hasattr(pipeline, 'unet') and pipeline.unet is not None:
            if self.verbose:
                print("\n🎨 Pruning UNet (Diffusion Generator)...")
            
            stats = self.prune_model(
                model=pipeline.unet,
                sparsity=unet_sparsity,
                skip_layers=['time_embedding', 'norm', 'conv_in', 'conv_out']
            )
            all_stats['unet'] = stats
        
        # Summary
        if self.verbose:
            self._print_summary(all_stats)
        
        return all_stats
    
    def _print_summary(self, all_stats: Dict[str, Dict]):
        """Print pruning summary for all components."""
        print(f"\n{'='*60}")
        print("Pruning Summary")
        print(f"{'='*60}\n")
        
        total_params_before = 0
        total_params_after = 0
        
        for component, stats in all_stats.items():
            params = stats['initial_params']
            final_sparsity = stats['final_sparsity']
            pruned_params = int(params * final_sparsity)
            remaining_params = params - pruned_params
            
            total_params_before += params
            total_params_after += remaining_params
            
            print(f"{component}:")
            print(f"  Initial: {params:,} params")
            print(f"  Pruned: {pruned_params:,} params ({final_sparsity*100:.1f}%)")
            print(f"  Remaining: {remaining_params:,} params\n")
        
        compression_ratio = total_params_before / max(total_params_after, 1)
        size_reduction_pct = (1 - total_params_after / max(total_params_before, 1)) * 100
        
        print(f"Overall:")
        print(f"  Total Before: {total_params_before:,} params")
        print(f"  Total After: {total_params_after:,} params")
        print(f"  Compression: {compression_ratio:.2f}x")
        print(f"  Size Reduction: {size_reduction_pct:.1f}%")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test with dummy model
    print("Testing Magnitude Pruner...\n")
    
    # Create a simple test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.fc2 = nn.Linear(50, 10)
            self.conv = nn.Conv2d(3, 16, 3)
        
        def forward(self, x):
            return x
    
    model = TestModel()
    
    # Test pruning
    pruner = MagnitudePruner(verbose=True)
    
    print("1. Individual layer pruning:")
    pruner.prune_layer(model.fc1, sparsity=0.5)
    
    print("\n2. Full model pruning:")
    model2 = TestModel()
    stats = pruner.prune_model(model2, sparsity=0.35)
    
    print(f"\n✅ Pruning test complete!")
    print(f"Final model sparsity: {stats['final_sparsity']*100:.1f}%")
