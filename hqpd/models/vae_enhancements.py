"""
Enhanced TinyVAE training with:
- Pretrained initialization from SD 1.5
- LPIPS perceptual loss
- Validation image generation
- Gradient checkpointing
"""

import torch
import torch.nn as nn
import lpips

class EnhancedVAELoss(nn.Module):
    """
    Enhanced loss with L1, L2, and LPIPS perceptual loss.
    
    CRITICAL: LPIPS loss is essential for maintaining perceptual quality!
    """
    
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 0.5,
        lpips_weight: float = 0.1,  # Critical for quality
        device: str = 'cuda'
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.lpips_weight = lpips_weight
        
        # Initialize LPIPS (perceptual loss)
        # Uses pretrained VGG network to compare feature representations
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        self.lpips_fn.requires_grad_(False)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Compute multi-component loss.
        
        Args:
            pred: Predicted image (B, 3, H, W) in [-1, 1]
            target: Target image (B, 3, H, W) in [-1, 1]
            mask: Optional mask (B, H, W) with 1 for valid pixels
            
        Returns:
            Dict with total loss and components
        """
        losses = {}
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
            
            # Masked L1 loss
            l1_loss = torch.abs(pred - target) * mask
            l1_loss = l1_loss.sum() / (mask.sum() * pred.shape[1] + 1e-8)
            
            # Masked L2 loss
            l2_loss = ((pred - target) ** 2) * mask
            l2_loss = l2_loss.sum() / (mask.sum() * pred.shape[1] + 1e-8)
        else:
            # Standard losses
            l1_loss = torch.nn.functional.l1_loss(pred, target)
            l2_loss = torch.nn.functional.mse_loss(pred, target)
        
        losses['l1'] = l1_loss
        losses['l2'] = l2_loss
        
        # LPIPS perceptual loss (CRITICAL for quality)
        # Compares high-level features, not just pixels
        # This prevents "blurry" reconstructions
        with torch.cuda.amp.autocast(enabled=False):
            # LPIPS expects float32, not float16
            pred_fp32 = pred.float()
            target_fp32 = target.float()
            
            # If mask provided, zero out padded regions
            if mask is not None:
                pred_fp32 = pred_fp32 * mask
                target_fp32 = target_fp32 * mask
            
            lpips_loss = self.lpips_fn(pred_fp32, target_fp32).mean()
        
        losses['lpips'] = lpips_loss
        
        # Total loss
        total = (
            self.l1_weight * l1_loss +
            self.l2_weight * l2_loss +
            self.lpips_weight * lpips_loss
        )
        losses['total'] = total
        
        return losses


def load_pretrained_illustrious_decoder(device='cuda'):
    """
    Load Illustrious VAE decoder as initialization (NOT random init!).
    
    This is CRITICAL: Starting from pretrained weights gives much better
    convergence than random initialization.
    
    Uses Illustrious (SDXL-based, anime-tuned) which matches your teacher model!
    """
    from diffusers import AutoencoderKL
    
    print("Loading Illustrious VAE for pretrained initialization...")
    print("  Model: martineux/janku6 (Illustrious anime-tuned SDXL)")
    
    illustrious_vae = AutoencoderKL.from_pretrained(
        "martineux/janku6",
        subfolder="vae",
        torch_dtype=torch.float32
    ).to(device)
    
    print("  ✅ Illustrious VAE loaded successfully!")
    
    return illustrious_vae.decoder


def initialize_from_pretrained(
    tiny_decoder,
    pretrained_decoder,
    strategy='copy_matching'
):
    """
    Initialize TinyVAE from pretrained SD 1.5 decoder.
    
    Strategies:
    - 'copy_matching': Copy weights for matching layer names/shapes
    - 'interpolate': Interpolate for mismatched sizes
    
    Args:
        tiny_decoder: Your TinyVAEDecoder model
        pretrained_decoder: SD 1.5 decoder
        strategy: Initialization strategy
    """
    print(f"\nInitializing TinyVAE from pretrained (strategy: {strategy})...")
    
    pretrained_dict = pretrained_decoder.state_dict()
    tiny_dict = tiny_decoder.state_dict()
    
    matched = 0
    skipped = 0
    
    for name, param in tiny_dict.items():
        if name in pretrained_dict:
            pretrained_param = pretrained_dict[name]
            
            # Check if shapes match
            if param.shape == pretrained_param.shape:
                # Direct copy
                tiny_dict[name] = pretrained_param.clone()
                matched += 1
            elif strategy == 'interpolate':
                # Interpolate for different sizes (advanced)
                # For now, skip these
                skipped += 1
            else:
                skipped += 1
        else:
            skipped += 1
    
    # Load initialized weights
    tiny_decoder.load_state_dict(tiny_dict)
    
    print(f"  Matched layers: {matched}")
    print(f"  Skipped layers: {skipped}")
    print(f"  ✅ Pretrained initialization complete!\n")
    
    return tiny_decoder


# Example usage in train_vae_decoder.py:
"""
# In main() function, after creating model:

if config['model'].get('use_pretrained_init', True):
    # Load Illustrious pretrained decoder (matches your teacher model!)
    pretrained_decoder = load_pretrained_illustrious_decoder(device)
    
    # Initialize TinyVAE from it
    model = initialize_from_pretrained(
        model,
        pretrained_decoder,
        strategy='copy_matching'
    )
    
    # Clean up
    del pretrained_decoder
    torch.cuda.empty_cache()

# Then continue with training...
"""
