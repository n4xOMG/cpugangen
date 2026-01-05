"""
Distillation loss functions for UNet knowledge transfer.

Implements multi-component loss with:
- Output MSE (primary signal)
- Per-layer feature distillation with adaptive weights
- Cosine similarity for direction preservation
- Optional LPIPS for perceptual quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class DistillationLoss(nn.Module):
    """
    Multi-component distillation loss for UNet.
    
    Components:
    1. Output MSE: Match teacher's noise prediction
    2. Feature MSE: Match intermediate layer activations
    3. Cosine similarity: Preserve feature directions
    4. (Optional) LPIPS: Perceptual quality preservation
    """
    
    # Per-layer weights based on importance
    DEFAULT_LAYER_WEIGHTS = {
        "down_blocks.1": 0.5,   # Less critical
        "down_blocks.2": 1.0,   # Important for detail
        "mid_block": 2.0,       # Most critical
        "up_blocks.0": 1.0,     # Important for detail
        "up_blocks.1": 0.5,     # Less critical
    }
    
    def __init__(
        self,
        output_weight: float = 1.0,
        feature_weight: float = 0.5,
        cosine_weight: float = 0.1,
        layer_weights: Optional[Dict[str, float]] = None,
        enable_lpips: bool = False,
    ):
        """
        Args:
            output_weight: Weight for output MSE loss
            feature_weight: Weight for feature distillation loss
            cosine_weight: Weight for cosine similarity loss
            layer_weights: Per-layer weights (uses defaults if None)
            enable_lpips: Enable LPIPS perceptual loss
        """
        super().__init__()
        
        self.output_weight = output_weight
        self.feature_weight = feature_weight
        self.cosine_weight = cosine_weight
        self.layer_weights = layer_weights or self.DEFAULT_LAYER_WEIGHTS
        
        self.enable_lpips = enable_lpips
        if enable_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='vgg')
            except ImportError:
                print("Warning: lpips not installed, disabling LPIPS loss")
                self.enable_lpips = False
    
    def _get_layer_weight(self, layer_name: str) -> float:
        """Get weight for a layer based on its name."""
        for key, weight in self.layer_weights.items():
            if key in layer_name:
                return weight
        return 0.5  # Default weight
    
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor],
        decoded_student: Optional[torch.Tensor] = None,
        decoded_teacher: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.
        
        Args:
            student_output: Student's noise prediction (B, 4, H, W)
            teacher_output: Teacher's noise prediction (B, 4, H, W)
            student_features: Student's intermediate features
            teacher_features: Teacher's intermediate features
            decoded_student: VAE-decoded student output (optional, for LPIPS)
            decoded_teacher: VAE-decoded teacher output (optional, for LPIPS)
            
        Returns:
            total_loss, metrics_dict
        """
        metrics = {}
        
        # 1. Output MSE loss
        output_loss = F.mse_loss(student_output, teacher_output.detach())
        metrics["output_loss"] = output_loss.item()
        
        # 2. Feature distillation loss
        feature_loss = torch.tensor(0.0, device=student_output.device)
        
        for name in student_features.keys():
            if name in teacher_features:
                s_feat = student_features[name]
                t_feat = teacher_features[name].detach()
                
                # Handle shape mismatch (student may have fewer channels)
                if s_feat.shape != t_feat.shape:
                    # Skip if shapes don't match (should use adapter)
                    continue
                
                layer_weight = self._get_layer_weight(name)
                feat_mse = F.mse_loss(s_feat, t_feat)
                feature_loss = feature_loss + layer_weight * feat_mse
                
                metrics[f"feat_{name.replace('.', '_')}"] = feat_mse.item()
        
        metrics["feature_loss"] = feature_loss.item()
        
        # 3. Cosine similarity loss
        cosine_loss = torch.tensor(0.0, device=student_output.device)
        
        for name in student_features.keys():
            if name in teacher_features:
                s_feat = student_features[name]
                t_feat = teacher_features[name].detach()
                
                if s_feat.shape != t_feat.shape:
                    continue
                
                # Flatten and normalize
                s_norm = F.normalize(s_feat.flatten(1), dim=1)
                t_norm = F.normalize(t_feat.flatten(1), dim=1)
                
                # 1 - cos_sim to minimize
                cos_dist = 1 - F.cosine_similarity(s_norm, t_norm).mean()
                cosine_loss = cosine_loss + cos_dist
        
        # Average over number of features
        num_features = len(student_features)
        if num_features > 0:
            cosine_loss = cosine_loss / num_features
        
        metrics["cosine_loss"] = cosine_loss.item()
        
        # 4. LPIPS loss (optional)
        lpips_loss = torch.tensor(0.0, device=student_output.device)
        if self.enable_lpips and decoded_student is not None:
            lpips_loss = self.lpips_fn(decoded_student, decoded_teacher).mean()
            metrics["lpips_loss"] = lpips_loss.item()
        
        # Combined loss
        total_loss = (
            self.output_weight * output_loss +
            self.feature_weight * feature_loss +
            self.cosine_weight * cosine_loss +
            (0.1 * lpips_loss if self.enable_lpips else 0)
        )
        
        metrics["total_loss"] = total_loss.item()
        
        return total_loss, metrics


class ProgressiveDistillationLoss(DistillationLoss):
    """
    Distillation loss with progressive weight schedule.
    
    Early training: Focus on output matching
    Later training: Increase feature importance
    """
    
    def __init__(
        self,
        max_steps: int = 50000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_steps = max_steps
        self.current_step = 0
    
    def update_step(self, step: int):
        """Update current training step."""
        self.current_step = step
    
    def get_progressive_weights(self) -> Tuple[float, float, float]:
        """
        Get progressive weights based on training step.
        
        Schedule:
        - Steps 0-10k: output=1.0, feature=0.2, cosine=0.05
        - Steps 10k-30k: output=1.0, feature=0.5, cosine=0.1
        - Steps 30k+: output=1.0, feature=0.8, cosine=0.15
        """
        progress = self.current_step / self.max_steps
        
        if progress < 0.2:
            return 1.0, 0.2, 0.05
        elif progress < 0.6:
            return 1.0, 0.5, 0.1
        else:
            return 1.0, 0.8, 0.15
    
    def forward(self, *args, **kwargs):
        # Update weights based on progress
        out_w, feat_w, cos_w = self.get_progressive_weights()
        self.output_weight = out_w
        self.feature_weight = feat_w
        self.cosine_weight = cos_w
        
        return super().forward(*args, **kwargs)


def compute_distillation_step(
    student: nn.Module,
    teacher: nn.Module,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: Dict,
    loss_fn: DistillationLoss,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute a single distillation training step.
    
    This is a convenience function that handles the full forward pass
    for both teacher and student, then computes the loss.
    
    Args:
        student: Student UNet wrapper (with feature extraction)
        teacher: Teacher UNet (frozen)
        sample: Noisy latent input
        timestep: Current timestep
        encoder_hidden_states: Text embeddings
        added_cond_kwargs: Additional SDXL conditioning
        loss_fn: Distillation loss function
        
    Returns:
        loss, metrics
    """
    # Teacher forward (no grad)
    with torch.no_grad():
        teacher_output = teacher(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        
        # Get teacher features (need to run with feature extraction)
        if hasattr(teacher, 'feature_extractor'):
            teacher_features = teacher.feature_extractor.get_features()
        else:
            teacher_features = {}
    
    # Student forward (with grad)
    student_output, student_features = student(
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
        return_features=True,
    )
    
    # Compute loss
    loss, metrics = loss_fn(
        student_output=student_output.sample,
        teacher_output=teacher_output.sample,
        student_features=student_features,
        teacher_features=teacher_features,
    )
    
    return loss, metrics


if __name__ == "__main__":
    print("Testing distillation losses...")
    
    # Create dummy data
    B, C, H, W = 2, 4, 64, 64
    
    student_out = torch.randn(B, C, H, W)
    teacher_out = torch.randn(B, C, H, W)
    
    student_features = {
        "down_blocks.1": torch.randn(B, 640, H//4, H//4),
        "down_blocks.2": torch.randn(B, 1280, H//8, H//8),
        "mid_block": torch.randn(B, 1280, H//8, H//8),
    }
    
    teacher_features = {
        "down_blocks.1": torch.randn(B, 640, H//4, H//4),
        "down_blocks.2": torch.randn(B, 1280, H//8, H//8),
        "mid_block": torch.randn(B, 1280, H//8, H//8),
    }
    
    # Test basic loss
    loss_fn = DistillationLoss()
    loss, metrics = loss_fn(student_out, teacher_out, student_features, teacher_features)
    
    print(f"Total loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test progressive loss
    prog_loss_fn = ProgressiveDistillationLoss(max_steps=50000)
    
    for step in [0, 10000, 30000, 50000]:
        prog_loss_fn.update_step(step)
        weights = prog_loss_fn.get_progressive_weights()
        print(f"\nStep {step}: output={weights[0]}, feature={weights[1]}, cosine={weights[2]}")
    
    print("\n✓ Loss tests passed!")
