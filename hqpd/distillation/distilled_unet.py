"""
Distilled UNet architecture for SDXL.

Based on Segmind SSD-1B architecture:
- Reduced transformer blocks (70 → 30)
- Simplified mid-block (no cross-attention)
- 50% parameter reduction (2.6B → 1.3B)

Optimized for CPU efficiency while maintaining image quality.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from diffusers import UNet2DConditionModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput


# Distilled UNet configuration (Segmind SSD-1B style)
DISTILLED_UNET_CONFIG = {
    "_class_name": "UNet2DConditionModel",
    "act_fn": "silu",
    "addition_embed_type": "text_time",
    "addition_embed_type_num_heads": 64,
    "addition_time_embed_dim": 256,
    "attention_head_dim": [5, 10, 20],
    "block_out_channels": [320, 640, 1280],
    "center_input_sample": False,
    "cross_attention_dim": 2048,
    "down_block_types": [
        "DownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D"
    ],
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_type": "UNetMidBlock2D",  # No cross-attention (key change)
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "projection_class_embeddings_input_dim": 2816,
    "sample_size": 128,
    # Key reduction: fewer transformer layers
    "transformer_layers_per_block": [1, 2, 4],  # Was [1, 2, 10]
    "up_block_types": [
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D"
    ],
    "use_linear_projection": True,
}


def create_distilled_unet(
    config: Optional[Dict] = None,
    pretrained_path: Optional[str] = None,
) -> UNet2DConditionModel:
    """
    Create a distilled UNet model.
    
    Args:
        config: Custom config (uses DISTILLED_UNET_CONFIG by default)
        pretrained_path: Path to pretrained weights (optional)
        
    Returns:
        UNet2DConditionModel with reduced architecture
    """
    cfg = config or DISTILLED_UNET_CONFIG.copy()
    
    print("Creating distilled UNet...")
    print(f"  Transformer layers: {cfg['transformer_layers_per_block']}")
    print(f"  Mid block: {cfg['mid_block_type']}")
    
    # Create model from config
    model = UNet2DConditionModel(**cfg)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
    
    # Load pretrained weights if provided
    if pretrained_path:
        print(f"  Loading weights from {pretrained_path}...")
        state_dict = torch.load(pretrained_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print("  Weights loaded (non-strict)")
    
    return model


def transfer_weights_from_teacher(
    teacher: UNet2DConditionModel,
    student: UNet2DConditionModel,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    """
    Transfer matching weights from teacher to student.
    
    Handles architecture mismatch by:
    1. Copying weights for matching layer names
    2. Skipping layers that don't exist in student
    3. Initializing new layers randomly
    
    Args:
        teacher: Full SDXL UNet (2.6B params)
        student: Distilled UNet (1.3B params)
        verbose: Print transfer details
        
    Returns:
        (transferred_keys, missing_keys)
    """
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    
    transferred = []
    missing = []
    
    for key in student_state.keys():
        if key in teacher_state:
            if teacher_state[key].shape == student_state[key].shape:
                student_state[key] = teacher_state[key].clone()
                transferred.append(key)
            else:
                missing.append(key)
                if verbose:
                    print(f"  Shape mismatch: {key}")
                    print(f"    Teacher: {teacher_state[key].shape}")
                    print(f"    Student: {student_state[key].shape}")
        else:
            missing.append(key)
    
    student.load_state_dict(student_state)
    
    if verbose:
        print(f"Weight transfer complete:")
        print(f"  Transferred: {len(transferred)} layers")
        print(f"  Missing/New: {len(missing)} layers")
    
    return transferred, missing


class FeatureExtractor:
    """
    Extract intermediate features from UNet for distillation.
    
    Attaches hooks to specified layers to capture activations
    during forward pass.
    """
    
    # Default feature tap points
    DEFAULT_TAPS = [
        "down_blocks.1.attentions.1",  # After down stage 1
        "down_blocks.2.attentions.1",  # After down stage 2
        "mid_block",                    # Mid block output
        "up_blocks.0.attentions.2",    # After up stage 0
        "up_blocks.1.attentions.2",    # After up stage 1
    ]
    
    def __init__(
        self,
        model: UNet2DConditionModel,
        tap_points: Optional[List[str]] = None,
    ):
        self.model = model
        self.tap_points = tap_points or self.DEFAULT_TAPS
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _get_module(self, name: str) -> nn.Module:
        """Get module by dot-separated path."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _register_hooks(self):
        """Register forward hooks on tap points."""
        for name in self.tap_points:
            try:
                module = self._get_module(name)
                hook = module.register_forward_hook(
                    lambda m, inp, out, n=name: self._save_feature(n, out)
                )
                self.hooks.append(hook)
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not find module '{name}': {e}")
    
    def _save_feature(self, name: str, output):
        """Save feature to dict."""
        if isinstance(output, tuple):
            output = output[0]
        self.features[name] = output
    
    def get_features(self) -> Dict[str, torch.Tensor]:
        """Get extracted features."""
        return self.features
    
    def clear(self):
        """Clear stored features."""
        self.features = {}
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class DistilledUNetWrapper(nn.Module):
    """
    Wrapper for distilled UNet with feature extraction support.
    
    Provides:
    - Forward pass with optional feature extraction
    - Feature adapters for channel alignment
    - CPU-optimized inference mode
    """
    
    def __init__(
        self,
        unet: UNet2DConditionModel,
        enable_feature_extraction: bool = True,
    ):
        super().__init__()
        self.unet = unet
        
        if enable_feature_extraction:
            self.feature_extractor = FeatureExtractor(unet)
        else:
            self.feature_extractor = None
        
        # Feature adapters (1x1 conv for channel alignment)
        # Will be populated if needed during distillation
        self.feature_adapters = nn.ModuleDict()
    
    def add_feature_adapter(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
    ):
        """Add 1x1 conv adapter for feature alignment."""
        if in_channels != out_channels:
            self.feature_adapters[name.replace(".", "_")] = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict] = None,
        return_features: bool = False,
        **kwargs,
    ) -> Union[UNet2DConditionOutput, Tuple[UNet2DConditionOutput, Dict]]:
        """
        Forward pass with optional feature extraction.
        
        Args:
            sample: Noisy latent (B, 4, H, W)
            timestep: Diffusion timestep
            encoder_hidden_states: Text embeddings (B, 77, 2048)
            added_cond_kwargs: Additional conditioning
            return_features: Return intermediate features
            
        Returns:
            UNet output, optionally with features dict
        """
        # Clear previous features
        if self.feature_extractor:
            self.feature_extractor.clear()
        
        # Forward through UNet
        output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            **kwargs,
        )
        
        if return_features and self.feature_extractor:
            features = self.feature_extractor.get_features()
            
            # Apply adapters if present
            adapted_features = {}
            for name, feat in features.items():
                adapter_name = name.replace(".", "_")
                if adapter_name in self.feature_adapters:
                    feat = self.feature_adapters[adapter_name](feat)
                adapted_features[name] = feat
            
            return output, adapted_features
        
        return output
    
    def enable_cpu_optimization(self):
        """Enable CPU-specific optimizations."""
        # Convert to channels-last memory format (better for CPU)
        self.unet = self.unet.to(memory_format=torch.channels_last)
        
        # Enable inference mode optimizations
        for module in self.unet.modules():
            if hasattr(module, "set_use_memory_efficient_attention"):
                module.set_use_memory_efficient_attention(True)
        
        print("CPU optimizations enabled")
    
    @torch.no_grad()
    def inference(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        encoder_hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Optimized inference (no grad, no feature extraction).
        """
        output = self.unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
        return output.sample


def load_segmind_ssd1b(device: str = "cuda") -> UNet2DConditionModel:
    """
    Load pre-trained Segmind SSD-1B UNet.
    
    This is a quick-start option using the already-distilled model.
    """
    from diffusers import UNet2DConditionModel
    
    print("Loading Segmind SSD-1B UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        "segmind/SSD-1B",
        subfolder="unet",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    unet = unet.to(device)
    
    num_params = sum(p.numel() for p in unet.parameters())
    print(f"Loaded: {num_params:,} parameters ({num_params/1e9:.2f}B)")
    
    return unet


if __name__ == "__main__":
    # Test distilled UNet creation
    print("=" * 70)
    print("Testing Distilled UNet")
    print("=" * 70)
    
    # Create model
    student = create_distilled_unet()
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    sample = torch.randn(1, 4, 128, 128)
    timestep = torch.tensor([500])
    encoder_hidden_states = torch.randn(1, 77, 2048)
    added_cond_kwargs = {
        "text_embeds": torch.randn(1, 1280),
        "time_ids": torch.randn(1, 6),
    }
    
    with torch.no_grad():
        output = student(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
    
    print(f"Output shape: {output.sample.shape}")
    print("✓ Forward pass successful!")
    
    # Test wrapper with features
    print("\nTesting feature extraction...")
    wrapper = DistilledUNetWrapper(student, enable_feature_extraction=True)
    
    output, features = wrapper(
        sample=sample,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
        return_features=True,
    )
    
    print(f"Extracted {len(features)} feature maps:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    print("\n✓ All tests passed!")
