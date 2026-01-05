"""
Dynamic INT8 quantization for UNet Linear layers.

This module applies dynamic quantization to Linear layers in the UNet,
which provides ~1.5-2x speedup on those layers. Since UNet is ~40% Linear,
expect ~1.5x total speedup.

Key points:
- Only quantizes Linear layers (attention + MLP)
- Conv2d layers remain FP32 (60% of UNet)
- No calibration needed (dynamic quantization)
- Must run on CPU (uses X86 backend)
"""

import torch
from torch.quantization import quantize_dynamic
from pathlib import Path


def quantize_linear_layers(model, inplace=False):
    """
    Apply dynamic INT8 quantization to Linear layers.
    
    Args:
        model: PyTorch model (UNet or VAE)
        inplace: If True, modify model in-place
        
    Returns:
        Quantized model
        
    Notes:
        - Model must be on CPU before quantization
        - Only Linear layers are quantized
        - Conv2d, LayerNorm, etc. remain FP32
    """
    # Ensure model is on CPU
    if next(model.parameters()).is_cuda:
        print("⚠️  Model is on GPU, moving to CPU for quantization...")
        model = model.cpu()
    
    # Ensure model is in eval mode
    model.eval()
    
    print("🔧 Applying dynamic INT8 quantization to Linear layers...")
    
    # Count layers before quantization
    linear_count = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
    print(f"   Found {linear_count} Linear layers to quantize")
    
    # Apply dynamic quantization
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Only quantize Linear layers
        dtype=torch.qint8,
        inplace=inplace
    )
    
    # Calculate size reduction
    def get_model_size_mb(model):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    original_size = get_model_size_mb(model if inplace else model)
    quantized_size = get_model_size_mb(quantized_model)
    
    print(f"   ✓ Quantization complete!")
    print(f"   ✓ Model size: {original_size:.2f} MB → {quantized_size:.2f} MB")
    print(f"   ✓ Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
    
    return quantized_model


def save_quantized_model(model, save_path):
    """
    Save quantized model.
    
    Args:
        model: Quantized model
        save_path: Path to save .pt file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save state dict only (more portable)
    torch.save({
        'model_state_dict': model.state_dict(),
        'quantized': True,
        'quantization_type': 'dynamic_int8_linear',
    }, save_path)
    
    print(f"✓ Saved quantized model to: {save_path}")


def load_quantized_model(model_class, checkpoint_path, *args, **kwargs):
    """
    Load quantized model from checkpoint.
    
    Args:
        model_class: Model class to instantiate
        checkpoint_path: Path to .pt file
        *args, **kwargs: Arguments for model_class
        
    Returns:
        Quantized model
    """
    # Create model instance
    model = model_class(*args, **kwargs)
    
    # Apply quantization
    model = quantize_linear_layers(model, inplace=True)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded quantized model from: {checkpoint_path}")
    
    return model


def analyze_quantization_impact(original_model, quantized_model):
    """
    Analyze the impact of quantization on model.
    
    Args:
        original_model: Original FP32 model
        quantized_model: Quantized INT8 model
        
    Returns:
        Dictionary with analysis results
    """
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    def count_quantized_layers(model):
        return sum(1 for m in model.modules() 
                  if hasattr(m, '_packed_params'))
    
    original_params = count_params(original_model)
    quantized_params = count_params(quantized_model)
    quantized_layers = count_quantized_layers(quantized_model)
    
    return {
        'original_params': original_params,
        'quantized_params': quantized_params,
        'quantized_layers': quantized_layers,
        'param_reduction': (1 - quantized_params / original_params) * 100,
    }


if __name__ == "__main__":
    # Test quantization on a simple model
    import torch.nn as nn
    
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(512, 512)
            self.conv1 = nn.Conv2d(4, 4, 3)
            self.linear2 = nn.Linear(512, 256)
            
        def forward(self, x):
            return x
    
    print("Testing dynamic INT8 quantization...")
    model = TestModel()
    quantized = quantize_linear_layers(model)
    
    print("\nAnalysis:")
    analysis = analyze_quantization_impact(model, quantized)
    for key, value in analysis.items():
        print(f"  {key}: {value}")
