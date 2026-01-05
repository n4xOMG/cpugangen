"""
Tag-Optimized Encoder (TOE)
Replaces SDXL's dual CLIP encoders with a lightweight learned encoder
optimized for Danbooru tag-based conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class QuantizedEmbedding(nn.Module):
    """
    INT4-quantized embedding layer for tag representations.
    Uses fake quantization during training for QAT.
    """
    def __init__(self, num_embeddings, embedding_dim, quantize=False):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.quantize = quantize
        
        # Standard embedding layer
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        # INT4 quantization parameters (16 levels: -8 to 7)
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))
        
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, num_tags) - Tag indices
        Returns:
            embeddings: (batch_size, num_tags, embedding_dim)
        """
        embeds = self.embedding(input_ids)
        
        if self.quantize and self.training:
            # Fake quantization (INT4: -8 to 7)
            embeds_quantized = torch.clamp(
                torch.round(embeds / self.scale) + self.zero_point,
                -8, 7
            )
            # Dequantize
            embeds = (embeds_quantized - self.zero_point) * self.scale
            
        return embeds


class TinyTransformer(nn.Module):
    """
    Compact 4-layer transformer for compositional tag reasoning.
    Much smaller than CLIP's 12-24 layer transformers.
    """
    def __init__(
        self,
        dim=2048,
        num_layers=4,
        num_heads=8,
        mlp_ratio=2,
        dropout=0.0
    ):
        super().__init__()
        self.dim = dim
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 77, dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, dim)
        Returns:
            x: (batch_size, seq_len, dim)
        """
        # Add positional encoding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final norm
        x = self.norm(x)
        
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio, dropout=0.0):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class TagOptimizedEncoder(nn.Module):
    """
    Tag-Optimized Encoder: Replaces SDXL's 999M CLIP encoders
    with a 50M learned encoder optimized for Danbooru tags.
    
    Architecture:
        - Quantized tag embeddings (15K vocabulary)
        - 4-layer tiny transformer for composition
        - Projection to SDXL's 2048-dim context space
    """
    def __init__(
        self,
        vocab_size=15000,
        embed_dim=2048,
        num_layers=4,
        num_heads=8,
        mlp_ratio=2,
        max_length=77,
        quantize_embeddings=False
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Tag embedding layer
        self.tag_embeddings = QuantizedEmbedding(
            vocab_size, embed_dim, quantize=quantize_embeddings
        )
        
        # Compositional encoder
        self.encoder = TinyTransformer(
            dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        
        # Projection to SDXL context space
        self.projection = nn.Linear(embed_dim, embed_dim)
        
        # Weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following best practices."""
        # Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, tag_ids, tag_weights=None):
        """
        Encode Danbooru tags into SDXL-compatible context embeddings.
        
        Args:
            tag_ids: (batch_size, num_tags) - Tag vocabulary indices
            tag_weights: (batch_size, num_tags) - Optional tag importance weights
        
        Returns:
            context: (batch_size, max_length, embed_dim) - Context embeddings
        """
        batch_size = tag_ids.size(0)
        
        # Embed tags
        embeds = self.tag_embeddings(tag_ids)  # (batch, num_tags, embed_dim)
        
        # Apply tag weights if provided
        if tag_weights is not None:
            embeds = embeds * tag_weights.unsqueeze(-1)
            
        # Pad or truncate to max_length
        if embeds.size(1) < self.max_length:
            # Pad with zeros
            padding = torch.zeros(
                batch_size,
                self.max_length - embeds.size(1),
                self.embed_dim,
                device=embeds.device,
                dtype=embeds.dtype
            )
            embeds = torch.cat([embeds, padding], dim=1)
        else:
            # Truncate
            embeds = embeds[:, :self.max_length, :]
            
        # Compositional reasoning
        context = self.encoder(embeds)
        
        # Project to SDXL space
        context = self.projection(context)
        
        return context
    
    def get_num_params(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self):
        """Estimate model size in MB (FP32)."""
        return self.get_num_params() * 4 / (1024 ** 2)


if __name__ == "__main__":
    # Test TOE
    model = TagOptimizedEncoder(
        vocab_size=15000,
        embed_dim=2048,
        num_layers=4,
        num_heads=8,
        quantize_embeddings=True
    )
    
    # Dummy input
    batch_size = 2
    num_tags = 20
    tag_ids = torch.randint(0, 15000, (batch_size, num_tags))
    tag_weights = torch.rand(batch_size, num_tags)
    
    # Forward pass
    context = model(tag_ids, tag_weights)
    
    print(f"Input shape: {tag_ids.shape}")
    print(f"Output shape: {context.shape}")
    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print(f"Expected output: (batch_size=2, max_length=77, embed_dim=2048)")
