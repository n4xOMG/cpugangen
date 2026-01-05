"""
Pre-compute CLIP embeddings from SDXL model for TOE training.
This generates real teacher embeddings for knowledge distillation.
"""

import os
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline


def load_sdxl_clip(model_name="stabilityai/stable-diffusion-xl-base-1.0", device="cuda"):
    """
    Load SDXL CLIP text encoders.
    
    Returns:
        tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2
    """
    print(f"Loading SDXL model: {model_name}")
    print(f"This will download ~7GB if not cached...")
    
    # Load pipeline (will download if needed)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
        variant="fp16" if device == "cuda" else None
    )
    
    # Extract text encoders
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder_1 = pipe.text_encoder.to(device)
    text_encoder_2 = pipe.text_encoder_2.to(device)
    
    # Set to eval mode
    text_encoder_1.eval()
    text_encoder_2.eval()
    
    print(f"✓ SDXL CLIP encoders loaded on {device}")
    
    # Free up memory from other pipeline components
    del pipe.unet, pipe.vae, pipe.scheduler
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2


def encode_prompt_sdxl(prompt, tokenizers, text_encoders, device="cuda"):
    """
    Encode a prompt using SDXL's dual CLIP encoders.
    
    Args:
        prompt: String with comma-separated tags
        tokenizers: (tokenizer_1, tokenizer_2)
        text_encoders: (text_encoder_1, text_encoder_2)
        
    Returns:
        Pooled embedding of shape (2048,) concatenated from both encoders
    """
    tokenizer_1, tokenizer_2 = tokenizers
    text_encoder_1, text_encoder_2 = text_encoders
    
    with torch.no_grad():
        # Tokenize with both tokenizers
        text_inputs_1 = tokenizer_1(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        text_inputs_2 = tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode with both text encoders
        text_embeddings_1 = text_encoder_1(
            text_inputs_1.input_ids.to(device),
            output_hidden_states=False  # We'll use last_hidden_state directly
        )
        text_embeddings_2 = text_encoder_2(
            text_inputs_2.input_ids.to(device),
            output_hidden_states=False
        )
        
        # Get the last hidden state (correct for SDXL)
        # last_hidden_state has shape (batch, seq_len, hidden_dim)
        hidden_1 = text_embeddings_1.last_hidden_state  # (1, 77, 768)
        hidden_2 = text_embeddings_2.last_hidden_state  # (1, 77, 1280)
        
        # Concatenate embeddings (SDXL combines both encoders)
        # Shape: (1, 77, 768) + (1, 77, 1280) = (1, 77, 2048)
        combined = torch.cat([hidden_1, hidden_2], dim=-1)
        
       # Normalize embeddings to prevent scale issues (CRITICAL FIX)
        # This ensures consistent magnitude with dummy random embeddings
        norm = torch.norm(combined, p=2, dim=-1, keepdim=True)
        combined = combined / (norm + 1e-8)  # L2 normalization
        
        # Return as numpy array
        return combined.squeeze(0).cpu().float().numpy()  # (77, 2048)


def generate_tag_combinations(vocabulary_path, num_samples=10000, min_tags=3, max_tags=15):
    """
    Generate random tag combinations from vocabulary.
    
    Args:
        vocabulary_path: Path to vocabulary.json
        num_samples: Number of samples to generate
        min_tags: Minimum tags per combination
        max_tags: Maximum tags per combination
        
    Returns:
        List of tag combinations (list of lists)
    """
    print(f"Loading vocabulary from {vocabulary_path}")
    with open(vocabulary_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    # Get top tags (excluding special tokens)
    if 'top_tags' in vocab_data:
        available_tags = [tag['name'] for tag in vocab_data['top_tags']]
    else:
        # Fallback: get from tag_to_id, filter special tokens
        available_tags = [
            tag for tag in vocab_data['tag_to_id'].keys()
            if tag not in ['<pad>', '<unk>', '<eos>']
        ]
    
    print(f"Available tags: {len(available_tags)}")
    print(f"Generating {num_samples} random tag combinations...")
    
    tag_combinations = []
    for _ in tqdm(range(num_samples), desc="Generating combinations"):
        num_tags = random.randint(min_tags, max_tags)
        tags = random.sample(available_tags, min(num_tags, len(available_tags)))
        tag_combinations.append(tags)
    
    return tag_combinations


def precompute_embeddings(tag_combinations, output_dir, model_name, device="cuda"):
    """
    Pre-compute CLIP embeddings for all tag combinations.
    
    Args:
        tag_combinations: List of tag lists
        output_dir: Directory to save embeddings
        model_name: SDXL model name
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SDXL CLIP encoders
    tokenizers = load_sdxl_clip(model_name, device)
    tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2 = tokenizers
    
    # Pre-allocate arrays for embeddings
    num_samples = len(tag_combinations)
    embeddings = np.zeros((num_samples, 77, 2048), dtype=np.float32)
    
    print(f"\nEncoding {num_samples} tag combinations...")
    
    # Encode all combinations
    for idx, tags in enumerate(tqdm(tag_combinations, desc="Encoding")):
        # Convert tag list to comma-separated prompt
        prompt = ", ".join(tags)
        
        # Encode with SDXL CLIP
        embedding = encode_prompt_sdxl(
            prompt,
            (tokenizer_1, tokenizer_2),
            (text_encoder_1, text_encoder_2),
            device
        )
        
        embeddings[idx] = embedding
    
    # Save embeddings as NPZ (compressed)
    embeddings_path = output_dir / "clip_embeddings.npz"
    print(f"\nSaving embeddings to {embeddings_path}")
    np.savez_compressed(embeddings_path, embeddings=embeddings)
    
    # Save metadata (tag combinations and indices)
    metadata = {
        'num_samples': num_samples,
        'tag_combinations': tag_combinations,
        'model_name': model_name,
        'embedding_shape': [77, 2048]
    }
    
    metadata_path = output_dir / "metadata.json"
    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    file_size_mb = embeddings_path.stat().st_size / (1024 * 1024)
    print(f"\n✓ Pre-computation complete!")
    print(f"  Samples: {num_samples}")
    print(f"  Embeddings shape: (77, 2048)")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Output: {output_dir}")
    
    return embeddings_path, metadata_path


def main():
    parser = argparse.ArgumentParser(description="Pre-compute CLIP embeddings for TOE training")
    parser.add_argument(
        '--vocab-path',
        type=str,
        default='data/vocabulary.json',
        help='Path to vocabulary.json'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of tag combinations to generate (default: 10000)'
    )
    parser.add_argument(
        '--min-tags',
        type=int,
        default=3,
        help='Minimum tags per combination (default: 3)'
    )
    parser.add_argument(
        '--max-tags',
        type=int,
        default=15,
        help='Maximum tags per combination (default: 15)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/clip_embeddings',
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help='SDXL model name'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Pre-computing CLIP Embeddings for TOE Training")
    print("=" * 60)
    print(f"Vocabulary: {args.vocab_path}")
    print(f"Samples: {args.num_samples}")
    print(f"Tags per sample: {args.min_tags}-{args.max_tags}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Generate tag combinations
    tag_combinations = generate_tag_combinations(
        args.vocab_path,
        num_samples=args.num_samples,
        min_tags=args.min_tags,
        max_tags=args.max_tags
    )
    
    # Pre-compute embeddings
    embeddings_path, metadata_path = precompute_embeddings(
        tag_combinations,
        args.output_dir,
        args.model_name,
        args.device
    )
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Update configs/toe_config.yaml to use pre-computed embeddings:")
    print(f"   clip_embeddings_path: '{args.output_dir}'")
    print("\n2. Re-run training:")
    print("   python scripts/train_toe.py --config configs/toe_config.yaml")
    print("\n3. Expected improvement:")
    print("   Validation loss should drop from ~1.0 to <0.1")
    print("=" * 60)


if __name__ == "__main__":
    main()
