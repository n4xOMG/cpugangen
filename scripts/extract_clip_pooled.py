"""
Extract CLIP pooled embeddings for pooling head training.

This script:
1. Loads existing CLIP embeddings dataset
2. Extracts pooled embeddings from CLIP
3. Saves in format ready for pooling head training

Usage:
    python scripts/extract_clip_pooled.py
"""

import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.utils.danbooru import DanbooruTagProcessor


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--vocab',
        type=str,
        default='data/vocabulary.json',
        help='Path to vocabulary'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/clip_pooled_embeddings.pt',
        help='Output file for pooled embeddings'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10000,
        help='Number of samples to generate (default: 10k)'
    )
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tag processor
    print(f"\nLoading vocabulary from {args.vocab}...")
    tag_processor = DanbooruTagProcessor(vocab_size=15000)
    tag_processor.load_vocabulary(args.vocab)
    print(f"✓ Loaded {len(tag_processor.tag_to_id)} tags")
    
    # Load CLIP
    print("\nLoading CLIP text encoder...")
    tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer_2"
    )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder_2",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    text_encoder.eval()
    print("✓ CLIP loaded")
    
    # Generate samples from vocabulary
    print(f"\nGenerating {args.num_samples} tag combinations...")
    
    import random
    all_tags = list(tag_processor.tag_to_id.keys())
    # Remove special tokens
    all_tags = [t for t in all_tags if t not in ['<pad>', '<unk>', '<eos>']]
    
    samples = []
    for _ in tqdm(range(args.num_samples)):
        # Random number of tags (3-12)
        num_tags = random.randint(3, 12)
        tags = random.sample(all_tags, num_tags)
        samples.append(tags)
    
    print(f"✓ Generated {len(samples)} samples")
    
    # Extract CLIP pooled embeddings
    print("\nExtracting CLIP pooled embeddings...")
    
    clip_pooled_list = []
    batch_size = 32
    
    for i in tqdm(range(0, len(samples), batch_size)):
        batch_tags = samples[i:i+batch_size]
        
        # Convert to prompts (comma-separated)
        prompts = [", ".join(tags) for tags in batch_tags]
        
        # Tokenize
        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        
        # Get pooled embeddings
        with torch.no_grad():
            outputs = text_encoder(text_input_ids)
            pooled = outputs[0]  # Pooled output
        
        clip_pooled_list.append(pooled.cpu())
    
    # Concatenate all
    clip_pooled = torch.cat(clip_pooled_list, dim=0)
    
    print(f"✓ Extracted {clip_pooled.shape[0]} pooled embeddings")
    print(f"  Shape: {clip_pooled.shape}")
    
    # Save dataset
    print(f"\nSaving to {args.output}...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    torch.save({
        'tags': samples,
        'clip_pooled': clip_pooled
    }, args.output)
    
    print("✓ Saved!")
    print("\n" + "=" * 70)
    print("Ready for pooling head training!")
    print("=" * 70)
    print("\nNext step:")
    print(f"  python scripts/train_toe_pooling.py --embeddings {args.output}")


if __name__ == "__main__":
    main()
