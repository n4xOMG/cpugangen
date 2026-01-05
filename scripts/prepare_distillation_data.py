#!/usr/bin/env python3
"""
Prepare latent dataset for UNet distillation.

This script:
1. Loads images and encodes them with VAE
2. Encodes prompts with CLIP/TOE
3. Saves latent-embedding pairs for distillation

Usage:
    python scripts/prepare_distillation_data.py \
        --image_dir data/images \
        --output_dir data/latents \
        --num_samples 10000
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Prepare distillation data")
    
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/images",
        help="Directory with training images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/latents",
        help="Output directory for latent files",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="JSON file with prompts (optional)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples to prepare",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL model ID",
    )
    parser.add_argument(
        "--use_toe",
        action="store_true",
        help="Use TOE for text encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Preparing Distillation Data")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load VAE
    print("\n1. Loading VAE...")
    from diffusers import AutoencoderKL
    
    vae = AutoencoderKL.from_pretrained(
        args.model_id,
        subfolder="vae",
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device)
    vae.eval()
    print("   ✓ VAE loaded")
    
    # Load text encoders
    print("\n2. Loading text encoders...")
    
    if args.use_toe:
        from hqpd.models.toe import TagOptimizedEncoder
        from hqpd.utils.danbooru import DanbooruTagProcessor
        
        # Load TOE
        tag_processor = DanbooruTagProcessor(vocab_size=15000)
        tag_processor.load_vocabulary("data/vocabulary.json")
        
        toe_model = TagOptimizedEncoder(
            vocab_size=15000,
            embed_dim=2048,
            enable_pooling=True,
        ).to(args.device)
        
        checkpoint = torch.load("checkpoints/toe/toe_with_pooling_best.pt", map_location=args.device)
        toe_model.load_state_dict(checkpoint['model_state_dict'])
        toe_model.eval()
        
        print("   ✓ TOE loaded")
        text_encoder = None
        text_encoder_2 = None
    else:
        # Load CLIP encoders
        tokenizer = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(args.model_id, subfolder="tokenizer_2")
        
        text_encoder = CLIPTextModel.from_pretrained(
            args.model_id, subfolder="text_encoder",
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        ).to(args.device)
        
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.model_id, subfolder="text_encoder_2",
            torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        ).to(args.device)
        
        text_encoder.eval()
        text_encoder_2.eval()
        
        print("   ✓ CLIP text encoders loaded")
        toe_model = None
        tag_processor = None
    
    # Find images or generate prompts
    print("\n3. Setting up data...")
    
    image_dir = Path(args.image_dir)
    if image_dir.exists():
        image_files = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        image_files = image_files[:args.num_samples]
        print(f"   Found {len(image_files)} images")
    else:
        print(f"   ⚠️  Image dir not found, generating random latents")
        image_files = None
    
    # Load or generate prompts
    prompts = []
    if args.prompts_file and Path(args.prompts_file).exists():
        with open(args.prompts_file) as f:
            prompts = json.load(f)
        print(f"   Loaded {len(prompts)} prompts")
    else:
        # Generate default anime prompts
        base_prompts = [
            "1girl, solo, long_hair, blue_eyes, smile",
            "2girls, yuri, romantic, sunset",
            "1boy, solo, black_hair, serious",
            "scenery, landscape, mountains, anime_style",
            "magical_girl, transformation, sparkles",
        ]
        prompts = [base_prompts[i % len(base_prompts)] for i in range(args.num_samples)]
        print(f"   Generated {len(prompts)} default prompts")
    
    # Process samples
    print("\n4. Processing samples...")
    
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    for idx in tqdm(range(min(args.num_samples, len(prompts)))):
        prompt = prompts[idx]
        
        # Get latent
        if image_files and idx < len(image_files):
            # Encode real image
            img = Image.open(image_files[idx]).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                latent = vae.encode(img_tensor.half() if args.device == "cuda" else img_tensor).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
        else:
            # Random latent
            latent = torch.randn(1, 4, 128, 128, device=args.device)
        
        # Get text embeddings
        if args.use_toe:
            tags = [t.strip() for t in prompt.split(',')]
            tag_ids, tag_weights = tag_processor.encode(tags, max_length=77)
            tag_ids = torch.tensor([tag_ids], dtype=torch.long).to(args.device)
            tag_weights = torch.tensor([tag_weights], dtype=torch.float32).to(args.device)
            
            with torch.no_grad():
                encoder_hidden_states, text_embeds = toe_model(
                    tag_ids, tag_weights, return_pooled=True
                )
            encoder_hidden_states = encoder_hidden_states.squeeze(0)
            text_embeds = text_embeds.squeeze(0)
        else:
            # Use CLIP
            text_inputs = tokenizer(
                prompt, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            )
            text_inputs_2 = tokenizer_2(
                prompt, padding="max_length", max_length=77,
                truncation=True, return_tensors="pt"
            )
            
            with torch.no_grad():
                enc1 = text_encoder(text_inputs.input_ids.to(args.device))
                enc2 = text_encoder_2(text_inputs_2.input_ids.to(args.device))
            
            # Concatenate hidden states
            encoder_hidden_states = torch.cat([
                enc1.last_hidden_state,
                enc2.last_hidden_state
            ], dim=-1).squeeze(0)
            text_embeds = enc2.text_embeds.squeeze(0)
        
        # Time IDs (SDXL conditioning)
        original_size = (1024, 1024)
        crops_coords_top_left = (0, 0)
        target_size = (1024, 1024)
        
        time_ids = torch.tensor([
            original_size[0], original_size[1],
            crops_coords_top_left[0], crops_coords_top_left[1],
            target_size[0], target_size[1],
        ], dtype=torch.float32)
        
        # Save
        save_data = {
            "latent": latent.squeeze(0).cpu(),
            "encoder_hidden_states": encoder_hidden_states.cpu(),
            "text_embeds": text_embeds.cpu(),
            "time_ids": time_ids,
            "prompt": prompt,
        }
        
        torch.save(save_data, output_dir / f"latent_{idx:06d}.pt")
    
    print("\n" + "=" * 70)
    print("✓ Data Preparation Complete!")
    print("=" * 70)
    print(f"\nSaved {args.num_samples} samples to: {output_dir}")
    print("\nNext: Run distillation training:")
    print(f"  python scripts/train_distilled_unet.py --data_dir {output_dir}")


if __name__ == "__main__":
    main()
