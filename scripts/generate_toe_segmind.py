#!/usr/bin/env python3
"""
Generate images using TOE + Segmind SSD-1B.

This combines:
- TOE (169M params) for text encoding
- Segmind SSD-1B (1.33B params) for UNet
- Standard SDXL VAE for decoding

Total optimization: ~70% parameter reduction vs full SDXL.

Usage:
    python scripts/generate_toe_segmind.py \
        --prompt "1girl, anime, blue_eyes, smile" \
        --output output.png
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from PIL import Image

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hqpd.models.toe import TagOptimizedEncoder
from hqpd.utils.danbooru import DanbooruTagProcessor


def create_toe_segmind_pipeline(
    toe_checkpoint: str,
    vocab_path: str,
    device: str = "cuda",
):
    """
    Create optimized pipeline with TOE + Segmind.
    
    Args:
        toe_checkpoint: Path to trained TOE model
        vocab_path: Path to tag vocabulary
        device: Device to use
        
    Returns:
        Dictionary with all pipeline components
    """
    print("=" * 70)
    print("Creating TOE + Segmind Pipeline")
    print("=" * 70)
    
    # Load TOE
    print("\n1. Loading TOE text encoder...")
    tag_processor = DanbooruTagProcessor(vocab_size=15000)
    tag_processor.load_vocabulary(vocab_path)
    
    checkpoint = torch.load(toe_checkpoint, map_location=device)
    has_pooling = any('pooling_head' in k for k in checkpoint['model_state_dict'].keys())
    
    toe_model = TagOptimizedEncoder(
        vocab_size=15000,
        embed_dim=2048,
        num_layers=4,
        num_heads=8,
        mlp_ratio=2,
        max_length=77,
        quantize_embeddings=False,
        enable_pooling=has_pooling,
    ).to(device)
    
    toe_model.load_state_dict(checkpoint['model_state_dict'])
    toe_model.eval()
    
    toe_params = sum(p.numel() for p in toe_model.parameters())
    print(f"   ✓ TOE: {toe_params:,} parameters ({toe_params/1e6:.1f}M)")
    
    # Load Segmind UNet
    print("\n2. Loading Segmind SSD-1B UNet...")
    from diffusers import UNet2DConditionModel
    
    unet = UNet2DConditionModel.from_pretrained(
        "segmind/SSD-1B",
        subfolder="unet",
        torch_dtype=torch.float16,
    ).to(device)
    unet.eval()
    
    unet_params = sum(p.numel() for p in unet.parameters())
    print(f"   ✓ UNet: {unet_params:,} parameters ({unet_params/1e9:.2f}B)")
    
    # Load VAE
    print("\n3. Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="vae",
        torch_dtype=torch.float16,
    ).to(device)
    vae.eval()
    
    print("   ✓ VAE loaded")
    
    # Load CLIP for pooled embeddings (hybrid mode)
    # Segmind was trained with CLIP, so we use CLIP pooled + TOE context
    print("\n4. Loading CLIP for pooled embeddings (hybrid mode)...")
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection
    
    tokenizer = CLIPTokenizer.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="tokenizer_2"
    )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
    ).to(device)
    text_encoder.eval()
    
    print("   ✓ CLIP loaded (for pooled embeddings)")
    
    # Scheduler
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler"
    )
    
    total_params = toe_params + unet_params
    print(f"\n✓ Total optimized parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   vs Full SDXL: ~3.5B (reduction: {(1 - total_params/3.5e9)*100:.1f}%)")
    print(f"\n⚠️  Using hybrid mode: TOE context + CLIP pooled")
    
    return {
        "toe_model": toe_model,
        "tag_processor": tag_processor,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "unet": unet,
        "vae": vae,
        "scheduler": scheduler,
        "device": device,
    }


@torch.no_grad()
def generate_image(
    pipeline: dict,
    prompt: str,
    negative_prompt: str = "lowres, bad_anatomy, bad_hands, text, error",
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = None,
    width: int = 1024,
    height: int = 1024,
):
    """
    Generate image using TOE + Segmind.
    
    Args:
        pipeline: Pipeline components dict
        prompt: Danbooru tags (comma-separated)
        negative_prompt: Negative tags
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG scale
        seed: Random seed
        width, height: Image dimensions
        
    Returns:
        PIL Image
    """
    device = pipeline["device"]
    
    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
    
    # Encode prompt with TOE (for context embeddings only)
    print("\n🎨 Encoding prompt...")
    tags = [tag.strip() for tag in prompt.split(',')]
    tag_ids, tag_weights = pipeline["tag_processor"].encode(tags, max_length=77)
    tag_ids = torch.tensor([tag_ids], dtype=torch.long).to(device)
    tag_weights = torch.tensor([tag_weights], dtype=torch.float32).to(device)
    
    # Get TOE context embeddings
    with torch.no_grad():
        context_embeds = pipeline["toe_model"](tag_ids, tag_weights, return_pooled=False)
        context_embeds = context_embeds.to(dtype=torch.float16)
    
    # Get CLIP pooled embeddings (Segmind expects these)
    text_inputs = pipeline["tokenizer"](
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        pooled_embeds = pipeline["text_encoder"](
            text_inputs.input_ids.to(device)
        )[0]  # text_embeds (pooled)
    
    # Negative prompt
    if negative_prompt:
        neg_tags = [tag.strip() for tag in negative_prompt.split(',')]
        neg_tag_ids, neg_tag_weights = pipeline["tag_processor"].encode(neg_tags, max_length=77)
        neg_tag_ids = torch.tensor([neg_tag_ids], dtype=torch.long).to(device)
        neg_tag_weights = torch.tensor([neg_tag_weights], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            neg_context_embeds = pipeline["toe_model"](neg_tag_ids, neg_tag_weights, return_pooled=False)
            neg_context_embeds = neg_context_embeds.to(dtype=torch.float16)
        
        neg_text_inputs = pipeline["tokenizer"](
            negative_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            neg_pooled_embeds = pipeline["text_encoder"](
                neg_text_inputs.input_ids.to(device)
            )[0]
    else:
        neg_context_embeds = torch.zeros_like(context_embeds)
        neg_pooled_embeds = torch.zeros_like(pooled_embeds)
    
    # Concatenate for CFG
    encoder_hidden_states = torch.cat([neg_context_embeds, context_embeds])
    
    # SDXL time conditioning
    original_size = (height, width)
    target_size = (height, width)
    crops_coords = (0, 0)
    
    add_time_ids = torch.tensor([
        list(original_size) + list(crops_coords) + list(target_size)
    ], dtype=torch.float16, device=device)
    add_time_ids = torch.cat([add_time_ids, add_time_ids])  # For CFG
    
    add_text_embeds = torch.cat([neg_pooled_embeds, pooled_embeds])
    
    # Initialize latents
    print(f"🖼️  Generating {width}x{height} image...")
    latents = torch.randn(
        (1, 4, height // 8, width // 8),
        device=device,
        dtype=torch.float16
    )
    latents = latents * pipeline["scheduler"].init_noise_sigma
    
    # Denoising loop
    pipeline["scheduler"].set_timesteps(num_inference_steps, device=device)
    
    from tqdm import tqdm
    for i, t in enumerate(tqdm(pipeline["scheduler"].timesteps, desc="Denoising")):
        # Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipeline["scheduler"].scale_model_input(latent_model_input, t)
        
        # Predict noise
        noise_pred = pipeline["unet"](
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={
                "text_embeds": add_text_embeds,
                "time_ids": add_time_ids,
            },
        ).sample
        
        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # Step
        latents = pipeline["scheduler"].step(noise_pred, t, latents).prev_sample
    
    # Decode
    print("🎨 Decoding image...")
    latents = latents / pipeline["vae"].config.scaling_factor
    image = pipeline["vae"].decode(latents).sample
    
    # Convert to PIL with proper NaN handling
    image = (image / 2 + 0.5).clamp(0, 1)
    
    # Replace NaN with 0 (prevents invalid cast warning)
    image = torch.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)
    
    return image



def main():
    parser = argparse.ArgumentParser(description="Generate with TOE + Segmind")
    
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Danbooru tags (comma-separated)",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="lowres, bad_anatomy, bad_hands",
        help="Negative prompt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--toe_checkpoint",
        type=str,
        default="checkpoints/toe/toe_with_pooling_best.pt",
        help="TOE checkpoint path",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        default="data/vocabulary.json",
        help="Vocabulary path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=25,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.5,
        help="Guidance scale",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = create_toe_segmind_pipeline(
        toe_checkpoint=args.toe_checkpoint,
        vocab_path=args.vocab,
        device=args.device,
    )
    
    # Generate
    image = generate_image(
        pipeline=pipeline,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg_scale,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )
    
    # Save
    image.save(args.output)
    print(f"\n✓ Saved to: {args.output}")


if __name__ == "__main__":
    main()
