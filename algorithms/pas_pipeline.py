"""
PAS-enabled Pipeline

Wrapper around StableDiffusionXLPipeline with Phase-aware Sampling.

Usage:
    from algorithms.pas_pipeline import create_pas_pipeline
    
    pipeline = create_pas_pipeline(
        model_id="martineux/janku6",
        pas_config_path="configs/pas_calibration.json",
        enable_lightning=True,
        device="cpu"
    )
    
    image = pipeline(
        prompt="1girl, blue_hair",
        num_inference_steps=20
    ).images[0]
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from typing import Optional
import json

from .pas_controller import PASController, PASConfig


class PASPipeline:
    """
    SDXL Pipeline with Phase-aware Sampling.
    
    Wraps diff users pipeline and injects PAS controller into denoising loop.
    """
    
    def __init__(
        self,
        base_pipeline: StableDiffusionXLPipeline,
        pas_controller: PASController
    ):
        """
        Initialize PAS pipeline.
        
        Args:
            base_pipeline: Base SDXL pipeline
            pas_controller: PAS controller instance
        """
        self.pipeline = base_pipeline
        self.controller = pas_controller
        
        # Expose pipeline attributes
        self.device = base_pipeline.device
        self.unet = base_pipeline.unet
        self.vae = base_pipeline.vae
        self.scheduler = base_pipeline.scheduler
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        **kwargs
    ):
        """
        Generate image with PAS.
        
        Args:
            prompt: Text prompt
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            generator: Random generator
            
        Returns:
            Generated images
        """
        # Reset controller state
        self.controller.reset()
        
        # Encode prompt
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=guidance_scale > 1.0
        )
        
        # Prepare latents
        latents = torch.randn(
            (1, 4, 128, 128),
            generator=generator,
            device=self.device,
            dtype=torch.float32
        )
        
        # Set timesteps
        self.pipeline.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.pipeline.scheduler.timesteps
        
        # Denoising loop with PAS
        for step_idx, t in enumerate(timesteps):
            # Expand for CFG
            latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
            
            # Prepare encoder hidden states
            encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds]) if guidance_scale > 1.0 else prompt_embeds
            
            # PAS-controlled U-Net execution
            noise_pred = self.controller.execute_step(
                unet=self.pipeline.unet,
                latent_model_input=latent_model_input,
                t=t,
                encoder_hidden_states=encoder_hidden_states,
                current_step=step_idx,
                total_steps=num_inference_steps
            )
            
            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.pipeline.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents
        images = self.pipeline.vae.decode(latents / self.pipeline.vae.config.scaling_factor).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # Convert to PIL
        from PIL import Image
        images = [Image.fromarray((img * 255).astype("uint8")) for img in images]
        
        # Return in diffusers format
        from dataclasses import dataclass
        
        @dataclass
        class PipelineOutput:
            images: list
        
        return PipelineOutput(images=images)
    
    def get_statistics(self):
        """Get PAS execution statistics."""
        return self.controller.get_statistics()


def create_pas_pipeline(
    model_id: str,
    pas_config_path: str,
    enable_lightning: bool = False,
    lightning_steps: int = 4,
    device: str = "cpu",
    verbose: bool = True
):
    """
    Create PAS-enabled pipeline.
    
    Args:
        model_id: Model ID or path
        pas_config_path: Path to PAS calibration JSON
        enable_lightning: Load Lightning LoRA
        lightning_steps: Steps for Lightning (4 or 8)
        device: Device to use
        verbose: Print info
        
    Returns:
        PAS-enabled pipeline
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Creating PAS Pipeline")
        print(f"{'='*60}\n")
        print(f"Model: {model_id}")
        print(f"PAS config: {pas_config_path}")
        print(f"Lightning LoRA: {'Enabled' if enable_lightning else 'Disabled'}")
        print(f"Device: {device}\n")
    
    # Load base pipeline
    if verbose:
        print("Loading base model...")
    
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        use_safetensors=True
    )
    pipeline = pipeline.to(device)
    
    # Load Lightning LoRA if enabled
    if enable_lightning:
        if verbose:
            print("Loading Lightning LoRA...")
        
        try:
            lightning_ckpt = hf_hub_download(
                "ByteDance/SDXL-Lightning",
                f"sdxl_lightning_{lightning_steps}step_lora.safetensors"
            )
            pipeline.load_lora_weights(lightning_ckpt)
            pipeline.fuse_lora()
            
            # Use proper scheduler for Lightning
            pipeline.scheduler = EulerDiscreteScheduler.from_config(
                pipeline.scheduler.config,
                timestep_spacing="trailing"
            )
            
            if verbose:
                print("✅ Lightning LoRA loaded")
        except Exception as e:
            if verbose:
                print(f"⚠️  Lightning LoRA failed: {e}")
    
    # Load PAS config
    if verbose:
        print("Loading PAS configuration...")
    
    pas_config = PASConfig.from_calibration(pas_config_path)
    
    if verbose:
        print(f"✅ PAS config loaded:")
        print(f"   T_sketch: {pas_config.T_sketch}")
        print(f"   T_complete: {pas_config.T_complete}")
        print(f"   T_sparse: {pas_config.T_sparse}")
        print(f"   D*: {pas_config.D_star}\n")
    
    # Create PAS controller
    controller = PASController(pas_config, verbose=False)
    
    # Wrap in PAS pipeline
    pas_pipeline = PASPipeline(pipeline, controller)
    
    if verbose:
        print("✅ PAS Pipeline ready!\n")
    
    return pas_pipeline


if __name__ == "__main__":
    print("PAS Pipeline Module")
    print("Use create_pas_pipeline() to create PAS-enabled pipeline")
