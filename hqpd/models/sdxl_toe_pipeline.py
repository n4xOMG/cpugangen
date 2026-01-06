"""
Custom SDXL Pipeline using TOE (Tag-Optimized Encoder) instead of CLIP.

This pipeline replaces SDXL's dual CLIP text encoders with the lightweight
TOE model, enabling faster text encoding for anime image generation.

Supports two modes:
- Hybrid: TOE context + CLIP pooled (Option B - quick testing)
- Full: TOE context + TOE pooled (Option A - requires pooling head)
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput


class StableDiffusionXLTOEPipeline(StableDiffusionXLPipeline):
    """
    SDXL pipeline using TOE (Tag-Optimized Encoder) instead of CLIP.
    
    Note: TOE components (toe_model, tag_processor, use_hybrid) should be set
    after initialization using set_toe_components() method.
    """
    
    def __init__(self, **kwargs):
        # Initialize parent pipeline with standard components
        super().__init__(**kwargs)
        
        # TOE components (set later via set_toe_components)
        self.toe_model = None
        self.tag_processor = None
        self.use_hybrid = True
    
    def set_toe_components(self, toe_model, tag_processor, use_hybrid=True):
        """
        Set TOE-specific components after pipeline initialization.
        
        Args:
            toe_model: Trained TagOptimizedEncoder model
            tag_processor: DanbooruTagProcessor for tokenization
            use_hybrid: If True, use CLIP for pooled embeddings (Option B)
        """
        self.toe_model = toe_model
        self.tag_processor = tag_processor
        self.use_hybrid = use_hybrid
        
        # In full mode, can remove CLIP to save memory
        if not use_hybrid:
            print("⚠️  Full TOE mode: Removing CLIP encoders to save memory")
            self.text_encoder = None
            self.text_encoder_2 = None
        else:
            print("✓ Hybrid mode: Using TOE context + CLIP pooled")
    
    def _parse_tags(self, prompt: Union[str, List[str]]) -> List[str]:
        """
        Parse Danbooru tags from prompt.
        
        Args:
            prompt: Comma-separated tag string or list of strings
            
        Returns:
            List of cleaned tags
        """
        if isinstance(prompt, str):
            tags = [tag.strip() for tag in prompt.split(',') if tag.strip()]
        elif isinstance(prompt, list):
            # Batch of prompts
            tags = []
            for p in prompt:
                tags.extend([tag.strip() for tag in p.split(',') if tag.strip()])
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        
        return tags
    
    def _encode_with_toe(
        self,
        prompt: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode prompt using TOE.
        
        Args:
            prompt: Tag-based prompt (comma-separated)
            device: Device to use
            
        Returns:
            (context_embeddings, pooled_embeddings)
            - context_embeddings: (1, 77, 2048)
            - pooled_embeddings: (1, 1280) or None if TOE doesn't support
        """
        # Parse tags
        tags = self._parse_tags(prompt)
        
        # Encode with tag processor
        tag_ids, tag_weights = self.tag_processor.encode(tags, max_length=77)
        tag_ids = torch.tensor([tag_ids], dtype=torch.long).to(device)
        tag_weights = torch.tensor([tag_weights], dtype=torch.float32).to(device)
        
        # Forward through TOE
        with torch.no_grad():
            # Check if TOE supports pooled embeddings
            if hasattr(self.toe_model, 'pooling_head') and self.toe_model.pooling_head is not None:
                context_embeds, pooled_embeds = self.toe_model(
                    tag_ids, tag_weights, return_pooled=True
                )
            else:
                context_embeds = self.toe_model(tag_ids, tag_weights)
                pooled_embeds = None
        
        return context_embeds, pooled_embeds
    
    def _encode_pooled_with_clip(
        self,
        prompt: str,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Fallback: Encode pooled embeddings using CLIP (hybrid mode).
        
        Args:
            prompt: Prompt string
            device: Device to use
            
        Returns:
            pooled_embeddings: (1, 1280)
        """
        if self.text_encoder_2 is None:
            raise RuntimeError(
                "CLIP text_encoder_2 not available. "
                "For full TOE mode, ensure TOE has pooling_head."
            )
        
        # Tokenize for CLIP
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        
        # Get CLIP pooled output
        with torch.no_grad():
            prompt_embeds = self.text_encoder_2(
                text_input_ids,
                output_hidden_states=True,
            )
            # SDXL uses pooled output from second text encoder
            pooled_prompt_embeds = prompt_embeds[0]
        
        return pooled_prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
    ):
        """
        Override encode_prompt to use TOE instead of CLIP.
        
        This method maintains compatibility with the original SDXL pipeline
        while using TOE for text encoding.
        """
        device = device or self._execution_device
        
        # Handle prompt preprocessing
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        
        # If embeddings are provided, use them directly
        if prompt_embeds is not None and pooled_prompt_embeds is not None:
            return (prompt_embeds, negative_prompt_embeds, 
                   pooled_prompt_embeds, negative_pooled_prompt_embeds)
        
        # Encode positive prompt with TOE
        if isinstance(prompt, list):
            # For simplicity, encode first prompt (extend for batch later)
            prompt_str = prompt[0]
        else:
            prompt_str = prompt
        
        context_embeds, pooled_embeds = self._encode_with_toe(prompt_str, device)
        
        # If TOE doesn't have pooling, use CLIP (hybrid mode)
        if pooled_embeds is None and self.use_hybrid:
            pooled_embeds = self._encode_pooled_with_clip(prompt_str, device)
        elif pooled_embeds is None:
            raise RuntimeError(
                "TOE model doesn't support pooled embeddings and hybrid mode is disabled. "
                "Either enable hybrid mode or train TOE with pooling head."
            )
        
        # Encode negative prompt
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = ""
            
            if isinstance(negative_prompt, str):
                negative_prompt_str = negative_prompt
            elif isinstance(negative_prompt, list):
                negative_prompt_str = negative_prompt[0]
            else:
                negative_prompt_str = ""
            
            # Encode negative with TOE
            negative_context_embeds, negative_pooled_embeds = self._encode_with_toe(
                negative_prompt_str if negative_prompt_str else "lowres, bad anatomy, bad hands, text, error, missing fingers",
                device
            )
            
            # Fallback to CLIP for negative pooled if needed
            if negative_pooled_embeds is None and self.use_hybrid:
                negative_pooled_embeds = self._encode_pooled_with_clip(
                    negative_prompt_str if negative_prompt_str else "lowres, bad anatomy",
                    device
                )
            
            # Concatenate for classifier-free guidance
            context_embeds = torch.cat([negative_context_embeds, context_embeds])
            pooled_embeds = torch.cat([negative_pooled_embeds, pooled_embeds])
        
        # Repeat for num_images_per_prompt
        if num_images_per_prompt > 1:
            # Duplicate embeddings for each image
            bs_embed, seq_len, _ = context_embeds.shape
            context_embeds = context_embeds.repeat(1, num_images_per_prompt, 1)
            context_embeds = context_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            
            bs_embed, embed_dim = pooled_embeds.shape
            pooled_embeds = pooled_embeds.repeat(1, num_images_per_prompt)
            pooled_embeds = pooled_embeds.view(bs_embed * num_images_per_prompt, embed_dim)
        
        # Return in SDXL format
        return (
            context_embeds,      # prompt_embeds
            None,                # negative_prompt_embeds (already concatenated)
            pooled_embeds,       # pooled_prompt_embeds  
            None,                # negative_pooled_prompt_embeds (already concatenated)
        )


def create_toe_pipeline(
    toe_checkpoint_path: str,
    vocab_path: str,
    sdxl_model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "cuda",
    use_hybrid: bool = True,
    torch_dtype = None,  # Auto-detect based on device
    enable_caching: bool = None,  # Auto: True for CPU, False for GPU
    cache_size: int = 128,
):
    """
    Create SDXL pipeline with TOE text encoder.
    
    This function uses a simpler approach: load standard SDXL pipeline,
    then monkey-patch the encode_prompt method to use TOE.
    
    Args:
        toe_checkpoint_path: Path to trained TOE checkpoint
        vocab_path: Path to vocabulary.json
        sdxl_model_path: HuggingFace model ID or path to .safetensors
        device: Device to use
        use_hybrid: Use hybrid mode (TOE context + CLIP pooled)
        torch_dtype: Data type for pipeline (None = auto: FP32 for CPU, FP16 for CUDA)
        enable_caching: Enable deterministic operation caching (None = auto: True for CPU)
        cache_size: Maximum cache size (number of entries)
        
    Returns:
        SDXL pipeline with TOE encoding (as monkey-patched method)
    """
    # Auto-detect dtype based on device
    if torch_dtype is None:
        torch_dtype = torch.float32 if device == "cpu" else torch.float16
        print(f"   Auto-selected dtype: {torch_dtype} for device={device}")
    
    # Auto-detect caching based on device
    if enable_caching is None:
        enable_caching = (device == "cpu")
        print(f"   Auto-selected caching: {enable_caching} for device={device}")
    
    from hqpd.models.toe import TagOptimizedEncoder
    from hqpd.utils.danbooru import DanbooruTagProcessor
    
    print("🚀 Creating SDXL-TOE Pipeline")
    print("=" * 70)
    
    # Load tag processor
    print(f"\n1. Loading vocabulary from {vocab_path}...")
    tag_processor = DanbooruTagProcessor(vocab_size=15000)
    tag_processor.load_vocabulary(vocab_path)
    print(f"   ✓ Loaded {len(tag_processor.tag_to_id)} tags")
    
    # Load TOE model
    print(f"\n2. Loading TOE from {toe_checkpoint_path}...")
    
    # Check if checkpoint has pooling head
    checkpoint = torch.load(toe_checkpoint_path, map_location=device)
    has_pooling = any('pooling_head' in k for k in checkpoint['model_state_dict'].keys())
    
    if has_pooling:
        print("   → Detected pooling head in checkpoint")
    
    toe_model = TagOptimizedEncoder(
        vocab_size=15000,
        embed_dim=2048,
        num_layers=4,
        num_heads=8,
        mlp_ratio=2,
        max_length=77,
        quantize_embeddings=False,
        enable_pooling=has_pooling  # Auto-detect
    ).to(device)
    
    toe_model.load_state_dict(checkpoint['model_state_dict'])
    toe_model.eval()
    
    print(f"   ✓ TOE loaded: {toe_model.get_num_params():,} parameters")
    print(f"   ✓ Model size: {toe_model.get_model_size_mb():.2f} MB")
    if has_pooling:
        print(f"   ✓ Pooling head enabled (full TOE mode available)")
    
    # Load standard SDXL pipeline
    print(f"\n3. Loading SDXL pipeline from {sdxl_model_path}...")
    
    if sdxl_model_path.endswith('.safetensors'):
        pipeline = StableDiffusionXLPipeline.from_single_file(
            sdxl_model_path,
            torch_dtype=torch_dtype,
        )
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            sdxl_model_path,
            torch_dtype=torch_dtype,
            variant="fp16" if torch_dtype == torch.float16 else None,
        )
    
    pipeline = pipeline.to(device)
    print("   ✓ SDXL components loaded")
    
    # Store TOE components as attributes
    pipeline.toe_model = toe_model
    pipeline.tag_processor = tag_processor
    pipeline.use_hybrid = use_hybrid
    
    # Store cache configuration
    pipeline.enable_caching = enable_caching
    pipeline.cache_size = cache_size
    
    # Initialize global cache if caching enabled
    if enable_caching:
        from hqpd.optimization import get_global_cache
        cache = get_global_cache(max_size=cache_size, enabled=True)
        print(f"\n   ✓ Deterministic caching enabled (cache size: {cache_size})")

    
    # Save original encode_prompt method
    pipeline._original_encode_prompt = pipeline.encode_prompt
    
    # Create TOE-based encode_prompt as a bound method
    def toe_encode_prompt(
        self,
        prompt,
        device=None,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
        **kwargs
    ):
        """TOE-based prompt encoding (monkey-patched method)."""
        device = device or self._execution_device
        
        # Parse tags from prompt
        if isinstance(prompt, str):
            tags = [tag.strip() for tag in prompt.split(',') if tag.strip()]
        else:
            tags = [tag.strip() for tag in prompt[0].split(',') if tag.strip()]
        
        # Encode with TOE
        tag_ids, tag_weights = self.tag_processor.encode(tags, max_length=77)
        tag_ids = torch.tensor([tag_ids], dtype=torch.long).to(device)
        tag_weights = torch.tensor([tag_weights], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            context_embeds = self.toe_model(tag_ids, tag_weights)
        
        # Cast to pipeline dtype (float16 for GPU)
        if hasattr(self.unet.config, 'dtype'):
            target_dtype = self.unet.dtype
        else:
            target_dtype = next(self.unet.parameters()).dtype
        context_embeds = context_embeds.to(dtype=target_dtype)
        
        # Get pooled embeddings from CLIP (hybrid mode)
        if self.use_hybrid:
            text_inputs = self.tokenizer_2(
                prompt if isinstance(prompt, str) else prompt[0],
                padding="max_length",
                max_length=self.tokenizer_2.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            with torch.no_grad():
                pooled_embeds = self.text_encoder_2(text_input_ids, output_hidden_states=True)[0]
        else:
            # Would need pooling head in TOE
            raise NotImplementedError("Full TOE mode needs pooling head")
        
        # DON'T concatenate - SDXL pipeline does that
        # Just prepare the embeddings separately
        if do_classifier_free_guidance:
            neg_prompt = negative_prompt if negative_prompt else ""
            if isinstance(neg_prompt, str):
                neg_tags = [tag.strip() for tag in neg_prompt.split(',') if tag.strip()]
            else:
                neg_tags = []
            
            if len(neg_tags) == 0:
                neg_tags = ["lowres", "bad_anatomy", "bad_hands"]
            
            neg_tag_ids, neg_tag_weights = self.tag_processor.encode(neg_tags, max_length=77)
            neg_tag_ids = torch.tensor([neg_tag_ids], dtype=torch.long).to(device)
            neg_tag_weights = torch.tensor([neg_tag_weights], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                neg_context_embeds = self.toe_model(neg_tag_ids, neg_tag_weights)
            
            # Cast to pipeline dtype
            neg_context_embeds = neg_context_embeds.to(dtype=target_dtype)
            
            # Negative pooled from CLIP
            if self.use_hybrid:
                neg_text_inputs = self.tokenizer_2(
                    neg_prompt if neg_prompt else "",
                    padding="max_length",
                    max_length=self.tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                neg_text_input_ids = neg_text_inputs.input_ids.to(device)
                with torch.no_grad():
                    neg_pooled_embeds = self.text_encoder_2(neg_text_input_ids, output_hidden_states=True)[0]
        else:
            neg_context_embeds = None
            neg_pooled_embeds = None
        
        # Repeat for num_images_per_prompt
        if num_images_per_prompt > 1:
            bs_embed, seq_len, _ = context_embeds.shape
            context_embeds = context_embeds.repeat(1, num_images_per_prompt, 1)
            context_embeds = context_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            
            bs_embed, embed_dim = pooled_embeds.shape
            pooled_embeds = pooled_embeds.repeat(1, num_images_per_prompt)
            pooled_embeds = pooled_embeds.view(bs_embed * num_images_per_prompt, embed_dim)
            
            if neg_context_embeds is not None:
                bs_embed, seq_len, _ = neg_context_embeds.shape
                neg_context_embeds = neg_context_embeds.repeat(1, num_images_per_prompt, 1)
                neg_context_embeds = neg_context_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
                
                bs_embed, embed_dim = neg_pooled_embeds.shape
                neg_pooled_embeds = neg_pooled_embeds.repeat(1, num_images_per_prompt)
                neg_pooled_embeds = neg_pooled_embeds.view(bs_embed * num_images_per_prompt, embed_dim)
        
        # Return in SDXL format (separate positive and negative)
        return context_embeds, neg_context_embeds, pooled_embeds, neg_pooled_embeds
    
    # Monkey-patch the method
    import types
    pipeline.encode_prompt = types.MethodType(toe_encode_prompt, pipeline)
    
    print(f"\n4. TOE integration complete!")
    if use_hybrid:
        print("   ✓ Hybrid mode: Using TOE context + CLIP pooled")
    else:
        print("   ⚠️  Full TOE mode: Not yet implemented (needs pooling head)")
    
    print("\n" + "=" * 70)
    print("✓ TOE-SDXL Pipeline Ready")
    print("=" * 70)
    
    return pipeline
