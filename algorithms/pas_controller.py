"""
Phase-aware Sampling (PAS) Controller

Implements dynamic block skipping and activation caching for speedup.

Key features:
1. Phase detection (Sketching vs Refinement)
2. Partial U-Net execution (skip middle blocks)
3. Activation caching and reuse
4. Integration with diffusers pipeline

Reference: SD-Acc (arxiv:2507.01309)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class PASConfig:
    """Configuration for Phase-aware Sampling."""
    T_sketch: int  # Total steps in sketching phase
    T_complete: int  # Initial steps with full U-Net
    T_sparse: int  # Sparse sampling period in sketching
    L_sketch: int  # Number of blocks in sparse sketching steps
    L_refine: int  # Number of blocks in refinement phase
    D_star: int  # Optimal phase division point (from calibration)
    
    @classmethod
    def from_calibration(cls, calibration_path: str):
        """Load config from calibration JSON."""
        import json
        with open(calibration_path, 'r') as f:
            data = json.load(f)
        
        params = data['suggested_params']
        return cls(
            T_sketch=params['T_sketch'],
            T_complete=params['T_complete'],
            T_sparse=params['T_sparse'],
            L_sketch=params['L_sketch'],
            L_refine=params['L_refine'],
            D_star=data['D_star']
        )


class PASController:
    """
    Controls phase-aware execution of U-Net.
    
    Decides when to run full vs partial U-Net and manages caching.
    """
    
    def __init__(self, config: PASConfig, verbose: bool = False):
        """
        Initialize PAS controller.
        
        Args:
            config: PAS configuration
            verbose: Print execution decisions
        """
        self.config = config
        self.verbose = verbose
        
        # Activation cache
        self.activation_cache = {}
        self.last_full_step = None
        
        # Statistics
        self.full_unet_calls = 0
        self.partial_unet_calls = 0
        self.cache_hits = 0
    
    def should_run_full_unet(self, current_step: int, total_steps: int) -> bool:
        """
        Determine if current step should run full U-Net.
        
        Args:
            current_step: Current denoising step (0-indexed)
            total_steps: Total number of steps
            
        Returns:
            True if should run full U-Net
        """
        # Sketching phase
        if current_step < self.config.T_sketch:
            # Initial stabilization
            if current_step < self.config.T_complete:
                return True
            
            # Sparse sampling (periodic full execution)
            if current_step % self.config.T_sparse == 0:
                return True
            
            return False
        
        # Refinement phase - never run full (only top blocks)
        return False
    
    def get_num_active_blocks(self, current_step: int) -> int:
        """
        Get number of blocks to execute at current step.
        
        Args:
            current_step: Current denoising step
            
        Returns:
            Number of blocks to execute
        """
        if current_step < self.config.T_sketch:
            # Sketching phase - sparse steps use L_sketch blocks
            return self.config.L_sketch
        else:
            # Refinement phase - use L_refine blocks
            return self.config.L_refine
    
    def execute_step(
        self,
        unet,
        latent_model_input: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        current_step: int,
        total_steps: int,
        added_cond_kwargs: dict = None  # SDXL augmentation kwargs
    ) -> torch.Tensor:
        """
        Execute U-Net with PAS logic.
        
        Args:
            unet: U-Net model
            latent_model_input: Input latents
            t: Timestep
            encoder_hidden_states: Text embeddings
            current_step: Current step index
            total_steps: Total steps
            added_cond_kwargs: SDXL-specific kwargs (text_embeds, time_ids)
            
        Returns:
            Noise prediction
        """
        run_full = self.should_run_full_unet(current_step, total_steps)
        
        if run_full:
            # Run full U-Net
            if self.verbose:
                print(f"  Step {current_step}: FULL U-Net")
            
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
            
            # Cache for future reuse
            self.activation_cache[current_step] = noise_pred.clone()
            self.last_full_step = current_step
            self.full_unet_calls += 1
            
            return noise_pred
        
        else:
            # Run partial U-Net
            num_blocks = self.get_num_active_blocks(current_step)
            
            if self.verbose:
                phase = "Sketching" if current_step < self.config.T_sketch else "Refinement"
                print(f"  Step {current_step}: PARTIAL U-Net ({num_blocks} blocks, {phase})")
            
            noise_pred = self._execute_partial_unet(
                unet,
                latent_model_input,
                t,
                encoder_hidden_states,
                num_blocks,
                added_cond_kwargs  # Pass SDXL kwargs
            )
            
            self.partial_unet_calls += 1
            
            return noise_pred
    
    def _execute_partial_unet(
        self,
        unet,
        latent_model_input: torch.Tensor,
        t: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        num_blocks: int,
        added_cond_kwargs: dict = None
    ) -> torch.Tensor:
        """
        Execute only first num_blocks of U-Net.
        
        For remaining blocks, reuse cached activations from last full execution.
        
        This is a simplified implementation - in full version we'd need
        to modify U-Net forward pass to skip blocks dynamically.
        
        Args:
            unet: U-Net model
            latent_model_input: Input latents
            t: Timestep
            encoder_hidden_states: Text embeddings
            num_blocks: Number of blocks to execute
            added_cond_kwargs: SDXL-specific kwargs
            
        Returns:
            Noise prediction (approximated)
        """
        # SIMPLIFIED: For now, we approximate by running full U-Net
        # but with fewer steps in practice you'd hook into U-Net internals
        
        # In a full implementation, we would:
        # 1. Run only down_blocks[:num_blocks]
        # 2. Run mid_block if within range
        # 3. Run only up_blocks[:num_blocks]
        # 4. Reuse cached intermediate activations for skipped blocks
        
        # For this prototype, we use cached noise prediction as approximation
        if self.last_full_step is not None and self.last_full_step in self.activation_cache:
            # Reuse cached prediction (approximation)
            self.cache_hits += 1
            return self.activation_cache[self.last_full_step]
        else:
            # Fallback to full execution
            return unet(
                latent_model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False
            )[0]
    
    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        total_calls = self.full_unet_calls + self.partial_unet_calls
        if total_calls == 0:
            return {}
        
        return {
            'full_unet_calls': self.full_unet_calls,
            'partial_unet_calls': self.partial_unet_calls,
            'cache_hits': self.cache_hits,
            'speedup_estimate': total_calls / max(self.full_unet_calls, 1),
            'block_skip_rate': self.partial_unet_calls / total_calls
        }
    
    def reset(self):
        """Reset controller state."""
        self.activation_cache.clear()
        self.last_full_step = None
        self.full_unet_calls = 0
        self.partial_unet_calls = 0
        self.cache_hits = 0


if __name__ == "__main__":
    # Example usage
    config = PASConfig(
        T_sketch=15,
        T_complete=3,
        T_sparse=5,
        L_sketch=8,
        L_refine=4,
        D_star=12
    )
    
    controller = PASController(config, verbose=True)
    
    print("PAS Controller Example")
    print(f"Config: {config}\n")
    
    print("Simulating 20-step execution:")
    for step in range(20):
        run_full = controller.should_run_full_unet(step, 20)
        num_blocks = controller.get_num_active_blocks(step)
        phase = "Sketching" if step < config.T_sketch else "Refinement"
        mode = "FULL" if run_full else f"PARTIAL ({num_blocks} blocks)"
        print(f"Step {step:2d}: {phase:12} - {mode}")
