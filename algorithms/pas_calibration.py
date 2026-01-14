"""
Phase-aware Sampling (PAS) - Calibration Module

Implements shift score analysis and phase division from SD-Acc paper.

Key steps:
1. Run prompts through model
2. Extract activations from upsampling blocks
3. Compute shift scores between consecutive timesteps
4. Find optimal phase division point D* using K-means

Reference: SD-Acc (arxiv:2507.01309)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json


class ShiftScoreAnalyzer:
    """
    Analyzes activation shifts to find optimal phase division point.
    """
    
    def __init__(self, device="cpu", verbose=True):
        """
        Initialize analyzer.
        
        Args:
            device: Device to run on
            verbose: Print progress
        """
        self.device = device
        self.verbose = verbose
        self.shift_scores = []
        self.activations_history = []
    
    def hook_upsampling_blocks(self, unet):
        """
        Register hooks to capture upsampling block activations.
        
        Args:
            unet: U-Net model
            
        Returns:
            List of hook handles
        """
        self.hooked_activations = {}
        handles = []
        
        # Hook into upsampling blocks (where shift is most visible)
        for i, up_block in enumerate(unet.up_blocks):
            def make_hook(block_idx):
                def hook(module, input, output):
                    # Store input to upsampling block
                    self.hooked_activations[f'up_block_{block_idx}'] = input[0].detach().cpu()
                return hook
            
            handle = up_block.register_forward_hook(make_hook(i))
            handles.append(handle)
        
        return handles
    
    def compute_shift_score(
        self,
        activations_curr: torch.Tensor,
        activations_prev: torch.Tensor
    ) -> float:
        """
        Compute normalized L2 shift score between consecutive activations.
        
        Formula: ||x_t - x_{t-1}||_2 / ||x_t||_2
        
        Args:
            activations_curr: Current step activations
            activations_prev: Previous step activations
            
        Returns:
            Shift score (0-1, higher = more change)
        """
        diff = activations_curr - activations_prev
        diff_norm = torch.norm(diff.flatten(), p=2)
        curr_norm = torch.norm(activations_curr.flatten(), p=2)
        
        # Avoid division by zero
        if curr_norm < 1e-8:
            return 0.0
        
        return (diff_norm / curr_norm).item()
    
    def calibrate(
        self,
        pipeline,
        prompts: List[str],
        num_steps: int = 20,
        max_prompts: int = 100
    ) -> Dict:
        """
        Run calibration to find optimal phase division.
        
        Args:
            pipeline: StableDiffusionXLPipeline
            prompts: List of calibration prompts
            num_steps: Number of denoising steps
            max_prompts: Maximum prompts to use
            
        Returns:
            Calibration results with D* and shift scores
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("PAS Calibration - Shift Score Analysis")
            print(f"{'='*60}\n")
            print(f"Prompts: {min(len(prompts), max_prompts)}")
            print(f"Steps: {num_steps}\n")
        
        # Sample prompts
        if len(prompts) > max_prompts:
            import random
            prompts = random.sample(prompts, max_prompts)
        
        # Initialize shift score matrix (prompts x steps)
        shift_matrix = np.zeros((len(prompts), num_steps - 1))
        
        # Hook upsampling blocks
        unet = pipeline.unet
        handles = self.hook_upsampling_blocks(unet)
        
        try:
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Calibrating")):
                # Generate with this prompt
                activations_per_step = []
                
                # Custom denoising loop to capture activations
                with torch.no_grad():
                    # Encode prompt
                    prompt_embeds = pipeline.encode_prompt(
                        prompt=prompt,
                        device=self.device,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True
                    )
                    
                    # Prepare latents
                    latents = torch.randn(
                        (1, 4, 128, 128),
                        device=self.device,
                        dtype=torch.float32
                    )
                    
                    # Set timesteps
                    pipeline.scheduler.set_timesteps(num_steps)
                    timesteps = pipeline.scheduler.timesteps
                    
                    # Denoising loop
                    for t_idx, t in enumerate(timesteps):
                        # Clear previous activations
                        self.hooked_activations = {}
                        
                        # Expand latents for CFG
                        latent_model_input = torch.cat([latents] * 2)
                        
                        # Predict noise
                        noise_pred = unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds[0],
                            return_dict=False
                        )[0]
                        
                        # CFG
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                        
                        # Scheduler step
                        latents = pipeline.scheduler.step(noise_pred, t, latents).prev_sample
                        
                        # Store activations (use first upsampling block as representative)
                        if 'up_block_0' in self.hooked_activations:
                            activations_per_step.append(
                                self.hooked_activations['up_block_0'].clone()
                            )
                
                # Compute shift scores for this prompt
                for t_idx in range(1, len(activations_per_step)):
                    shift = self.compute_shift_score(
                        activations_per_step[t_idx],
                        activations_per_step[t_idx - 1]
                    )
                    shift_matrix[prompt_idx, t_idx - 1] = shift
        
        finally:
            # Remove hooks
            for handle in handles:
                handle.remove()
        
        # Average shift scores across prompts
        avg_shift_scores = shift_matrix.mean(axis=0)
        
        # Find D* using K-means (K=2)
        D_star = self._find_phase_division(avg_shift_scores)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Calibration Complete")
            print(f"{'='*60}\n")
            print(f"Optimal phase division: D* = {D_star}")
            print(f"Sketching phase: steps 0-{D_star-1}")
            print(f"Refinement phase: steps {D_star}-{num_steps-1}\n")
        
        return {
            'D_star': int(D_star),
            'num_steps': num_steps,
            'shift_scores': avg_shift_scores.tolist(),
            'shift_matrix': shift_matrix.tolist(),
            'num_prompts': len(prompts)
        }
    
    def _find_phase_division(self, shift_scores: np.ndarray) -> int:
        """
        Find optimal phase division using K-means clustering.
        
        Args:
            shift_scores: Array of shift scores (length = num_steps - 1)
            
        Returns:
            D* - optimal division point
        """
        # Reshape for sklearn
        X = shift_scores.reshape(-1, 1)
        
        # K-means with K=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Find transition point (where label changes from high to low variance)
        # High shift scores = label for sketching phase
        high_shift_label = labels[0]  # Early steps should have high shift
        
        # Find last occurrence of high shift label
        D_star = 0
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] == high_shift_label:
                D_star = i + 1  # +1 because we want the boundary
                break
        
        # Ensure D* is reasonable (not too early or late)
        D_star = max(3, min(D_star, len(shift_scores) - 3))
        
        return D_star
    
    def visualize_shift_scores(
        self,
        shift_scores: List[float],
        D_star: int,
        output_path: str = "shift_score_analysis.png"
    ):
        """
        Create visualization of shift scores and phase division.
        
        Args:
            shift_scores: List of shift scores
            D_star: Phase division point
            output_path: Where to save plot
        """
        plt.figure(figsize=(12, 6))
        
        steps = np.arange(1, len(shift_scores) + 1)
        plt.plot(steps, shift_scores, 'b-', linewidth=2, label='Shift Score')
        
        # Mark D*
        plt.axvline(x=D_star, color='r', linestyle='--', linewidth=2, label=f'D* = {D_star}')
        
        # Shade regions
        plt.axvspan(1, D_star, alpha=0.2, color='orange', label='Sketching Phase')
        plt.axvspan(D_star, len(shift_scores), alpha=0.2, color='green', label='Refinement Phase')
        
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Shift Score (normalized L2)', fontsize=12)
        plt.title('Phase-aware Sampling: Shift Score Analysis', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=150)
        print(f"✅ Visualization saved to: {output_path}")


if __name__ == "__main__":
    print("PAS Calibration Module")
    print("Use calibrate_pas.py script to run calibration")
