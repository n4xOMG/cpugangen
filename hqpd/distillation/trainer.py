"""
Distillation trainer for UNet knowledge transfer.

Handles:
- Teacher/student forward passes
- Multi-component loss computation
- Gradient accumulation
- Mixed precision training
- Memory optimization
- Checkpointing and logging
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Learning rate
    learning_rate: float = 5e-6
    min_lr: float = 5e-7
    warmup_steps: int = 2000
    
    # Batch size
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    
    # Training duration
    max_steps: int = 50000
    
    # Stability
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.9999
    ema_update_every: int = 10
    
    # Mixed precision
    use_fp16: bool = True
    
    # Checkpointing
    save_every_steps: int = 5000
    eval_every_steps: int = 2000
    keep_last_n_checkpoints: int = 3
    
    # Logging
    log_every_steps: int = 100
    
    # Memory optimization
    enable_gradient_checkpointing: bool = True
    enable_xformers: bool = True
    

class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    @torch.no_grad()
    def update(self):
        """Update shadow weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
    
    def apply_shadow(self):
        """Apply shadow weights to model."""
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return self.shadow.copy()
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict.copy()


class CosineAnnealingWithWarmup:
    """Learning rate scheduler with warmup and cosine annealing."""
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self) -> float:
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (
                self.max_steps - self.warmup_steps
            )
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
            return self.base_lr * (self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay)


class DistillationTrainer:
    """
    Trainer for UNet distillation.
    
    Handles the complete training loop with:
    - Teacher-student distillation
    - Memory optimization
    - Mixed precision
    - Checkpointing
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        loss_fn: nn.Module,
        config: TrainingConfig,
        output_dir: str = "checkpoints/distilled_unet",
        device: str = "cuda",
    ):
        self.teacher = teacher
        self.student = student
        self.loss_fn = loss_fn
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup teacher
        self.teacher = self.teacher.to(device)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Setup student
        self.student = self.student.to(device)
        
        # Memory optimizations
        if config.enable_gradient_checkpointing:
            if hasattr(self.student, 'unet'):
                self.student.unet.enable_gradient_checkpointing()
            else:
                self.student.enable_gradient_checkpointing()
        
        if config.enable_xformers:
            try:
                if hasattr(self.student, 'unet'):
                    self.student.unet.enable_xformers_memory_efficient_attention()
                else:
                    self.student.enable_xformers_memory_efficient_attention()
                print("✓ xFormers enabled")
            except Exception as e:
                print(f"Warning: Could not enable xFormers: {e}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.student.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        
        # LR scheduler
        self.scheduler = CosineAnnealingWithWarmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=config.min_lr / config.learning_rate,
        )
        
        # EMA
        if config.ema_decay > 0:
            self.ema = EMA(self.student, decay=config.ema_decay)
        else:
            self.ema = None
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_fp16 else None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.checkpoint_paths = []
        
        # Logging
        self.log_history = []
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            batch: Dictionary with:
                - latents: (B, 4, H, W)
                - timesteps: (B,)
                - encoder_hidden_states: (B, 77, 2048)
                - added_cond_kwargs: dict
                
        Returns:
            loss, metrics
        """
        # Move to device
        latents = batch['latents'].to(self.device)
        timesteps = batch['timesteps'].to(self.device)
        encoder_hidden_states = batch['encoder_hidden_states'].to(self.device)
        
        added_cond_kwargs = {}
        if 'text_embeds' in batch:
            added_cond_kwargs['text_embeds'] = batch['text_embeds'].to(self.device)
        if 'time_ids' in batch:
            added_cond_kwargs['time_ids'] = batch['time_ids'].to(self.device)
        
        # Teacher forward (no grad, fp16)
        with torch.no_grad():
            with autocast(enabled=self.config.use_fp16):
                teacher_output = self.teacher(
                    sample=latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                )
        
        # Student forward (with grad)
        with autocast(enabled=self.config.use_fp16):
            if hasattr(self.student, 'forward') and 'return_features' in str(self.student.forward.__code__.co_varnames):
                student_output, student_features = self.student(
                    sample=latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_features=True,
                )
            else:
                student_output = self.student(
                    sample=latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                )
                student_features = {}
            
            # Get teacher features if available
            teacher_features = {}
            if hasattr(self.teacher, 'feature_extractor'):
                teacher_features = self.teacher.feature_extractor.get_features()
            
            # Compute loss
            loss, metrics = self.loss_fn(
                student_output=student_output.sample,
                teacher_output=teacher_output.sample,
                student_features=student_features,
                teacher_features=teacher_features,
            )
        
        return loss, metrics
    
    def train(
        self,
        dataloader: DataLoader,
        resume_from: Optional[str] = None,
    ):
        """
        Main training loop.
        
        Args:
            dataloader: Training data loader
            resume_from: Path to checkpoint to resume from
        """
        # Resume if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        print("=" * 70)
        print("Starting UNet Distillation Training")
        print("=" * 70)
        print(f"Max steps: {self.config.max_steps}")
        print(f"Batch size: {self.config.batch_size} × {self.config.gradient_accumulation_steps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Device: {self.device}")
        print("=" * 70)
        
        self.student.train()
        accumulated_loss = 0.0
        accumulated_metrics = {}
        
        progress = tqdm(total=self.config.max_steps, initial=self.global_step)
        data_iter = iter(dataloader)
        
        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            
            # Train step
            loss, metrics = self.train_step(batch)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Accumulate
            accumulated_loss += loss.item()
            for k, v in metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0) + v / self.config.gradient_accumulation_steps
            
            # Optimizer step
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(),
                    self.config.gradient_clip_norm,
                )
                
                # Step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                # EMA update
                if self.ema and self.global_step % self.config.ema_update_every == 0:
                    self.ema.update()
            
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config.log_every_steps == 0:
                log_entry = {
                    "step": self.global_step,
                    "loss": accumulated_loss,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    **accumulated_metrics,
                }
                self.log_history.append(log_entry)
                
                progress.set_postfix({
                    "loss": f"{accumulated_loss:.4f}",
                    "lr": f"{log_entry['lr']:.2e}",
                })
                
                accumulated_loss = 0.0
                accumulated_metrics = {}
            
            # Checkpointing
            if self.global_step % self.config.save_every_steps == 0:
                self.save_checkpoint()
            
            progress.update(1)
        
        progress.close()
        
        # Final checkpoint
        self.save_checkpoint(is_final=True)
        
        print("\n" + "=" * 70)
        print("✓ Training Complete!")
        print("=" * 70)
    
    def save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        step_str = f"step_{self.global_step}" if not is_final else "final"
        checkpoint_path = self.output_dir / f"checkpoint_{step_str}.pt"
        
        # Prepare state
        state = {
            "global_step": self.global_step,
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_step": self.scheduler.current_step,
            "best_loss": self.best_loss,
            "config": asdict(self.config),
        }
        
        if self.ema:
            state["ema_state_dict"] = self.ema.state_dict()
        
        if self.scaler:
            state["scaler_state_dict"] = self.scaler.state_dict()
        
        # Save
        torch.save(state, checkpoint_path)
        print(f"\n✓ Saved checkpoint: {checkpoint_path}")
        
        # Save log history
        log_path = self.output_dir / "training_log.json"
        with open(log_path, 'w') as f:
            json.dump(self.log_history, f, indent=2)
        
        # Manage checkpoint count
        self.checkpoint_paths.append(checkpoint_path)
        if len(self.checkpoint_paths) > self.config.keep_last_n_checkpoints:
            old_path = self.checkpoint_paths.pop(0)
            if old_path.exists() and "final" not in str(old_path):
                old_path.unlink()
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        print(f"Loading checkpoint from {path}...")
        
        state = torch.load(path, map_location=self.device)
        
        self.student.load_state_dict(state["student_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.current_step = state["scheduler_step"]
        self.global_step = state["global_step"]
        self.best_loss = state.get("best_loss", float('inf'))
        
        if self.ema and "ema_state_dict" in state:
            self.ema.load_state_dict(state["ema_state_dict"])
        
        if self.scaler and "scaler_state_dict" in state:
            self.scaler.load_state_dict(state["scaler_state_dict"])
        
        print(f"✓ Resumed from step {self.global_step}")


if __name__ == "__main__":
    print("Testing trainer components...")
    
    # Test EMA
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
    
    model = DummyModel()
    ema = EMA(model, decay=0.999)
    
    # Simulate update
    model.linear.weight.data += 0.1
    ema.update()
    
    print("✓ EMA test passed")
    
    # Test scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingWithWarmup(optimizer, warmup_steps=100, max_steps=1000)
    
    lrs = []
    for _ in range(1000):
        scheduler.step()
        lrs.append(scheduler.get_lr())
    
    print(f"✓ Scheduler test passed (LR range: {min(lrs):.6f} - {max(lrs):.6f})")
    
    print("\n✓ All trainer tests passed!")
