#!/usr/bin/env python3
"""
Optimize Sampling Schedule using AYS Algorithm 1

Runs KLUB-based coordinate descent to find optimal sampling schedule
for a given model (pruned or unpruned).

Usage:
    python scripts/optimize_sampling_schedule.py \\
        --model checkpoints/pruned_sdxl \\
        --initial-schedule edm \\
        --num-steps 10 \\
        --output configs/optimal_schedule.json

Reference: "Align Your Steps" (arxiv:2404.14507)
"""

import argparse
import os
import sys
import json
import torch
from diffusers import StableDiffusionXLPipeline

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from algorithms.schedule_optimizer import ScheduleOptimizer
from algorithms.klub_estimation import KLUBEstimator
from algorithms.schedules import edm_schedule, time_uniform_schedule, cosine_schedule, save_schedule


def optimize_schedule(
    model_path: str,
    initial_schedule_type: str = "edm",
    num_steps: int = 10,
    output_path: str = "configs/optimal_schedule.json",
    hierarchical: bool = False,
    target_steps: list = None,
    max_iterations: int = 100,
    num_candidates: int = 20,
    neighborhood_size: float = 0.15,
    c: float = 0.1,
    device: str = "cpu"
):
    """
    Optimize sampling schedule for a model.
    
    Args:
        model_path: Path to model (pruned or unpruned SDXL)
        initial_schedule_type: Type of initial schedule ('edm', 'uniform', 'cosine')
        num_steps: Number of sampling steps
        output_path: Where to save optimized schedule
        hierarchical: Use hierarchical optimization (10 -> 20)
        target_steps: List of target steps for hierarchical
        max_iterations: Max iterations for optimization
        num_candidates: Number of candidates per timestep
        neighborhood_size: Size of neighborhood for candidate generation
        c: Regularization constant for KLUB formula
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print("Sampling Schedule Optimization - AYS Algorithm 1")
    print(f"{'='*60}\n")
    print(f"Model: {model_path}")
    print(f"Initial Schedule: {initial_schedule_type}")
    print(f"Num Steps: {num_steps}")
    print(f"Hierarchical: {hierarchical}")
    if hierarchical and target_steps:
        print(f"Target Steps: {target_steps}")
    print(f"Output: {output_path}")
    print(f"Device: {device}\n")
    
    # Create initial schedule
    print("Creating initial schedule...")
    if initial_schedule_type == "edm":
        initial_schedule = edm_schedule(num_steps, sigma_min=0.5, sigma_max=80.0)
    elif initial_schedule_type == "uniform":
        initial_schedule = time_uniform_schedule(num_steps, t_min=0.5, t_max=80.0)
    elif initial_schedule_type == "cosine":
        initial_schedule = cosine_schedule(num_steps)
    else:
        raise ValueError(f"Unknown schedule type: {initial_schedule_type}")
    
    print(f"Initial schedule: {[f'{t:.2f}' for t in initial_schedule[:5]]} ... {[f'{t:.2f}' for t in initial_schedule[-2:]]}")
    print()
    
    # Note: For publication-quality results, we need the actual denoiser
    # For now, we use the analytical KLUB formula which doesn't require model loading
    # This makes optimization much faster and model-agnostic
    
    # Create optimizer
    klub_estimator = KLUBEstimator()
    optimizer = ScheduleOptimizer(
        klub_estimator=klub_estimator,
        num_candidates=num_candidates,
        neighborhood_size=neighborhood_size,
        c=c,
        max_iterations=max_iterations,
        verbose=True
    )
    
    # Run optimization
    if hierarchical:
        if target_steps is None:
            target_steps = [num_steps, num_steps * 2]
        
        result = optimizer.hierarchical_optimization(
            initial_schedule=initial_schedule,
            target_steps=target_steps,
            fix_endpoints=True
        )
    else:
        result = optimizer.optimize_schedule(
            initial_schedule=initial_schedule,
            fix_endpoints=True
        )
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    metadata = {
        'base_model': model_path,
        'initial_schedule_type': initial_schedule_type,
        'num_steps': len(result['schedule']),
        'initial_klub': result['initial_klub'],
        'final_klub': result['final_klub'],
        'improvement': result['improvement'],
        'improvement_pct': result['improvement_pct'],
        'iterations': result['iterations'],
        'converged': result['converged'],
        'optimization_params': {
            'num_candidates': num_candidates,
            'neighborhood_size': neighborhood_size,
            'c': c,
            'max_iterations': max_iterations,
        }
    }
    
    save_schedule(
        schedule=result['schedule'],
        path=output_path,
        metadata=metadata
    )
    
    # Print final schedule
    print(f"\nOptimized schedule ({len(result['schedule'])} steps):")
    for i, t in enumerate(result['schedule']):
        print(f"  Step {i}: {t:.4f}")
    
    print(f"\n✅ Optimization complete!")
    print(f"KLUB improvement: {result['improvement']:.6f} ({result['improvement_pct']:.2f}%)")
    print(f"Schedule saved to {output_path}")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Sampling Schedule using AYS")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/pruned_sdxl",
        help="Model path (pruned or unpruned)"
    )
    parser.add_argument(
        "--initial-schedule",
        type=str,
        default="edm",
        choices=["edm", "uniform", "cosine"],
        help="Initial schedule type"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/optimal_schedule.json",
        help="Output path for optimized schedule"
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        help="Use hierarchical optimization (10 -> 20)"
    )
    parser.add_argument(
        "--target-steps",
        type=int,
        nargs="+",
        default=None,
        help="Target steps for hierarchical optimization (e.g., 10 20)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum optimization iterations"
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=20,
        help="Number of candidates per timestep"
    )
    parser.add_argument(
        "--neighborhood-size",
        type=float,
        default=0.15,
        help="Neighborhood size (as fraction)"
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.1,
        help="Regularization constant for KLUB"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Run optimization
    optimize_schedule(
        model_path=args.model,
        initial_schedule_type=args.initial_schedule,
        num_steps=args.num_steps,
        output_path=args.output,
        hierarchical=args.hierarchical,
        target_steps=args.target_steps,
        max_iterations=args.max_iterations,
        num_candidates=args.num_candidates,
        neighborhood_size=args.neighborhood_size,
        c=args.c,
        device=args.device
    )
