"""
Baseline Sampling Schedules for Diffusion Models

Implements standard schedules from literature:
- EDM (Karras et al., 2022)
- Time-uniform
- Cosine schedule

References:
- EDM: https://arxiv.org/abs/2206.00364
- AYS: https://arxiv.org/abs/2404.14507
"""

import numpy as np
import torch
from typing import List, Union
import json


def edm_schedule(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0
) -> List[float]:
    """
    EDM (Elucidating Diffusion Models) sampling schedule.
    
    From Karras et al. 2022: σ_i = (σ_max^(1/ρ) + (i/(N-1)) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ
    
    Args:
        num_steps: Number of sampling steps
        sigma_min: Minimum noise level (default: 0.002)
        sigma_max: Maximum noise level (default: 80.0)
        rho: Schedule curvature parameter (default: 7.0)
        
    Returns:
        List of sigma values (noise levels) in descending order
    """
    # Generate rho-spaced values
    ramp = np.linspace(0, 1, num_steps)
    
    # Apply EDM formula
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    
    # Convert to descending list (high noise -> low noise)
    return sigmas.tolist()


def time_uniform_schedule(
    num_steps: int,
    t_min: float = 0.0,
    t_max: float = 1.0
) -> List[float]:
    """
    Time-uniform sampling schedule.
    
    Evenly spaced timesteps in the range [t_max, t_min].
    
    Args:
        num_steps: Number of sampling steps
        t_min: Minimum timestep (default: 0.0)
        t_max: Maximum timestep (default: 1.0)
        
    Returns:
        List of timesteps in descending order
    """
    timesteps = np.linspace(t_max, t_min, num_steps)
    return timesteps.tolist()


def cosine_schedule(
    num_steps: int,
    s: float = 0.008
) -> List[float]:
    """
    Cosine noise schedule.
    
    From Improved DDPM (Nichol & Dhariwal, 2021).
    
    Args:
        num_steps: Number of sampling steps
        s: Offset to prevent β going to 0 near t=0 (default: 0.008)
        
    Returns:
        List of alpha_bar values (signal levels) in descending order
    """
    steps = num_steps + 1
    x = np.linspace(0, num_steps, steps)
    alphas_cumprod = np.cos(((x / num_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Convert to sigma values (noise levels)
    # sigma^2 = (1 - alpha_bar) / alpha_bar
    sigmas = np.sqrt((1 - alphas_cumprod) / alphas_cumprod)
    
    # Return in descending order (high noise -> low noise)
    return sigmas[::-1].tolist()[:num_steps]


def linear_schedule(
    num_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0
) -> List[float]:
    """
    Linear noise schedule (simple baseline).
    
    Args:
        num_steps: Number of sampling steps
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        
    Returns:
        List of sigma values in descending order
    """
    sigmas = np.linspace(sigma_max, sigma_min, num_steps)
    return sigmas.tolist()


def load_schedule(path: str) -> List[float]:
    """
    Load custom schedule from JSON file.
    
    Args:
        path: Path to JSON file containing schedule
        
    Returns:
        List of timesteps/sigmas
        
    Example JSON format:
        {
            "schedule": [80.0, 50.3, 30.2, 15.1, 5.0, 1.2, 0.002],
            "metadata": {
                "num_steps": 7,
                "optimization_method": "AYS",
                "base_model": "martineux/janku6"
            }
        }
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "schedule" in data:
        return data["schedule"]
    else:
        raise ValueError(f"Invalid schedule format in {path}")


def save_schedule(
    schedule: List[float],
    path: str,
    metadata: dict = None
):
    """
    Save schedule to JSON file.
    
    Args:
        schedule: List of timesteps/sigmas
        path: Output path
        metadata: Optional metadata dictionary
    """
    data = {
        "schedule": schedule,
        "num_steps": len(schedule),
    }
    
    if metadata:
        data["metadata"] = metadata
        
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Schedule saved to {path}")


def subdivide_schedule(schedule: List[float]) -> List[float]:
    """
    Hierarchical subdivision: Insert midpoints between existing timesteps.
    
    Used in AYS hierarchical strategy (10 -> 20 -> 40 steps).
    
    Args:
        schedule: Current schedule
        
    Returns:
        New schedule with 2*len(schedule)-1 steps
        
    Example:
        [10.0, 5.0, 1.0] -> [10.0, 7.5, 5.0, 3.0, 1.0]
    """
    if len(schedule) < 2:
        return schedule
    
    new_schedule = []
    for i in range(len(schedule) - 1):
        new_schedule.append(schedule[i])
        # Insert geometric mean (better for log-space noise levels)
        midpoint = np.sqrt(schedule[i] * schedule[i + 1])
        new_schedule.append(float(midpoint))
    
    new_schedule.append(schedule[-1])
    return new_schedule


def validate_schedule(schedule: List[float]) -> bool:
    """
    Validate that schedule is monotonically decreasing.
    
    Args:
        schedule: List of timesteps/sigmas
        
    Returns:
        True if valid, False otherwise
    """
    if len(schedule) < 2:
        return True
    
    for i in range(len(schedule) - 1):
        if schedule[i] <= schedule[i + 1]:
            print(f"⚠️  Schedule not monotonically decreasing at index {i}: "
                  f"{schedule[i]} <= {schedule[i+1]}")
            return False
    
    return True


if __name__ == "__main__":
    # Test schedules
    print("EDM Schedule (10 steps):")
    edm = edm_schedule(10)
    print(edm)
    print(f"Valid: {validate_schedule(edm)}\n")
    
    print("Time-uniform Schedule (10 steps):")
    uniform = time_uniform_schedule(10)
    print(uniform)
    print(f"Valid: {validate_schedule(uniform)}\n")
    
    print("Cosine Schedule (10 steps):")
    cos = cosine_schedule(10)
    print(cos)
    print(f"Valid: {validate_schedule(cos)}\n")
    
    print("Subdivision (10 -> 19 steps):")
    subdivided = subdivide_schedule(edm)
    print(f"Original: {len(edm)} steps")
    print(f"Subdivided: {len(subdivided)} steps")
    print(f"Valid: {validate_schedule(subdivided)}")
