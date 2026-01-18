"""
KLUB (Kullback-Leibler Upper Bound) Estimation - Algorithm 2

Estimates the discretization error of a sampling schedule using Monte Carlo estimation.
This measures how well the discrete sampling steps approximate the continuous ODE flow.

Reference: AYS (Align Your Steps) - https://arxiv.org/abs/2404.14507
"""

import torch
import numpy as np
from typing import List, Callable, Optional
from tqdm import tqdm


class KLUBEstimator:
    """
    KLUB Estimator using Monte Carlo sampling.
    
    The KLUB measures discretization error when approximating the continuous
    probability flow ODE with discrete timesteps. Lower KLUB = better schedule.
    
    Algorithm 2 from AYS paper:
    1. Sample random noise x_t at current timestep
    2. Predict noise using denoiser: ε_pred = D_θ(x_t, σ_t)
    3. Estimate local error between predicted and actual trajectory
    4. Average over multiple Monte Carlo samples
    """
    
    def __init__(
        self,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize KLUB estimator.
        
        Args:
            device: Device to run computations on
            dtype: Data type for computations
        """
        self.device = device
        self.dtype = dtype
    
    def estimate_local_klub(
        self,
        t_prev: float,
        t_curr: float,
        t_next: float,
        c: float = 0.1,
        num_quadrature_points: int = 100
    ) -> float:
        """
        Estimate KLUB for a single triplet of timesteps using the integral formula.
        
        From AYS paper (Lemma 3.3):
        KLUB(t_{i-1}, t_i) ∝ ∫_{t_{i-1}}^{t_i} (1/t³) · (1/(t² + c²) - 1/(t_i² + c²)) dt
        
        The total cost at t_i is: KLUB(t_{i-1}, t_i) + KLUB(t_i, t_{i+1})
        
        Args:
            t_prev: Previous timestep t_{i-1}
            t_curr: Current timestep t_i (the one being evaluated)
            t_next: Next timestep t_{i+1}
            c: Regularization constant (default: 0.1, prevents division by zero)
            num_quadrature_points: Number of points for numerical integration
            
        Returns:
            KLUB score (lower is better)
        """
        # Compute KLUB(t_{i-1}, t_i) - from previous to current
        if t_prev is not None and t_prev > t_curr:
            klub_prev_to_curr = self._compute_klub_integral(
                t_start=t_curr,
                t_end=t_prev,
                t_i=t_curr,
                c=c,
                num_points=num_quadrature_points
            )
        else:
            klub_prev_to_curr = 0.0
        
        # Compute KLUB(t_i, t_{i+1}) - from current to next
        if t_next is not None and t_curr > t_next:
            klub_curr_to_next = self._compute_klub_integral(
                t_start=t_next,
                t_end=t_curr,
                t_i=t_next,
                c=c,
                num_points=num_quadrature_points
            )
        else:
            klub_curr_to_next = 0.0
        
        # Total KLUB at this timestep
        total_klub = klub_prev_to_curr + klub_curr_to_next
        return total_klub
    
    def _compute_klub_integral(
        self,
        t_start: float,
        t_end: float,
        t_i: float,
        c: float,
        num_points: int = 100
    ) -> float:
        """
        Compute the KLUB integral using trapezoidal rule.
        
        KLUB(t_start, t_end) ∝ ∫_{t_start}^{t_end} (1/t³) · (1/(t² + c²) - 1/(t_i² + c²)) dt
        
        Args:
            t_start: Start of integration interval
            t_end: End of integration interval
            t_i: Reference timestep
            c: Regularization constant
            num_points: Number of quadrature points
            
        Returns:
            Integral value
        """
        if abs(t_end - t_start) < 1e-8:
            return 0.0
        
        # Create integration points (log-space for better accuracy)
        # Use log-space because the integrand has 1/t³ term
        t_vals = np.logspace(
            np.log10(max(t_start, 1e-6)),
            np.log10(max(t_end, 1e-5)),
            num_points
        )
        
        # Compute integrand at each point
        # f(t) = (1/t³) · (1/(t² + c²) - 1/(t_i² + c²))
        t_i_term = 1.0 / (t_i**2 + c**2)
        
        integrand_vals = []
        for t in t_vals:
            if t < 1e-8:  # Avoid division by zero
                continue
            
            factor_1 = 1.0 / (t**3)
            factor_2 = 1.0 / (t**2 + c**2) - t_i_term
            integrand = factor_1 * factor_2
            integrand_vals.append(integrand)
        
        if len(integrand_vals) == 0:
            return 0.0
        
        # Trapezoidal integration
        integral = np.trapz(integrand_vals, t_vals[:len(integrand_vals)])
        
        return abs(integral)  # Return absolute value
    
    def estimate_full_schedule_klub(
        self,
        schedule: List[float],
        c: float = 0.1,
        num_quadrature_points: int = 100,
        verbose: bool = True
    ) -> float:
        """
        Estimate total KLUB for entire sampling schedule.
        
        Sums up local KLUB estimates for all timesteps.
        
        Args:
            schedule: List of timesteps [t_0, t_1, ..., t_n] (descending order)
            c: Regularization constant
            num_quadrature_points: Number of points for numerical integration
            verbose: Print progress
            
        Returns:
            Total KLUB score (sum of local KLUBs)
        """
        if len(schedule) < 2:
            return 0.0
        
        total_klub = 0.0
        
        iterator = range(1, len(schedule) - 1)
        if verbose:
            iterator = tqdm(iterator, desc="Computing KLUB")
        
        for i in iterator:
            t_prev = schedule[i - 1] if i > 0 else None
            t_curr = schedule[i]
            t_next = schedule[i + 1] if i < len(schedule) - 1 else None
            
            local_klub = self.estimate_local_klub(
                t_prev=t_prev,
                t_curr=t_curr,
                t_next=t_next,
                c=c,
                num_quadrature_points=num_quadrature_points
            )
            
            total_klub += local_klub
        
        if verbose:
            avg_klub = total_klub / max(len(schedule) - 2, 1)
            print(f"Total KLUB: {total_klub:.6f} (avg per step: {avg_klub:.6f})")
        
        return total_klub
    
    def estimate_batch_klub(
        self,
        t_prev: float,
        t_curr_candidates: List[float],
        t_next: float,
        c: float = 0.1,
        num_quadrature_points: int = 100
    ) -> List[float]:
        """
        Estimate KLUB for multiple candidate t_i values (for greedy search).
        
        This is used in Algorithm 1 to efficiently evaluate multiple candidates.
        
        Args:
            t_prev: Previous timestep t_{i-1}
            t_curr_candidates: List of candidate t_i values to evaluate
            t_next: Next timestep t_{i+1}
            c: Regularization constant
            num_quadrature_points: Number of points for integration
            
        Returns:
            List of KLUB scores (one per candidate)
        """
        klub_scores = []
        
        for t_curr in t_curr_candidates:
            klub = self.estimate_local_klub(
                t_prev=t_prev,
                t_curr=t_curr,
                t_next=t_next,
                c=c,
                num_quadrature_points=num_quadrature_points
            )
            klub_scores.append(klub)
        
        return klub_scores

if __name__ == "__main__":
    # Test KLUB Estimator with the integral formula
    print("Testing KLUB Estimator with Integral Formula...")
    
    estimator = KLUBEstimator()
    
    # Test local KLUB
    print("\n1. Testing Local KLUB:")
    klub = estimator.estimate_local_klub(
        t_prev=80.0,
        t_curr=50.0,
        t_next=20.0,
        c=0.1,
        num_quadrature_points=100
    )
    print(f"KLUB(50.0 | 80.0, 20.0) = {klub:.6f}")
    
    # Test full schedule KLUB
    print("\n2. Testing Full Schedule KLUB:")
    schedule = [80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 1.0, 0.5]
    total_klub = estimator.estimate_full_schedule_klub(
        schedule=schedule,
        c=0.1,
        verbose=True
    )
    
    # Test batch KLUB
    print("\n3. Testing Batch KLUB (for candidate evaluation):")
    candidates = [48.0, 49.0, 50.0, 51.0, 52.0]
    batch_klubs = estimator.estimate_batch_klub(
        t_prev=80.0,
        t_curr_candidates=candidates,
        t_next=20.0,
        c=0.1
    )
    for cand, k in zip(candidates, batch_klubs):
        print(f"  Candidate {cand:.1f}: KLUB = {k:.6f}")
    
    best_idx = np.argmin(batch_klubs)
    print(f"  Best candidate: {candidates[best_idx]:.1f} (KLUB = {batch_klubs[best_idx]:.6f})")
