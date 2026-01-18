"""
Schedule Optimizer - Algorithm 1 from AYS Paper

Implements iterative coordinate descent to find optimal sampling schedule
that minimizes KLUB (discretization error).

Reference: AYS (Align Your Steps) - https://arxiv.org/abs/2404.14507
"""

import numpy as np
from typing import List, Optional, Callable
from tqdm import tqdm
import copy

from .klub_estimation import KLUBEstimator


class ScheduleOptimizer:
    """
    Implements Algorithm 1: Coordinate Descent Optimization for Sampling Schedules.
    
    Algorithm Flow (from user's description):
    1. Initialize with a baseline schedule (EDM, time-uniform, etc.)
    2. Outer loop: Iterate until convergence (no changes found)
    3. Inner loop: For each timestep t_i (except endpoints):
       a. Generate candidate values in neighborhood of t_i
       b. Evaluate KLUB for each candidate
       c. Greedily select candidate with lowest KLUB
       d. Update t_i if improvement found
    4. Repeat until no changes (convergence)
    
    Key Update Rule:
    t_i^new = arg min_{τ ∈ Neighbourhood(t_i)} (KLUB(t_{i-1}, τ) + KLUB(τ, t_{i+1}))
    """
    
    def __init__(
        self,
        klub_estimator: Optional[KLUBEstimator] = None,
        num_candidates: int = 20,
        neighborhood_size: float = 0.1,
        c: float = 0.1,
        convergence_threshold: float = 1e-4,
        max_iterations: int = 100,
        verbose: bool = True
    ):
        """
        Initialize Schedule Optimizer.
        
        Args:
            klub_estimator: KLUB estimator instance (creates default if None)
            num_candidates: Number of candidate values to test per timestep
            neighborhood_size: Size of neighborhood (as fraction of current value)
            c: Regularization constant for KLUB formula
            convergence_threshold: Stop if no change larger than this
            max_iterations: Maximum number of outer loop iterations
            verbose: Print progress
        """
        self.klub_estimator = klub_estimator or KLUBEstimator()
        self.num_candidates = num_candidates
        self.neighborhood_size = neighborhood_size
        self.c = c
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Statistics
        self.iteration_history = []
    
    def optimize_schedule(
        self,
        initial_schedule: List[float],
        fix_endpoints: bool = True
    ) -> dict:
        """
        Optimize sampling schedule using Algorithm 1.
        
        Args:
            initial_schedule: Initial schedule [t_0, t_1, ..., t_n] (descending)
            fix_endpoints: Keep first and last timesteps fixed
            
        Returns:
            Dictionary with:
                - 'schedule': Optimized schedule
                - 'initial_klub': KLUB before optimization
                - 'final_klub': KLUB after optimization
                - 'iterations': Number of iterations
                - 'history': KLUB history per iteration
        """
        # Deep copy to avoid modifying input
        schedule = copy.deepcopy(initial_schedule)
        
        # Compute initial KLUB
        if self.verbose:
            print(f"\n{'='*60}")
            print("Schedule Optimization - Algorithm 1")
            print(f"{'='*60}")
            print(f"Initial schedule: {len(schedule)} steps")
            print(f"Range: [{schedule[0]:.3f}, {schedule[-1]:.3f}]")
        
        initial_klub = self.klub_estimator.estimate_full_schedule_klub(
            schedule=schedule,
            c=self.c,
            verbose=False
        )
        
        if self.verbose:
            print(f"Initial KLUB: {initial_klub:.6f}\n")
        
        # Outer loop: Iterate until convergence
        iteration = 0
        no_change = False
        klub_history = [initial_klub]
        
        progress_bar = None
        if self.verbose:
            progress_bar = tqdm(total=self.max_iterations, desc="Optimization")
        
        while not no_change and iteration < self.max_iterations:
            no_change = True  # Assume no changes until proven otherwise
            changes_made = 0
            
            # Inner loop: Coordinate descent over each timestep
            start_idx = 1 if fix_endpoints else 0
            end_idx = len(schedule) - 1 if fix_endpoints else len(schedule)
            
            for i in range(start_idx, end_idx):
                t_prev = schedule[i - 1] if i > 0 else None
                t_curr = schedule[i]
                t_next = schedule[i + 1] if i < len(schedule) - 1 else None
                
                # Generate candidates in neighborhood
                candidates = self._generate_candidates(
                    t_curr=t_curr,
                    t_prev=t_prev,
                    t_next=t_next
                )
                
                # Evaluate KLUB for each candidate
                klub_scores = self.klub_estimator.estimate_batch_klub(
                    t_prev=t_prev,
                    t_curr_candidates=candidates,
                    t_next=t_next,
                    c=self.c
                )
                
                # Find best candidate (greedy selection)
                min_idx = np.argmin(klub_scores)
                best_candidate = candidates[min_idx]
                
                # Update if different from current
                if abs(best_candidate - t_curr) > self.convergence_threshold:
                    schedule[i] = best_candidate
                    no_change = False
                    changes_made += 1
            
            # Compute KLUB after this iteration
            current_klub = self.klub_estimator.estimate_full_schedule_klub(
                schedule=schedule,
                c=self.c,
                verbose=False
            )
            klub_history.append(current_klub)
            
            iteration += 1
            if self.verbose and progress_bar:
                improvement = klub_history[-2] - current_klub if len(klub_history) > 1 else 0
                progress_bar.set_postfix({
                    'KLUB': f'{current_klub:.6f}',
                    'Changes': changes_made,
                    'Δ': f'{improvement:.6f}'
                })
                progress_bar.update(1)
        
        if self.verbose and progress_bar:
            progress_bar.close()
        
        final_klub = klub_history[-1]
        improvement = initial_klub - final_klub
        improvement_pct = (improvement / initial_klub) * 100 if initial_klub > 0 else 0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("Optimization Complete")
            print(f"{'='*60}")
            print(f"Iterations: {iteration}")
            print(f"Initial KLUB: {initial_klub:.6f}")
            print(f"Final KLUB: {final_klub:.6f}")
            print(f"Improvement: {improvement:.6f} ({improvement_pct:.2f}%)")
            print(f"Converged: {'Yes' if no_change else 'No (max iterations)'}")
        
        return {
            'schedule': schedule,
            'initial_klub': initial_klub,
            'final_klub': final_klub,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'iterations': iteration,
            'history': klub_history,
            'converged': no_change
        }
    
    def _generate_candidates(
        self,
        t_curr: float,
        t_prev: Optional[float] = None,
        t_next: Optional[float] = None
    ) -> List[float]:
        """
        Generate candidate timesteps in neighborhood of t_curr.
        
        Candidates are generated within valid bounds (between t_prev and t_next).
        
        Args:
            t_curr: Current timestep
            t_prev: Previous timestep (upper bound)
            t_next: Next timestep (lower bound)
            
        Returns:
            List of candidate timesteps
        """
        # Determine neighborhood range
        delta = t_curr * self.neighborhood_size
        
        # Determine valid bounds
        lower_bound = t_next if t_next is not None else max(t_curr - delta, 0.001)
        upper_bound = t_prev if t_prev is not None else t_curr + delta
        
        # Ensure t_curr is within bounds
        lower_bound = min(lower_bound, t_curr - delta)
        upper_bound = max(upper_bound, t_curr + delta)
        
        # Generate candidates (linear spacing)
        candidates = np.linspace(lower_bound, upper_bound, self.num_candidates)
        
        # Filter to ensure valid order (t_prev > candidates > t_next)
        valid_candidates = []
        for cand in candidates:
            is_valid = True
            if t_prev is not None and cand >= t_prev:
                is_valid = False
            if t_next is not None and cand <= t_next:
                is_valid = False
            if is_valid:
                valid_candidates.append(cand)
        
        # Always include current value
        if t_curr not in valid_candidates:
            valid_candidates.append(t_curr)
        
        return sorted(valid_candidates, reverse=True)  # Descending order
    
    def hierarchical_optimization(
        self,
        initial_schedule: List[float],
        target_steps: List[int] = [10, 20],
        fix_endpoints: bool = True
    ) -> dict:
        """
        Hierarchical optimization: optimize at multiple resolutions.
        
        Strategy from AYS paper:
        1. Optimize 10-step schedule
        2. Subdivide to 20 steps (insert midpoints)
        3. Optimize new points only
        4. (Optional) Continue to 40 steps
        
        Args:
            initial_schedule: Initial schedule (typically 10 steps)
            target_steps: List of target step counts [10, 20, 40, ...]
            fix_endpoints: Keep endpoints fixed
            
        Returns:
            Dictionary with final optimized schedule and history
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("Hierarchical Schedule Optimization")
            print(f"{'='*60}")
            print(f"Target steps: {target_steps}")
        
        current_schedule = initial_schedule
        all_results = []
        
        for target_step in target_steps:
            if self.verbose:
                print(f"\n--- Optimizing for {target_step} steps ---")
            
            # Subdivide if needed
            if len(current_schedule) < target_step:
                from .schedules import subdivide_schedule
                while len(current_schedule) < target_step:
                    current_schedule = subdivide_schedule(current_schedule)
                
                if self.verbose:
                    print(f"Subdivided to {len(current_schedule)} steps")
            
            # Optimize
            result = self.optimize_schedule(
                initial_schedule=current_schedule,
                fix_endpoints=fix_endpoints
            )
            
            current_schedule = result['schedule']
            all_results.append(result)
        
        # Return final result
        final_result = all_results[-1]
        final_result['hierarchical_history'] = all_results
        
        return final_result


if __name__ == "__main__":
    # Test Schedule Optimizer
    from .schedules import edm_schedule
    
    print("Testing Schedule Optimizer (Algorithm 1)...")
    
    # Create initial schedule
    initial = edm_schedule(num_steps=10, sigma_min=0.5, sigma_max=80.0)
    print(f"\nInitial EDM schedule (10 steps):")
    print([f"{t:.2f}" for t in initial])
    
    # Create optimizer
    optimizer = ScheduleOptimizer(
        num_candidates=15,
        neighborhood_size=0.15,
        max_iterations=20,
        verbose=True
    )
    
    # Optimize
    result = optimizer.optimize_schedule(initial_schedule=initial)
    
    print(f"\nOptimized schedule:")
    print([f"{t:.2f}" for t in result['schedule']])
    
    print(f"\nKLUB improvement: {result['improvement']:.6f}")
