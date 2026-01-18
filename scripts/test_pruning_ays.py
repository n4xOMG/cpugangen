#!/usr/bin/env python3
"""
Quick Test Script for Pruning + AYS Experiment

This script runs a quick test of the full pipeline:
1. Initialize small schedule
2. Optimize using Algorithm 1
3. Show results

Usage:
    python scripts/test_pruning_ays.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.schedules import edm_schedule
from algorithms.schedule_optimizer import ScheduleOptimizer
from algorithms.klub_estimation import KLUBEstimator


def main():
    print("\n" + "="*60)
    print("Quick Test: Pruning + AYS Pipeline")
    print("="*60 + "\n")
    
    # Step 1: Create initial schedule
    print("Step 1: Creating initial EDM schedule (10 steps)...")
    initial_schedule = edm_schedule(num_steps=10, sigma_min=0.5, sigma_max=80.0)
    print(f"Initial schedule: {[f'{t:.2f}' for t in initial_schedule]}\n")
    
    # Step 2: Compute initial KLUB
    print("Step 2: Computing initial KLUB...")
    klub_estimator = KLUBEstimator()
    initial_klub = klub_estimator.estimate_full_schedule_klub(
        schedule=initial_schedule,
        c=0.1,
        verbose=False
    )
    print(f"Initial KLUB: {initial_klub:.6f}\n")
    
    # Step 3: Optimize schedule
    print("Step 3: Optimizing schedule (Algorithm 1)...")
    optimizer = ScheduleOptimizer(
        klub_estimator=klub_estimator,
        num_candidates=15,
        neighborhood_size=0.15,
        max_iterations=20,  # Reduced for quick test
        verbose=True
    )
    
    result = optimizer.optimize_schedule(
        initial_schedule=initial_schedule,
        fix_endpoints=True
    )
    
    # Step 4: Show results
    print("\n" + "="*60)
    print("Results")
    print("="*60 + "\n")
    
    print("Initial schedule:")
    for i, t in enumerate(initial_schedule):
        print(f"  t[{i}] = {t:.4f}")
    
    print("\nOptimized schedule:")
    for i, t in enumerate(result['schedule']):
        delta = t - initial_schedule[i]
        print(f"  t[{i}] = {t:.4f} (Δ = {delta:+.4f})")
    
    print(f"\nKLUB Improvement:")
    print(f"  Initial: {result['initial_klub']:.6f}")
    print(f"  Final: {result['final_klub']:.6f}")
    print(f"  Improvement: {result['improvement']:.6f} ({result['improvement_pct']:.2f}%)")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Converged: {result['converged']}")
    
    print("\n✅ Test complete!\n")
    
    # Step 5: Test hierarchical optimization
    print("="*60)
    print("Bonus: Testing Hierarchical Optimization (10 -> 20)")
    print("="*60 + "\n")
    
    hierarchical_result = optimizer.hierarchical_optimization(
        initial_schedule=initial_schedule,
        target_steps=[10, 20],
        fix_endpoints=True
    )
    
    print(f"\nHierarchical optimization complete!")
    print(f"Final schedule: {len(hierarchical_result['schedule'])} steps")
    print(f"KLUB improvement: {hierarchical_result['improvement']:.6f} ({hierarchical_result['improvement_pct']:.2f}%)")


if __name__ == "__main__":
    main()
