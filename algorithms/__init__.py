"""
AYS (Align Your Steps) + Magnitude Pruning Algorithms

This package implements:
1. KLUB Estimation (Algorithm 2) - Monte Carlo estimation of discretization error
2. Schedule Optimization (Algorithm 1) - Coordinate descent for optimal sampling
3. Magnitude Pruning - Simple magnitude-based weight pruning
4. Baseline Schedules - EDM, Time-uniform, etc.
"""

from .schedules import edm_schedule, time_uniform_schedule, cosine_schedule
from .klub_estimation import KLUBEstimator
from .schedule_optimizer import ScheduleOptimizer
from .magnitude_pruning import MagnitudePruner

__all__ = [
    "edm_schedule",
    "time_uniform_schedule", 
    "cosine_schedule",
    "KLUBEstimator",
    "ScheduleOptimizer",
    "MagnitudePruner",
]
