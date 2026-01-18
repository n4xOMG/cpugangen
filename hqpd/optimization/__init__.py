"""
Optimization utilities for CPU inference.

This module provides:
- Deterministic operation caching
- Process and thread pinning
- Performance profiling tools
"""

from .deterministic_cache import (
    DeterministicCache,
    cached_operation,
    CacheContext,
    get_global_cache,
)
from .process_pinning import (
    pin_to_cores,
    configure_threading,
    get_cpu_info,
    optimize_for_inference,
)

__all__ = [
    "DeterministicCache",
    "cached_operation",
    "CacheContext",
    "get_global_cache",
    "pin_to_cores",
    "configure_threading",
    "get_cpu_info",
    "optimize_for_inference",
]


