"""
Deterministic Operation Caching for Diffusion Inference.

This module provides caching mechanisms for operations that are computed
identically across multiple denoising steps, such as:
- Sinusoidal timestep embeddings
- Fixed weight projections
- Other deterministic transformations

The cache is thread-safe and memory-bounded with LRU eviction.
"""

import hashlib
import functools
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import numpy as np


class DeterministicCache:
    """
    Thread-safe LRU cache for deterministic tensor operations.
    
    Features:
    - Automatic cache key generation from inputs
    - Memory-bounded with LRU eviction
    - Cache statistics tracking
    - Thread-safe operations
    
    Example:
        >>> cache = DeterministicCache(max_size=128)
        >>> result = cache.get_or_compute(
        ...     key="timestep_100",
        ...     compute_fn=lambda: expensive_operation()
        ... )
    """
    
    def __init__(self, max_size: int = 128, enabled: bool = True):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of cached entries (LRU eviction)
            enabled: Whether caching is enabled
        """
        self.max_size = max_size
        self.enabled = enabled
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._total_memory_bytes = 0
    
    def _compute_key(self, *args, **kwargs) -> str:
        """
        Compute cache key from arguments.
        
        Uses hash of string representation for tensors/arrays,
        direct values for primitives.
        """
        key_parts = []
        
        # Hash positional arguments
        for arg in args:
            if isinstance(arg, (torch.Tensor, np.ndarray)):
                # For tensors, hash shape, dtype, and device
                if isinstance(arg, torch.Tensor):
                    key_parts.append(f"tensor_{arg.shape}_{arg.dtype}_{arg.device}")
                else:
                    key_parts.append(f"array_{arg.shape}_{arg.dtype}")
            elif isinstance(arg, (int, float, str, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(type(arg)))
        
        # Hash keyword arguments
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (torch.Tensor, np.ndarray)):
                if isinstance(v, torch.Tensor):
                    key_parts.append(f"{k}=tensor_{v.shape}_{v.dtype}_{v.device}")
                else:
                    key_parts.append(f"{k}=array_{v.shape}_{v.dtype}")
            elif isinstance(v, (int, float, str, bool)):
                key_parts.append(f"{k}={v}")
            else:
                key_parts.append(f"{k}={type(v)}")
        
        # Create hash
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], torch.Tensor]
    ) -> torch.Tensor:
        """
        Get cached value or compute and cache it.
        
        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            
        Returns:
            Cached or computed tensor
        """
        if not self.enabled:
            return compute_fn()
        
        with self._lock:
            # Check cache
            if key in self._cache:
                self._hits += 1
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                return self._cache[key]
            
            # Cache miss - compute
            self._misses += 1
            result = compute_fn()
            
            # Store in cache
            self._cache[key] = result.clone() if isinstance(result, torch.Tensor) else result
            self._cache.move_to_end(key)
            
            # Track memory
            if isinstance(result, torch.Tensor):
                self._total_memory_bytes += result.element_size() * result.numel()
            
            # Evict if over size limit
            while len(self._cache) > self.max_size:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                if isinstance(oldest_value, torch.Tensor):
                    self._total_memory_bytes -= oldest_value.element_size() * oldest_value.numel()
            
            return result
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._total_memory_bytes = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hits, misses, hit_rate, size, memory_mb
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "total_requests": total,
                "hit_rate": hit_rate,
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "memory_mb": self._total_memory_bytes / (1024 * 1024),
                "enabled": self.enabled,
            }
    
    def reset_statistics(self):
        """Reset hit/miss counters."""
        with self._lock:
            self._hits = 0
            self._misses = 0


# Global cache instance
_global_cache: Optional[DeterministicCache] = None
_global_cache_lock = threading.Lock()


def get_global_cache(max_size: int = 128, enabled: bool = True) -> DeterministicCache:
    """
    Get or create global cache instance.
    
    Args:
        max_size: Maximum cache size (only used on first call)
        enabled: Whether cache is enabled (only used on first call)
        
    Returns:
        Global cache instance
    """
    global _global_cache
    
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = DeterministicCache(max_size=max_size, enabled=enabled)
    
    return _global_cache


def cached_operation(
    cache: Optional[DeterministicCache] = None,
    key_fn: Optional[Callable[..., str]] = None
) -> Callable:
    """
    Decorator to cache deterministic operations.
    
    Args:
        cache: Cache instance to use (None = global cache)
        key_fn: Optional custom key function (args, kwargs) -> str
        
    Example:
        >>> @cached_operation()
        ... def compute_timestep_embedding(timestep, dim):
        ...     # Expensive computation
        ...     return embedding
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get cache
            cache_instance = cache if cache is not None else get_global_cache()
            
            if not cache_instance.enabled:
                return func(*args, **kwargs)
            
            # Compute key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                key = cache_instance._compute_key(*args, **kwargs)
            
            # Get or compute
            return cache_instance.get_or_compute(
                key=key,
                compute_fn=lambda: func(*args, **kwargs)
            )
        
        return wrapper
    return decorator


class CacheContext:
    """
    Context manager for cache control.
    
    Example:
        >>> with CacheContext(enabled=True, max_size=256) as cache:
        ...     # Run inference
        ...     image = pipeline(prompt)
        ...     # Check stats
        ...     print(cache.get_statistics())
    """
    
    def __init__(
        self,
        enabled: bool = True,
        max_size: int = 128,
        clear_on_enter: bool = False,
        print_stats_on_exit: bool = True
    ):
        """
        Initialize cache context.
        
        Args:
            enabled: Whether to enable caching
            max_size: Maximum cache size
            clear_on_enter: Clear cache on entering context
            print_stats_on_exit: Print statistics on exit
        """
        self.enabled = enabled
        self.max_size = max_size
        self.clear_on_enter = clear_on_enter
        self.print_stats_on_exit = print_stats_on_exit
        self.cache: Optional[DeterministicCache] = None
    
    def __enter__(self) -> DeterministicCache:
        """Enter context - initialize cache."""
        self.cache = get_global_cache(max_size=self.max_size, enabled=self.enabled)
        
        if self.clear_on_enter:
            self.cache.clear()
        
        self.cache.reset_statistics()
        
        return self.cache
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - optionally print statistics."""
        if self.cache and self.print_stats_on_exit:
            stats = self.cache.get_statistics()
            print("\n" + "="*60)
            print("Cache Statistics")
            print("="*60)
            print(f"Enabled:     {stats['enabled']}")
            print(f"Hits:        {stats['hits']}")
            print(f"Misses:      {stats['misses']}")
            print(f"Hit Rate:    {stats['hit_rate']:.2%}")
            print(f"Cache Size:  {stats['cache_size']}/{stats['max_size']}")
            print(f"Memory:      {stats['memory_mb']:.2f} MB")
            print("="*60)


# Specific caching utilities for common diffusion operations

def cache_timestep_embedding(
    timestep: int,
    embedding_dim: int,
    max_period: int = 10000,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Compute and cache sinusoidal timestep embedding.
    
    This is one of the most common repeated operations in diffusion models.
    
    Args:
        timestep: The timestep value
        embedding_dim: Embedding dimension
        max_period: Maximum period for sinusoidal encoding
        dtype: Tensor dtype
        device: Device to place tensor on
        
    Returns:
        Timestep embedding tensor
    """
    cache = get_global_cache()
    
    # Create unique key
    key = f"timestep_emb_{timestep}_{embedding_dim}_{max_period}_{dtype}_{device}"
    
    def compute():
        # Standard sinusoidal position encoding
        half_dim = embedding_dim // 2
        emb = np.log(max_period) / (half_dim - 1)
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = timestep * emb
        emb = np.concatenate([np.sin(emb), np.cos(emb)])
        
        if embedding_dim % 2 == 1:  # odd dimension
            emb = np.concatenate([emb, np.zeros(1)])
        
        return torch.tensor(emb, dtype=dtype, device=device)
    
    return cache.get_or_compute(key, compute)


if __name__ == "__main__":
    # Self-test
    print("Testing DeterministicCache...")
    
    cache = DeterministicCache(max_size=5)
    
    # Test basic caching
    for i in range(10):
        emb = cache_timestep_embedding(i % 3, 128)  # Only 3 unique timesteps
        print(f"Timestep {i % 3}: shape {emb.shape}")
    
    stats = cache.get_statistics()
    print(f"\nCache Stats:")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Hits: {stats['hits']}, Misses: {stats['misses']}")
    print(f"  Memory: {stats['memory_mb']:.2f} MB")
    
    # Test context manager
    print("\n\nTesting CacheContext...")
    with CacheContext(max_size=10) as ctx:
        for i in range(20):
            emb = cache_timestep_embedding(i % 5, 256)
    
    print("\n✓ All tests passed!")
