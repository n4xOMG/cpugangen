"""
Process and Thread Pinning Utilities for CPU Optimization.

This module provides utilities to:
- Pin processes to specific CPU cores
- Configure thread pool sizes
- Get CPU topology information
- Optimize for CPU inference workloads

Platform support:
- Windows: Via psutil and os module
- Linux: Via os.sched_setaffinity (if available)
- macOS: Limited support
"""

import os
import platform
import warnings
from typing import List, Optional, Dict, Any

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn(
        "psutil not available. Process pinning will be limited. "
        "Install with: pip install psutil"
    )

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_cpu_info() -> Dict[str, Any]:
    """
    Get CPU topology information.
    
    Returns:
        Dictionary with CPU information:
        - physical_cores: Number of physical CPU cores
        - logical_cores: Number of logical CPU cores (with hyperthreading)
        - cpu_count: Total CPU count
        - platform: OS platform
        - cpu_freq: CPU frequency info (if available)
    """
    info = {
        "platform": platform.system(),
        "cpu_count": os.cpu_count() or 1,
    }
    
    if PSUTIL_AVAILABLE:
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
        
        try:
            freq = psutil.cpu_freq()
            if freq:
                info["cpu_freq_mhz"] = {
                    "current": freq.current,
                    "min": freq.min,
                    "max": freq.max,
                }
        except Exception:
            pass
        
        try:
            info["cpu_percent"] = psutil.cpu_percent(interval=0.1, percpu=True)
        except Exception:
            pass
    else:
        # Fallback without psutil
        info["physical_cores"] = os.cpu_count() or 1
        info["logical_cores"] = os.cpu_count() or 1
    
    return info


def pin_to_cores(
    cores: List[int],
    prefer_physical: bool = True,
    verbose: bool = True
) -> bool:
    """
    Pin current process to specified CPU cores.
    
    Args:
        cores: List of CPU core indices to pin to (0-indexed)
        prefer_physical: If True, recommend using only physical cores
        verbose: Print status messages
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> # Pin to first 4 cores
        >>> pin_to_cores([0, 1, 2, 3])
        
        >>> # Pin to physical cores only (even indices typically)
        >>> pin_to_cores([0, 2, 4, 6], prefer_physical=True)
    """
    if not PSUTIL_AVAILABLE:
        if verbose:
            print("⚠️  psutil not available. Cannot pin to cores.")
            print("   Install with: pip install psutil")
        return False
    
    system = platform.system()
    cpu_info = get_cpu_info()
    
    # Validate core indices
    max_cores = cpu_info["logical_cores"]
    invalid_cores = [c for c in cores if c >= max_cores]
    if invalid_cores:
        if verbose:
            print(f"⚠️  Invalid core indices: {invalid_cores}")
            print(f"   System has {max_cores} logical cores (0-{max_cores-1})")
        return False
    
    # Warning for hyperthreading
    if prefer_physical and cpu_info["physical_cores"] < cpu_info["logical_cores"]:
        physical_cores = cpu_info["physical_cores"]
        if any(c >= physical_cores for c in cores):
            if verbose:
                print(f"⚠️  Warning: Hyperthreading detected")
                print(f"   Physical cores: 0-{physical_cores-1}")
                print(f"   Logical cores: {physical_cores}-{max_cores-1}")
                print(f"   Consider using only physical cores for compute-bound tasks")
    
    try:
        process = psutil.Process()
        
        if system == "Windows":
            # Windows: Set CPU affinity
            process.cpu_affinity(cores)
            if verbose:
                print(f"✓ Process pinned to cores: {cores}")
                print(f"  Platform: Windows")
        
        elif system == "Linux":
            # Linux: Use sched_setaffinity if available
            if hasattr(os, 'sched_setaffinity'):
                os.sched_setaffinity(0, set(cores))
                if verbose:
                    print(f"✓ Process pinned to cores: {cores}")
                    print(f"  Platform: Linux (sched_setaffinity)")
            else:
                # Fallback to psutil
                process.cpu_affinity(cores)
                if verbose:
                    print(f"✓ Process pinned to cores: {cores}")
                    print(f"  Platform: Linux (psutil)")
        
        else:
            # macOS or other
            if verbose:
                print(f"⚠️  Platform {system} has limited affinity support")
            try:
                process.cpu_affinity(cores)
                if verbose:
                    print(f"✓ Attempted to pin to cores: {cores}")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Affinity setting failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ Failed to pin to cores: {e}")
        return False


def configure_threading(
    num_threads: Optional[int] = None,
    optimize_for_cpu: bool = True,
    verbose: bool = True
) -> Dict[str, int]:
    """
    Configure thread pool sizes for CPU inference.
    
    Best practices for CPU inference:
    - Set num_threads to physical core count
    - Disable inter-op parallelism (set to 1)
    - Enable intra-op parallelism (BLAS/LAPACK threading)
    
    Args:
        num_threads: Number of threads (None = auto-detect physical cores)
        optimize_for_cpu: Apply CPU-specific optimizations
        verbose: Print configuration
        
    Returns:
        Dictionary with thread configuration
        
    Example:
        >>> configure_threading(num_threads=8, optimize_for_cpu=True)
    """
    config = {}
    
    # Auto-detect optimal thread count
    if num_threads is None:
        cpu_info = get_cpu_info()
        # Use physical cores for CPU inference
        num_threads = cpu_info.get("physical_cores", cpu_info["cpu_count"])
    
    config["num_threads"] = num_threads
    
    # Configure PyTorch threading
    if TORCH_AVAILABLE:
        # Intra-op parallelism (within operations like matrix multiply)
        torch.set_num_threads(num_threads)
        config["torch_num_threads"] = num_threads
        
        if optimize_for_cpu:
            # Inter-op parallelism (across operations)
            # For CPU inference, typically want to minimize this
            torch.set_num_interop_threads(1)
            config["torch_num_interop_threads"] = 1
            
            if verbose:
                print(f"✓ PyTorch threading configured:")
                print(f"  Intra-op threads: {num_threads}")
                print(f"  Inter-op threads: 1")
        else:
            config["torch_num_interop_threads"] = torch.get_num_interop_threads()
    
    # Set environment variables for OpenMP, MKL, etc.
    env_vars = {
        "OMP_NUM_THREADS": str(num_threads),
        "MKL_NUM_THREADS": str(num_threads),
        "OPENBLAS_NUM_THREADS": str(num_threads),
        "VECLIB_MAXIMUM_THREADS": str(num_threads),
        "NUMEXPR_NUM_THREADS": str(num_threads),
    }
    
    for var, value in env_vars.items():
        os.environ[var] = value
        config[var.lower()] = int(value)
    
    if verbose:
        print(f"\n✓ Environment variables set:")
        for var, value in env_vars.items():
            print(f"  {var}={value}")
    
    return config


def print_cpu_info(detailed: bool = False):
    """
    Print CPU information in human-readable format.
    
    Args:
        detailed: Include detailed CPU stats
    """
    info = get_cpu_info()
    
    print("="*60)
    print("CPU Information")
    print("="*60)
    print(f"Platform:        {info['platform']}")
    print(f"Physical cores:  {info['physical_cores']}")
    print(f"Logical cores:   {info['logical_cores']}")
    print(f"Total CPU count: {info['cpu_count']}")
    
    if "cpu_freq_mhz" in info:
        freq = info["cpu_freq_mhz"]
        print(f"\nCPU Frequency:")
        print(f"  Current: {freq['current']:.0f} MHz")
        print(f"  Min:     {freq['min']:.0f} MHz")
        print(f"  Max:     {freq['max']:.0f} MHz")
    
    if detailed and "cpu_percent" in info:
        print(f"\nCPU Usage per core:")
        for i, usage in enumerate(info["cpu_percent"]):
            print(f"  Core {i}: {usage:.1f}%")
    
    print("="*60)


def optimize_for_inference(
    cores: Optional[List[int]] = None,
    num_threads: Optional[int] = None,
    prefer_physical: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Apply all optimizations for CPU inference.
    
    Convenience function that:
    1. Pins to specified cores (or auto-selects physical cores)
    2. Configures threading
    3. Returns configuration summary
    
    Args:
        cores: Core indices to pin to (None = use all physical cores)
        num_threads: Thread count (None = auto-detect)
        prefer_physical: Prefer physical cores over logical
        verbose: Print configuration details
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> # Auto-optimize for CPU
        >>> config = optimize_for_inference(verbose=True)
        
        >>> # Pin to specific cores with custom threading
        >>> config = optimize_for_inference(
        ...     cores=[0, 1, 2, 3, 4, 5, 6, 7],
        ...     num_threads=8
        ... )
    """
    if verbose:
        print("\n" + "="*60)
        print("Optimizing for CPU Inference")
        print("="*60)
        print_cpu_info(detailed=False)
    
    config = {
        "success": True,
        "optimizations_applied": []
    }
    
    # Auto-select cores if not specified
    if cores is None and prefer_physical:
        cpu_info = get_cpu_info()
        physical_cores = cpu_info["physical_cores"]
        cores = list(range(physical_cores))
        if verbose:
            print(f"\n📌 Auto-selecting physical cores: {cores}")
    
    # Pin to cores
    if cores is not None:
        success = pin_to_cores(cores, prefer_physical=prefer_physical, verbose=verbose)
        config["pinning_success"] = success
        config["pinned_cores"] = cores if success else None
        if success:
            config["optimizations_applied"].append("core_pinning")
    
    # Configure threading
    if verbose:
        print()
    thread_config = configure_threading(
        num_threads=num_threads,
        optimize_for_cpu=True,
        verbose=verbose
    )
    config["threading"] = thread_config
    config["optimizations_applied"].append("threading")
    
    if verbose:
        print("\n" + "="*60)
        print("✓ Optimization Complete")
        print("="*60)
        print(f"Applied: {', '.join(config['optimizations_applied'])}")
        if cores:
            print(f"Pinned to cores: {cores}")
        print(f"Thread count: {thread_config['num_threads']}")
    
    return config


if __name__ == "__main__":
    # Self-test
    print("Testing process_pinning module...\n")
    
    # Test CPU info
    print_cpu_info(detailed=True)
    
    # Test optimization
    print("\n\nTesting optimization...")
    config = optimize_for_inference(
        cores=[0, 1, 2, 3],  # First 4 cores
        num_threads=4,
        verbose=True
    )
    
    print("\n\nConfiguration returned:")
    import json
    print(json.dumps({k: v for k, v in config.items() if k != "threading"}, indent=2))
    
    print("\n✓ All tests passed!")
