# Hypothesis 6: Deterministic Operator Caching & Process Pinning

## Quick Start Guide

This guide shows how to use deterministic caching and process pinning optimizations for CPU inference.

### 1. Basic Usage with Caching

Caching is **automatically enabled** for CPU inference:

```python
from hqpd.models.sdxl_toe_pipeline import create_toe_pipeline

# Load pipeline (caching auto-enabled for CPU)
pipeline = create_toe_pipeline(
    toe_checkpoint_path="checkpoints/toe/toe_with_pooling_best.pt",
    vocab_path="data/vocabulary.json",
    sdxl_model_path="segmind/SSD-1B",
    device="cpu"
)

# Generate image - caching happens automatically
image = pipeline(
    prompt="1girl, blue_hair, anime_style",
    num_inference_steps=25,
    guidance_scale=7.5
).images[0]
```

### 2. Manual Cache Control

```python
from hqpd.optimization import CacheContext

# Use cache context for fine-grained control
with CacheContext(
    enabled=True,
    max_size=256,  # Larger cache
    clear_on_enter=True,  # Start fresh
    print_stats_on_exit=True  # See cache performance
) as cache:
    # Run inference
    image = pipeline(prompt="...", num_inference_steps=25).images[0]
    
    # Access stats during inference
    stats = cache.get_statistics()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

### 3. Process Pinning

Pin inference to specific CPU cores for reduced jitter:

```python
from hqpd.optimization import optimize_for_inference, pin_to_cores

# Option 1: Auto-optimize (uses all physical cores)
config = optimize_for_inference(
    cores=None,  # Auto-select physical cores
    num_threads=None,  # Auto-detect
    verbose=True
)

# Option 2: Manual core selection
pin_to_cores([0, 2, 4, 6], prefer_physical=True)  # Even cores = physical

# Option 3: Custom threading
from hqpd.optimization import configure_threading
configure_threading(num_threads=8, optimize_for_cpu=True)
```

### 4. Combined Optimization

```python
from hqpd.models.sdxl_toe_pipeline import create_toe_pipeline
from hqpd.optimization import optimize_for_inference, CacheContext

# Step 1: Apply process optimizations
optimize_for_inference(cores=[0, 1, 2, 3], num_threads=4)

# Step 2: Load pipeline with caching
pipeline = create_toe_pipeline(
    toe_checkpoint_path="checkpoints/toe/toe_with_pooling_best.pt",
    vocab_path="data/vocabulary.json",
    sdxl_model_path="segmind/SSD-1B",
    device="cpu",
    enable_caching=True,  # Explicitly enable
    cache_size=128
)

# Step 3: Run inference with cache tracking
with CacheContext(enabled=True) as cache:
    for i in range(10):
        image = pipeline(
            prompt=f"prompt_{i}",
            num_inference_steps=25
        ).images[0]
        image.save(f"output_{i}.png")
    
    # Check results
    stats = cache.get_statistics()
    print(f"Total speedup from caching: ~{stats['hit_rate']*0.15:.1%}")
```

## Benchmarking

### Profile to Identify Cacheable Operations

```bash
# Use venv Python
d:\LTPython\scripts\cpugangen\venv\Scripts\python.exe scripts/profile_inference.py \
    --toe_checkpoint checkpoints/toe/toe_with_pooling_best.pt \
    --vocab data/vocabulary.json \
    --steps 25 \
    --output outputs/profiling/
```

This will:
- Profile the entire inference pipeline
- Identify frequently called functions
- Suggest cacheable operations
- Save detailed report

### Comprehensive Benchmark

```bash
d:\LTPython\scripts\cpugangen\venv\Scripts\python.exe scripts/benchmark_with_caching.py \
    --toe_checkpoint checkpoints/toe/toe_with_pooling_best.pt \
    --vocab data/vocabulary.json \
    --steps 25 \
    --num-samples 20 \
    --cache-size 128 \
    --cores 0 1 2 3 \
    --num-threads 4 \
    --output outputs/caching_benchmark/
```

This compares:
- Baseline (no optimizations)
- Caching only
- Full optimization (caching + pinning)

## Configuration Options

### Cache Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_caching` | Auto (True for CPU) | Enable/disable caching |
| `cache_size` | 128 | Maximum cached entries (LRU eviction) |
| `clear_on_enter` | False | Clear cache when entering context |
| `print_stats_on_exit` | True | Print stats when exiting context |

### Process Pinning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cores` | None (all physical) | List of CPU core indices [0, 1, 2, ...] |
| `num_threads` | None (auto-detect) | Thread pool size |
| `prefer_physical` | True | Warn if using hyperthreaded cores |

### Recommended Settings

**For maximum speed** (4-core CPU):
```python
cores = [0, 1, 2, 3]  # All physical cores
num_threads = 4
cache_size = 256
```

**For consistency** (reduce jitter):
```python
cores = [0, 2]  # Fewer cores, less contention
num_threads = 2
cache_size = 128
```

**For memory-constrained systems**:
```python
cache_size = 64  # Smaller cache
cores = [0, 1]  # Fewer cores
```

## Expected Performance

Based on hypothesis and testing:

| Optimization | Expected Speedup | Jitter Reduction |
|-------------|------------------|------------------|
| Caching only | 8-12% | 30-40% |
| Pinning only | 5-10% | 20-30% |
| **Combined** | **12-20%** | **40-60%** |

**Note**: Windows has less benefit from process pinning compared to Linux (no taskset/jemalloc). Caching provides most of the speedup.

## Troubleshooting

### Cache not working

```python
from hqpd.optimization import get_global_cache

cache = get_global_cache()
stats = cache.get_statistics()
print(stats)  # Check if enabled and getting hits
```

### Process pinning failed

```python
# Check if psutil is installed
pip install psutil

# Verify core count
from hqpd.optimization import get_cpu_info
print(get_cpu_info())
```

### Low cache hit rate

- Increase cache size: `cache_size=256`
- Check if operations are actually deterministic
- Review profiling output to identify issues

## API Reference

See docstrings in:
- `hqpd/optimization/deterministic_cache.py` - Caching utilities
- `hqpd/optimization/process_pinning.py` - Process affinity
- `hqpd/models/sdxl_toe_pipeline.py` - Pipeline integration

## Implementation Details

**What is cached:**
- Timestep embeddings (sinusoidal position encodings)
- Repeated tensor operations with identical inputs
- Any operation wrapped with `@cached_operation` decorator

**What is NOT cached:**
- Random noise generation
- Non-deterministic operations
- Model weights (too large)
- Intermediate activations (memory intensive)

**Cache invalidation:**
- Automatic LRU eviction when full
- Manual clear via `cache.clear()`
- Context manager handles cleanup

**Thread safety:**
- All cache operations are thread-safe
- Safe for multi-threaded inference
- Lock contention is minimal (fast operations)
