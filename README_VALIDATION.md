# SDXL-Lightning PyTorch Inference

## Overview

This project uses **PyTorch** for SDXL-Lightning inference on CPU. ONNX was tested but did not provide sufficient performance benefits on CPU.

## Quick Start

### Generate Images

**Single image:**
```bash
python scripts/test_pytorch.py --prompt "anime girl with blue hair, highly detailed"
```

**Multiple images with different seeds:**
```bash
python scripts/test_pytorch.py --prompt "beautiful landscape" --num-images 5
```

**Custom seed for reproducibility:**
```bash
python scripts/test_pytorch.py --seed 42 --prompt "cyber punk city"
```

### Benchmark Performance

Run performance benchmarks:
```bash
python scripts/test_pytorch.py --benchmark --num-images 5
```

This will:
- ✅ Run 5 generation iterations
- ✅ Measure time for each iteration
- ✅ Calculate average, min, max times
- ✅ Save results to `outputs/pytorch/pytorch_benchmark.txt`

## Script Arguments

### test_pytorch.py

```bash
--model MODEL          Base model (default: martineux/janku6)
--steps {2,4,8}        Lightning steps (default: 4)
--prompt TEXT          Generation prompt
--seed INT             Random seed (default: 42)
--output DIR           Output directory (default: outputs/pytorch)
--num-images INT       Number of images (default: 1)
--benchmark            Enable benchmark mode
```

## Examples

### High Quality Anime
```bash
python scripts/test_pytorch.py \
  --prompt "anime girl, beautiful eyes, detailed face, studio lighting" \
  --seed 100 \
  --steps 4
```

### Batch Generation
```bash
python scripts/test_pytorch.py \
  --prompt "fantasy landscape, mountains, sunset" \
  --num-images 10 \
  --seed 1
```

### Performance Testing
```bash
python scripts/test_pytorch.py \
  --benchmark \
  --num-images 10 \
  --prompt "test prompt"
```

## Output Structure

```
outputs/
└── pytorch/
    ├── pytorch_42.png              # Generated images
    ├── pytorch_43.png
    └── pytorch_benchmark.txt       # Performance stats (if --benchmark)
```

## Performance

### Expected CPU Performance
- **4-step generation**: ~15-30 seconds per image (depends on CPU)
- **Model loading**: ~10-20 seconds (one-time cost)
- **Memory usage**: ~4-6 GB RAM

### Optimization Tips
1. Use `--steps 2` for faster generation (lower quality)
2. Use `--steps 8` for higher quality (slower)
3. Batch multiple generations to amortize loading time
4. Close other applications to free up CPU cycles

## Model Information

- **Base Model**: martineux/janku6 (SDXL-based)
- **Lightning LoRA**: ByteDance/SDXL-Lightning (4-step by default)
- **Scheduler**: Euler Discrete (trailing timesteps)
- **Guidance**: CFG disabled (guidance_scale=0)

## Troubleshooting

**Q: Out of memory**
- Close other applications
- Restart Python session before running
- Use a smaller base model if available

**Q: Too slow**
- Reduce steps: `--steps 2`
- This is expected on CPU (GPU is 10-20x faster)
- Consider cloud GPU if speed is critical

**Q: Poor image quality**
- Increase steps: `--steps 8`
- Improve prompt with more details
- Try different seeds: `--seed <number>`

## Next Steps

1. Test the basic generation: `python scripts/test_pytorch.py`
2. Run benchmark to measure your CPU performance
3. Experiment with different prompts and seeds
4. For production, consider GPU deployment for better performance
