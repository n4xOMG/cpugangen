# UNet Distillation Quick Start

## Overview

Distill SDXL UNet from 2.6B → 1.3B params (50% reduction) using Segmind SSD-1B architecture.

---

## Option A: Quick Start (Use Pre-trained Segmind)

**Fastest path - use already-distilled model:**

```bash
# Test with synthetic data
python scripts/train_distilled_unet.py \
    --use_segmind \
    --use_synthetic \
    --max_steps 1000

# Or load directly for inference
python -c "
from hqpd.distillation import load_segmind_ssd1b
unet = load_segmind_ssd1b(device='cuda')
print(f'Loaded: {sum(p.numel() for p in unet.parameters()):,} params')
"
```

---

## Option B: Custom Distillation

**Full training for best quality:**

### Step 1: Prepare Data (~1-2 hours)

```bash
# With images + prompts
python scripts/prepare_distillation_data.py \
    --image_dir data/images \
    --output_dir data/latents \
    --num_samples 10000

# Or: Just generate synthetic data (for testing)
# Training script can generate on-the-fly
```

### Step 2: Train (~40-50 hours on 3090)

```bash
python scripts/train_distilled_unet.py \
    --data_dir data/latents \
    --output_dir checkpoints/distilled_unet \
    --max_steps 50000 \
    --batch_size 1 \
    --gradient_accumulation 8
```

### Step 3: Evaluate

```bash
# Coming soon: eval script
python scripts/eval_distilled_unet.py \
    --checkpoint checkpoints/distilled_unet/checkpoint_final.pt
```

---

## Files Created

```
hqpd/distillation/
├── __init__.py           # Module exports
├── distilled_unet.py     # Architecture + weight transfer
├── losses.py             # Multi-component loss
└── trainer.py            # Training loop

scripts/
├── train_distilled_unet.py       # Main training
├── prepare_distillation_data.py  # Data prep
```

---

## Key Features

**Architecture:**
- Segmind SSD-1B style (1.3B params)
- Reduced transformer blocks (10 → 4 in deepest stage)
- No mid-block cross-attention
- Weight transfer from teacher

**Training:**
- Multi-component loss (output + feature + cosine)
- Progressive loss scheduling
- EMA for stable checkpoints
- Gradient checkpointing + xFormers

**CPU Optimization (added after distillation):**
- Channels-last memory format
- INT8 quantization ready
- Memory-efficient attention

---

## Expected Results

| Metric | Full SDXL | Distilled |
|--------|-----------|-----------|
| Params | 2.6B | 1.3B |
| VRAM | ~8GB | ~4GB |
| Speed (GPU) | 6.5s | ~3-4s |
| Speed (CPU) | ~180s | ~80-100s |
