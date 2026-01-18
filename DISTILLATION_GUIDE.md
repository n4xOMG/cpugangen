# Phase 3: Knowledge Distillation Training - Setup Guide

## What You'll Need

✅ **Dataset:** 35k anime images + JSON metadata (you're collecting)  
✅ **GPU:** RTX 3090 24GB VRAM (you have)  
✅ **Storage:** ~50-100GB for dataset + checkpoints  
✅ **Time:** ~1-2 weeks training (10 epochs)

---

## Quick Start

### Step 1: Organize Your Dataset

```bash
# Expected structure:
cpugangen/
├── data/
│   ├── images/                    # Your 35k images
│   │   ├── 0000f2fd041da56922b3903dd38b99c9.jpg
│   │   ├── 000335b27175b164a8c7920fa1ea64ec.jpg
│   │   └── ...
│   └── metadata.json              # Your JSON with tags
└── ...
```

### Step 2: Create Train/Val Split

```bash
python scripts/dataset_anime.py \
  --metadata data/metadata.json \
  --images-dir data/images \
  --create-split \
  --split-output data/splits
```

**Output:**
- `data/splits/train_metadata.json` (95% of data)
- `data/splits/val_metadata.json` (5% of data)

### Step 3: Test Dataset Loading

```bash
python scripts/dataset_anime.py \
  --metadata data/splits/train_metadata.json \
  --images-dir data/images
```

**Expected output:**
```
Dataset size: 33250
First sample:
  Filename: 0000f2fd041da56922b3903dd38b99c9.jpg
  Image shape: torch.Size([3, 1024, 1024])
  Prompt: hatsune miku, 1girl, solo, twintails, long_hair, spring_onion...
```

### Step 4: Configure Training

Edit `configs/distillation_config.json`:

```json
{
  "data": {
    "train_metadata": "data/splits/train_metadata.json",  ← Update paths
    "val_metadata": "data/splits/val_metadata.json",
    "images_dir": "data/images"                            ← Update path
  },
  "training": {
    "batch_size": 1,      ← Keep at 1 for 3090 24GB
    "num_epochs": 10      ← Adjust based on time available
  }
}
```

### Step 5: Install Dependencies

```bash
# If not already installed
pip install wandb  # For training monitoring (optional)
wandb login  # If using WandB
```

Or disable WandB in config:
```json
{
  "logging": {
    "use_wandb": false  ← Set to false
  }
}
```

### Step 6: Start Training!

```bash
python scripts/train_distillation.py \
  --config configs/distillation_config.json
```

**Expected output:**
```
Using device: cuda
Loading teacher model: martineux/janku6
Loading student model: segmind/SSD-1B

Teacher UNet: 2,568,741,888 parameters
Student UNet: 1,284,370,944 parameters
Reduction: 50.0%

Train samples: 33250
Val samples: 1750

Starting Training...
Epoch 1/10
100%|████████| 33250/33250 [2:45:00<00:00]
Train Loss: 0.0245
Val Loss: 0.0198
✅ Saved best model! Val loss: 0.0198
```

---

## Training Timeline

**Per epoch on 3090:**
- ~2.5-3 hours (batch_size=1, 35k images)

**Total for 10 epochs:**
- ~25-30 hours (~1-2 days continuous)
- Or ~1 week if running overnight

**Recommendations:**
- Use `screen` or `tmux` to keep training running if disconnecting
- Monitor with WandB or check logs periodically
- Best model saved automatically when validation improves

---

## While Training Runs

### Monitor Progress

**Option 1: WandB Dashboard**
- Visit https://wandb.ai/your-username/anime-student-distillation
- See real-time loss curves, learning rate, etc.

**Option 2: Local Logs**
- Training prints progress to console
- Loss values updated every batch

### Check Checkpoints

```bash
checkpoints/anime_student/
├── best_student_unet.pt           # Best model (lowest val loss)
├── student_unet_epoch2.pt         # Periodic checkpoints
├── student_unet_epoch4.pt
└── ...
```

### Early Stopping

If validation loss stops improving:
- Stop training early (don't need all 10 epochs)
- Use the best checkpoint

---

## After Training

### Test the Anime-Specialized Student

Use the same test script but with your trained model:

```bash
# First, create a full pipeline with your student UNet
# Then test it like we did with generic SSD-1B

python scripts/test_trained_student.py \
  --student-checkpoint checkpoints/anime_student/best_student_unet.pt \
  --benchmark \
  --compare-quality
```

**Expected results:**
- Speed: ~2.3x faster (same as SSD-1B architecture)
- Quality: Better than generic SSD-1B (anime-specialized!)
- Style: More anime-like, less realistic

---

## Troubleshooting

### "CUDA out of memory"

**Solutions:**
1. Reduce batch size in config (already at 1, can't go lower)
2. Reduce resolution: `"resolution": 768` instead of 1024
3. Use gradient checkpointing (add to training script)

### "Training is too slow"

**Expected:** ~2.5-3 hours per epoch is normal for distillation
**If slower:**
- Check GPU utilization: `nvidia-smi`
- Reduce num_workers if CPU bottleneck
- Ensure data is on fast SSD

### "Validation loss not improving"

**Try:**
- Reduce learning rate: `1e-6` instead of `1e-5`
- Increase feature_weight: `1.0` instead of `0.5`
- Check if dataset has issues (corrupted images, bad tags)

---

## Dataset Format Notes

Your JSON format is perfect! The script handles:

- ✅ `filename` - Image filename
- ✅ `caption` - Comma-separated tags (used as prompt)
- ✅ `general_tags` - Array of tags (fallback)
- ✅ `character_tags` - Character names (prepended to prompt)
- ✅ `general_scores` - Tag confidence scores (filtered by min_tag_score)

**Tag Priority:**
1. Character tags (if present)
2. Caption string
3. General tags filtered by score threshold

**Adjustable in dataset_anime.py:**
- `max_tags=75` - Maximum tags per prompt
- `min_tag_score=0.35` - Minimum confidence to include tag
- `use_caption=True` - Use caption field
- `use_character_tags=True` - Prepend character tags

---

## Next Steps After Training

1. ✅ Test anime-specialized student vs generic SSD-1B
2. ✅ Measure quality improvement
3. ✅ Deploy best model for CPU inference
4. Optional: Try INT8 quantization for additional speedup

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/dataset_anime.py` | Dataset loader for JSON metadata |
| `scripts/train_distillation.py` | Main training script |
| `configs/distillation_config.json` | Training configuration |
| `DISTILLATION_GUIDE.md` | This guide |

---

## Estimated Costs

**GPU Time (3090 on cloud):**
- ~$1-2/hour on vast.ai
- ~25-30 hours training
- **Total: ~$25-60**

**Storage:**
- Dataset: ~20-30GB
- Checkpoints: ~20-30GB
- **Total: ~50GB**

---

## Questions?

- Dataset issues: Check `dataset_anime.py` code
- Training issues: Check `train_distillation.py` code
- Config issues: Edit `distillation_config.json`

Ready to start when your 35k images are ready! 🚀
