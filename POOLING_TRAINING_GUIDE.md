# Training TOE Pooling Head - Quick Guide

## What This Does

Add a pooling head to TOE so it can generate pooled embeddings, eliminating CLIP dependency completely.

**Goal:** Full TOE mode (no CLIP needed!)

## Steps

### Step 1: Extract CLIP Pooled Embeddings (~5 min)

```bash
cd d:/LTPython/scripts/cpugangen

# Generate 10k training samples
python scripts/extract_clip_pooled.py --num_samples 10000
```

**What it does:**
- Loads CLIP text encoder
- Generates random tag combinations from vocabulary
- Extracts pooled embeddings (1280-dim)
- Saves to `data/clip_pooled_embeddings.pt`

**Expected time:** ~5 minutes on GPU

---

### Step 2: Train Pooling Head (~30-60 min)

```bash
# Train on 3090
python scripts/train_toe_pooling.py \
    --checkpoint checkpoints/toe/toe_best.pt \
    --embeddings data/clip_pooled_embeddings.pt \
    --batch_size 64 \
    --epochs 10
```

**What it does:**
- Loads your pre-trained TOE (frozen)
- Adds pooling head (trainable, ~1.6M params)
- Trains to match CLIP pooled embeddings
- Saves best model to `checkpoints/toe/toe_with_pooling_best.pt`

**Expected:** 
- Time: 30-60 minutes on RTX 3090
- Final cosine similarity: > 0.95 (good), > 0.98 (excellent)

---

### Step 3: Test Full TOE Mode

```bash
# Generate with full TOE (no CLIP!)
python scripts/integrate_toe_full.py \
    --checkpoint checkpoints/toe/toe_with_pooling_best.pt \
    --hybrid False  # Full TOE mode!
```

**Expected:**
- Images generate successfully ✅
- No CLIP dependency ✅
- Additional ~50-100ms speedup ✅
- -1.5GB memory savings ✅

---

## Monitoring Training

Watch for:
- **MSE loss**: Should decrease to < 0.01
- **Cosine similarity**: Should increase to > 0.95
- **Converges in ~5-10 epochs** typically

Example good training:
```
Epoch 1: Loss: 0.245, MSE: 0.180, Cos: 0.850
Epoch 5: Loss: 0.042, MSE: 0.012, Cos: 0.960
Epoch 10: Loss: 0.018, MSE: 0.005, Cos: 0.982  ← Excellent!
```

---

## Troubleshooting

**"CUDA out of memory"**
```bash
# Reduce batch size
python scripts/train_toe_pooling.py --batch_size 32
```

**"Embeddings file not found"**
```bash
# Run step 1 first
python scripts/extract_clip_pooled.py
```

**"Images look worse with full TOE"**
- Check cosine similarity > 0.95
- May need more training epochs
- Can fall back to hybrid mode

---

## Success Criteria

✅ Training complete
✅ Cosine similarity > 0.95
✅ Images generate with `--hybrid False`
✅ Visual quality acceptable

---

## Next Steps After This

**Option A:** Deploy full TOE
- Use `toe_with_pooling_best.pt` in production
- No CLIP dependency
- Maximum memory savings

**Option B:** Continue to Phase 2
- VQ-UNet distillation (HQPD Phase 2)
- Further speedup + size reduction
- Ultimate goal: <2s generation on CPU

**Option C:** Both!
- Deploy TOE now
- Work on Phase 2 in parallel
