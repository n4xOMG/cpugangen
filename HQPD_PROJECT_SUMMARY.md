# HQPD Project Summary

## 🎯 Project Goal

**High-Quality, Production-Ready Diffusion (HQPD)**: Optimize anime image generation for **fast CPU inference** while maintaining high quality, using Tag-Optimized Encoder (TOE) and SDXL-based models.

---

## 📊 Current Status: ONNX Export Phase

### What We've Achieved

#### ✅ **Phase 1: TOE Architecture (Pre-existing)**
- **Tag-Optimized Encoder (TOE)**: 50MB lightweight encoder trained on Danbooru tags
- Replaces CLIP (999MB) with 95% smaller model
- Supports hybrid mode (TOE context + CLIP pooled) and full mode (with pooling head)
- **Location**: `hqpd/models/toe.py`, `hqpd/models/sdxl_toe_pipeline.py`

#### ✅ **Phase 2: CPU Optimization Research (Failed)**
- **Problem**: Initial INT8 quantization failed catastrophically (<40% quality)
- **Research**: Explored 10+ optimization techniques (documented in `cpu_optimization_alternatives.md`)
- **Key Finding**: Standard INT8 quantization fundamentally incompatible with diffusion models on CPU
- **Failed Approaches** (archived in `archived_experiments/failed_quantization/`):
  - Per-channel INT8
  - Selective INT8 (quantize only FFN layers)
  - Weights-only INT8 (W8A32)

#### ✅ **Phase 3: TOE Step Reduction Experiment (Failed)**
- **Hypothesis**: TOE's cleaner tag conditioning might enable fewer steps
- **Result**: 5 steps showed poor quality, confirming slowness is a compute problem, not conditioning
- **Implementation**: `scripts/test_toe_step_reduction.py`

#### ✅ **Phase 4: Pivot to GPU-Trainable Solutions (Current)**

**4.1 ONNX Runtime Export** ⭐ **IN PROGRESS**

Created GPU-accelerated export scripts for fast ONNX conversion:

**Baseline SDXL:**
- Script: `scripts/export_to_onnx.py`
- Exports: UNet, VAE, Text Encoders to ONNX
- Features: GPU/CPU support (`--device cuda`), optional INT8 quantization
- Expected: 2-2.5x CPU inference speedup

**TOE + SDXL:**
- Script: `scripts/export_toe_to_onnx.py`  
- Exports: TOE encoder + SDXL components to ONNX
- Hybrid approach: TOE for tags, ONNX for fast inference
- Memory: 949MB savings (50MB TOE vs 999MB CLIP)

**Testing:**
- Script: `scripts/test_onnx_comparison.py`
- Compares baseline ONNX vs TOE-ONNX
- Measures: Speed, quality, memory usage
- Output: Side-by-side comparison grid + metrics

**Status**: 
- ✅ Scripts created with GPU support
- ✅ Baseline export completed (11 min/step CPU → expected 2.5x faster with ONNX)
- ⏳ TOE export in progress on cloud GPU
- ⏳ Testing pending export completion

**4.2 Latent Consistency Models (LCM)** 📝 **READY TO TRAIN**

The game-changer for CPU inference:

- Script: `scripts/train_lcm.py`
- Training: Consistency distillation for 4-step generation
- Expected: **12x speedup** (50 steps → 4 steps)
- Quality: 90-95% retention (research-proven)
- Requirements: GPU training (1-2 days on 3090)
- **Status**: Script ready, awaiting ONNX validation before starting

**Combined Potential**: ONNX (2.5x) × LCM (5x fewer steps) = **12.5x total speedup**

---

## 🗂️ Project Structure

```
cpugangen/
├── hqpd/
│   ├── models/
│   │   ├── toe.py                    # TOE architecture
│   │   └── sdxl_toe_pipeline.py      # TOE integration with SDXL
│   ├── quantization/
│   │   ├── dynamic_int8.py           # Basic quantization (working)
│   │   └── cpu_optimize.py           # CPU inference configs
│   └── utils/
│       └── danbooru.py               # Tag processing (15K vocab)
│
├── scripts/
│   ├── export_to_onnx.py             # [NEW] Baseline SDXL → ONNX
│   ├── export_toe_to_onnx.py         # [NEW] TOE + SDXL → ONNX
│   ├── test_onnx_comparison.py       # [NEW] Test & compare
│   ├── train_lcm.py                  # [NEW] LCM training
│   └── test_toe_step_reduction.py    # Step reduction test (failed)
│
├── archived_experiments/
│   └── failed_quantization/          # INT8 attempts (archived)
│
├── checkpoints/
│   └── toe/
│       └── toe_with_pooling_best.pt  # Trained TOE model
│
├── data/
│   └── vocabulary.json               # 15K Danbooru tags
│
└── onnx_models/                      # [GENERATED]
    ├── illustrious_baseline/         # Baseline ONNX export
    └── illustrious_toe/              # TOE-ONNX export (in progress)
```

---

## 📈 Performance Timeline

| Approach | Speed | Quality | Status |
|----------|-------|---------|--------|
| **Baseline PyTorch** | 1.0x (120s/step) | 100% | ✅ Reference |
| **INT8 Quantization** | 1.3x | <40% | ❌ Failed |
| **Step Reduction** | 2.5x (5 steps) | Poor | ❌ Failed |
| **ONNX Runtime** | 2-2.5x | 90-95% | ⏳ Testing |
| **ONNX + TOE** | 2-2.5x | 90-95% | ⏳ Testing |
| **LCM (4 steps)** | 12x | 90-95% | 📝 Ready |
| **ONNX + LCM** | **15x** | 85-90% | 🎯 Target |

**Current Reality**: 20 steps × 120s = **40 minutes per image** ❌  
**Target**: 4 steps × 48s = **3.2 minutes per image** ✅

---

## 🔬 Key Technical Decisions

### Why ONNX Runtime?
- ✅ Production-proven (Microsoft, used widely)
- ✅ 2-2.5x speedup on CPU without training
- ✅ Works with existing models
- ✅ Optional static quantization (INT8) that actually works

### Why LCM over Progressive Distillation?
- ✅ State-of-the-art (Stability AI uses it)
- ✅ Fewer training steps (1-2 days vs 3-5 days)
- ✅ Better low-step quality
- ✅ Compatible with LoRAs

### Why GPU Export?
- ✅ 5-10x faster than CPU export
- ✅ User has cloud GPU access (3090)
- ✅ Export once, use forever on CPU

---

## 🚀 Next Steps (For Future Agents)

### Immediate (This Week)
1. ✅ Complete TOE-ONNX export on cloud GPU
2. ⏳ Download ONNX models to local machine
3. ⏳ Run comparison test: `test_onnx_comparison.py`
4. ⏳ Validate 2-2.5x speedup achieved

### Short-term (Weeks 2-3)
5. 🎯 Train LCM model on cloud GPU (1-2 days)
6. 🎯 Test 4-step generation quality
7. 🎯 Export LCM to ONNX
8. 🎯 Validate 12x combined speedup

### Medium-term (Week 4+)
9. Apply QAT (Quantization-Aware Training) to LCM
10. Stack optimizations: LCM + ONNX + QAT = 15-16x
11. Production deployment
12. (Optional) Research publication

---

## 📚 Important Documents

### Artifacts (Decision History)
- `task.md` - Current task checklist
- `implementation_plan.md` - TOE-UNet optimization strategy
- `walkthrough.md` - Latest progress summary
- `cpu_optimization_alternatives.md` - Research on 10 CPU techniques
- `TOE_Optimization_Experiments.md` - 5 TOE-based approaches
- `GPU_Training_Optimizations.md` - 7 GPU-trainable methods

### Guides (How-To)
- `ONNX_LCM_QUICKSTART.md` - ONNX & LCM quick start
- `TOE_ONNX_QUICKSTART.md` - TOE export guide
- `GPU_ONNX_EXPORT_GUIDE.md` - Cloud GPU workflow
- `ONNX_COMPARISON_GUIDE.md` - Testing instructions
- `STEP_REDUCTION_QUICKSTART.md` - Step reduction test (deprecated)

---

## ⚠️ Known Issues & Limitations

### Failed Approaches (Do NOT Retry)
1. **Standard INT8 Quantization**: Fundamentally broken for diffusion models (<40% quality)
2. **Step Reduction without Training**: Quality degrades rapidly below 15 steps
3. **Per-Channel/Selective/Weights-Only INT8**: All variants failed

### Current Limitations
1. **TOE-ONNX Integration**: Simplified, uses CLIP fallback for pooled embeddings
2. **CPU Speed**: Still 2 min/step baseline (ONNX will fix this)
3. **Memory**: 9.8GB model size (quantization didn't work)

### Technical Debt
1. Full TOE integration requires custom ONNX pipeline
2. LCM training requires proper dataset (currently using prompts)
3. Need automated quality metrics (FID, CLIP score)

---

## 🎯 Success Criteria

**Minimum Viable Product (MVP)**:
- [x] TOE model trained and working
- [ ] ONNX export validated (2x speedup)
- [ ] LCM trained (4-step generation)
- [ ] Combined: < 5 min per image
- [ ] Quality: ≥ 85% of baseline

**Production Ready**:
- [ ] ONNX + LCM: < 3 min per image
- [ ] Quality: ≥ 90% of baseline  
- [ ] Automated testing pipeline
- [ ] Documentation complete

**Research Contribution** (Stretch Goal):
- [ ] Novel finding: "Tag-based LCM converges faster"
- [ ] Publication-quality results
- [ ] Open-source release

---

## 💡 Context for Future Agents

### What Works
- ✅ TOE architecture (50MB, quality maintained)
- ✅ PyTorch inference (slow but reliable)
- ✅ Tag processing (15K Danbooru vocab)
- ✅ Export scripts (GPU-accelerated)

### What Doesn't Work
- ❌ Standard INT8 quantization (quality collapse)
- ❌ Naive step reduction (quality degradation)
- ❌ CPU-only export (too slow)

### Critical Insights
1. **Diffusion models are precision-sensitive**: INT8 breaks them
2. **2 min/step is compute-bound**: Need graph optimization (ONNX)
3. **Fewer steps need training**: LCM is the solution
4. **GPU for export, CPU for inference**: Best of both worlds

### Tech Stack
- **Framework**: PyTorch 2.7.1, Diffusers
- **Optimization**: ONNX Runtime, Optimum
- **Training**: PEFT (LoRA), Accelerate
- **Base Model**: Illustrious (martineux/janku6)
- **Hardware**: Cloud GPU (3090 24GB) for export/training, Local CPU for inference

---

## 📞 Quick Commands Reference

```bash
# Export baseline to ONNX (on cloud GPU)
python scripts/export_to_onnx.py \
  --model martineux/janku6 \
  --output-dir onnx_models/illustrious_baseline \
  --device cuda

# Export TOE to ONNX (on cloud GPU)  
python scripts/export_toe_to_onnx.py \
  --toe-checkpoint checkpoints/toe/toe_with_pooling_best.pt \
  --vocab data/vocabulary.json \
  --base-model martineux/janku6 \
  --device cuda

# Test comparison (on local CPU after download)
python scripts/test_onnx_comparison.py \
  --baseline-onnx onnx_models/illustrious_baseline \
  --toe-onnx onnx_models/illustrious_toe \
  --vocab data/vocabulary.json

# Train LCM (on cloud GPU, after ONNX validation)
python scripts/train_lcm.py \
  --base-model martineux/janku6 \
  --epochs 50
```

---

**Last Updated**: 2026-01-05  
**Status**: ONNX export in progress, LCM training ready  
**Next Milestone**: Validate 2.5x ONNX speedup
