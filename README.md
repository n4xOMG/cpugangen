# HQPD: Hybrid Quantized Progressive Distillation

CPU-optimized SDXL Illustrious for anime generation.

## Project Structure

```
cpugangen/
├── venv/                  # Virtual environment
├── hqpd/                  # Main package
│   ├── models/           # Model architectures
│   ├── quantization/     # Quantization utilities
│   ├── distillation/     # Distillation methods
│   └── utils/            # Helper functions
├── scripts/              # Training/inference scripts
├── configs/              # Configuration files
├── checkpoints/          # Saved models
└── data/                 # Datasets
```

## Phase 1: Tag-Optimized Encoder (TOE)

Replace SDXL's 999M CLIP encoders with a 50M learned encoder.

### Quick Start

1. **Activate virtual environment**:
```bash
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Test TOE architecture**:
```bash
python -m hqpd.models.toe
```

4. **Train TOE (with dummy data)**:
```bash
python scripts/train_toe.py --config configs/toe_config.yaml
```

### Model Architecture

- **Tag Embeddings**: 15K vocabulary, INT4 quantized
- **Transformer**: 4 layers, 8 heads, 2048 dim
- **Total Parameters**: ~50M (vs 999M CLIP)
- **Output**: SDXL-compatible context embeddings (77 × 2048)

### Next Steps

1. Download/prepare Danbooru dataset subset
2. Pre-compute CLIP embeddings (teacher)
3. Train TOE with knowledge distillation
4. Validate quality vs original CLIP

## License

Research project for CPU-optimized anime generation.
