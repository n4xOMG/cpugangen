# Cloud GPU Deployment Guide (Vast.ai / 3090 24GB)

## Files to Transfer

### Essential Files & Folders

#### 1. Code (Required)
```
cpugangen/
├── hqpd/                    # Main package - TRANSFER ALL
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── toe.py           # Tag-Optimized Encoder
│   └── utils/
│       ├── __init__.py
│       └── danbooru.py      # Tag processor
│
├── scripts/                 # Training scripts - TRANSFER ALL
│   ├── train_toe.py         # Main training script
│   ├── build_vocabulary.py  # Vocabulary builder (optional if vocab exists)
│   └── test_vocabulary.py   # Test script (optional)
│
├── configs/                 # Configuration - TRANSFER ALL
│   └── toe_config.yaml
│
├── data/                    # Data files - TRANSFER SELECTIVE
│   └── vocabulary.json      # ✓ REQUIRED (1.8 MB)
│   └── tags.json            # ✗ NOT NEEDED (already processed)
│
├── requirements.txt         # ✓ REQUIRED
└── README.md                # Optional
```

#### 2. Summary of What to Transfer

**Must Transfer:**
- `hqpd/` (entire folder with all Python files)
- `scripts/train_toe.py`
- `configs/toe_config.yaml`
- `data/vocabulary.json` (1.8 MB)
- `requirements.txt`

**Optional (but recommended):**
- `scripts/test_vocabulary.py` (for validation)
- `README.md`

**Do NOT transfer:**
- `data/tags.json` (42 MB - not needed, already processed)
- `venv/` (create fresh on cloud)
- `checkpoints/` (will be generated during training)
- `__pycache__/` (Python cache)
- `.git/` (if exists)

---

## Transfer Methods

### Option 1: Using SCP/RSYNC (Recommended)

```bash
# Create a clean copy directory first
mkdir cpugangen_transfer
cd cpugangen_transfer

# Copy only essential files
cp -r ../cpugangen/hqpd .
cp -r ../cpugangen/scripts .
cp -r ../cpugangen/configs .
mkdir data
cp ../cpugangen/data/vocabulary.json data/
cp ../cpugangen/requirements.txt .
cp ../cpugangen/README.md .

# Transfer to Vast.ai instance
# Replace <USER> and <HOST> with your Vast.ai SSH details
scp -r cpugangen_transfer/ <USER>@<HOST>:/workspace/cpugangen
```

### Option 2: Using Git (If you have a repo)

```bash
# On cloud GPU
git clone <your-repo-url>
cd cpugangen

# Copy vocabulary.json separately (if not in repo)
# Use scp or wget from cloud storage
```

### Option 3: Using Zip/Archive

```bash
# On local machine - create archive excluding unnecessary files
cd d:\LTPython\scripts\cpugangen
tar -czf cpugangen_deploy.tar.gz \
  --exclude='venv' \
  --exclude='__pycache__' \
  --exclude='checkpoints' \
  --exclude='data/tags.json' \
  --exclude='data/api_captions_*.json' \
  hqpd/ scripts/ configs/ data/vocabulary.json requirements.txt README.md

# Transfer archive to cloud
scp cpugangen_deploy.tar.gz <USER>@<HOST>:/workspace/

# On cloud GPU - extract
cd /workspace
tar -xzf cpugangen_deploy.tar.gz
```

### Option 4: PowerShell Script (Windows)

```powershell
# Create deployment package
$deployDir = "d:\LTPython\scripts\cpugangen_deploy"
New-Item -ItemType Directory -Path $deployDir -Force

# Copy essential files
Copy-Item -Recurse "hqpd" "$deployDir\hqpd"
Copy-Item -Recurse "scripts" "$deployDir\scripts"
Copy-Item -Recurse "configs" "$deployDir\configs"
New-Item -ItemType Directory -Path "$deployDir\data" -Force
Copy-Item "data\vocabulary.json" "$deployDir\data\"
Copy-Item "requirements.txt" "$deployDir\"
Copy-Item "README.md" "$deployDir\"

# Create archive
Compress-Archive -Path "$deployDir\*" -DestinationPath "cpugangen_deploy.zip"
```

---

## Setup on Cloud GPU (Vast.ai 3090)

### 1. Connect to Instance

```bash
# Use SSH details from Vast.ai dashboard
ssh <USER>@<HOST> -p <PORT>
```

### 2. Setup Environment

```bash
# Navigate to workspace
cd /workspace

# Extract if you transferred archive
tar -xzf cpugangen_deploy.tar.gz  # or unzip cpugangen_deploy.zip

cd cpugangen

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### 3. Verify Setup

```bash
# Test vocabulary loading
python scripts/test_vocabulary.py

# Quick architecture test
python -m hqpd.models.toe
```

### 4. Update Config for GPU

Edit `configs/toe_config.yaml`:
```yaml
hardware:
  device: "cuda"  # Make sure this is set to cuda
  enable_tf32: true

training:
  mixed_precision: true  # Enable for faster training
```

### 5. Start Training

```bash
# Start training with screen/tmux to keep running after disconnect
screen -S toe_training
# or
tmux new -s toe_training

# Run training
python scripts/train_toe.py --config configs/toe_config.yaml

# Detach from screen: Ctrl+A, D
# Reattach: screen -r toe_training
```

---

## Estimated Transfer Size

- Code files (hqpd/ + scripts/ + configs/): ~50 KB
- vocabulary.json: ~1.8 MB
- requirements.txt: ~1 KB
- **Total: ~2 MB**

Much smaller than the original 42 MB tags.json because we already processed it!

---

## Quick Commands Summary

```bash
# On cloud GPU after transfer
cd /workspace/cpugangen
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/test_vocabulary.py
python scripts/train_toe.py --config configs/toe_config.yaml
```

---

## Monitoring Training

### View logs in real-time
```bash
# If using screen
screen -r toe_training

# If using tmux
tmux attach -t toe_training
```

### Download checkpoints periodically
```bash
# From local machine
scp -r <USER>@<HOST>:/workspace/cpugangen/checkpoints/ ./checkpoints_backup/
```

---

## Notes

- Total transfer is only ~2 MB (vs 42 MB) thanks to vocabulary preprocessing!
- Training on 3090 24GB should handle batch size 8-16 easily
- Checkpoints will be saved to `checkpoints/toe/` on cloud GPU
- Remember to download checkpoints before instance termination
