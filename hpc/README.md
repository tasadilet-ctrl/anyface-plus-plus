# HPC / Cluster Usage Guide

This guide covers training and running Anyface++ on an HPC cluster via SSH
(no GUI, SLURM job scheduler, GPU compute nodes).

## 1. First-Time Setup

```bash
ssh user@cluster

# Clone (or rsync from local)
git clone <repo-url> anyface-plus-plus
cd anyface-plus-plus

# One-command setup: creates conda env, checks CUDA, pre-downloads YOLO26n
bash hpc/setup.sh
```

If your cluster doesn't use conda, create a venv manually:

```bash
module load cuda/12.4 python/3.11
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Adapting `environment.yml`

| Cluster quirk | Fix |
|---------------|-----|
| No NVIDIA channel | Replace `pytorch::pytorch-cuda>=12.1` with `pytorch::pytorch-cuda=11.8` |
| Older CUDA (11.8) | `module load cuda/11.8` + change to `pytorch-cuda=11.8` |
| AMD GPUs (ROCm) | Use `pytorch::pytorch-rocm` instead of `pytorch-cuda` |
| No GPU at all | Remove `pytorch-cuda` line; runs on CPU (slow) |

## 2. SLURM Job Configuration

Edit `#SBATCH` flags in each `scripts/*.slurm` file to match your cluster:

```bash
#SBATCH --partition=gpu          # → your GPU partition name
#SBATCH --gpus=1                 # → number of GPUs
#SBATCH --cpus-per-task=8        # → data-loading workers
#SBATCH --mem=32G                # → depends on batch size
#SBATCH --time=24:00:00          # → training walltime limit
#SBATCH --mail-user=you@edu      # → your email
```

Common partition names: `gpu`, `gpu-a100`, `gpu-v100`, `compute`, `learn`.

Check available partitions: `sinfo -s | grep gpu`

## 3. Training

### Submit via SLURM

```bash
# Train both modules (sequential, one GPU)
sbatch scripts/train_all.slurm

# Or individually
sbatch scripts/train_age.slurm
sbatch scripts/train_mood.slurm
```

### Monitor

```bash
squeue -u $USER          # list your jobs
tail -f hpc/logs/age_*.out  # live training log
```

### Interactive training (GPU node, ad-hoc)

```bash
salloc --partition=gpu --gpus=1 --cpus-per-task=8 --mem=32G --time=8:00:00
conda activate anyface
python train_age.py --data data/ages.csv --img-dir data/utkface \
    --output checkpoints/age.pth --epochs 50 --batch-size 64 --device cuda
```

### Multi-GPU training

For faster training on multiple GPUs, wrap with `torchrun`:

```bash
torchrun --nproc_per_node=4 train_age.py \
    --data data/ages.csv --img-dir data/utkface \
    --output checkpoints/age.pth --epochs 50 --batch-size 256 --device cuda
```

(Note: the training scripts currently use single-GPU DataParallel. For native DDP, add `torch.nn.DataParallel(net)` in the training loop.)

## 4. Inference (Headless)

The headless inference script (`scripts/infer_headless.py`) requires **no
display server** and produces:

- **Annotated images** (`<name>_annotated.jpg`) with bounding boxes
- **Annotated videos** (`<name>_annotated.mp4`)
- **JSONL results file** with face coordinates, age, emotion, confidence

### Batch inference via SLURM

```bash
sbatch scripts/infer_headless.slurm \
    -- --source data/test_images/ --output-dir results/batch

# With custom weights
sbatch scripts/infer_headless.slurm \
    -- --source data/test_images/ \
       --output-dir results/batch \
       --age-weights checkpoints/age.pth \
       --mood-weights checkpoints/mood.pth
```

### Direct usage (interactive node)

```bash
conda activate anyface

# Single image
python scripts/infer_headless.py \
    --source photo.jpg --output-dir results/img --device cuda

# Directory (recursive)
python scripts/infer_headless.py \
    --source data/photos/ --output-dir results/photos \
    --device cuda --recursive

# Video (every 5th frame, max 500 frames)
python scripts/infer_headless.py \
    --source data/clip.mp4 --output-dir results/vid \
    --device cuda --video-every 5 --video-max-frames 500

# Custom extensions
python scripts/infer_headless.py \
    --source data/raw/ --output-dir results/raw \
    --device cuda --extensions png,bmp,tiff
```

### Output format

**JSONL** (`results.jsonl`) — one JSON object per line:

```json
{"file": "group01.jpg", "faces": [
  {"x1": 120, "y1": 80, "x2": 340, "y2": 380, "confidence": 0.94,
   "age": 34.2, "emotion": "happy", "emotion_conf": 0.87,
   "emotions": {"happy": 0.87, "neutral": 0.08, ...}}
]}
```

**Video JSONL** includes frame number:

```json
{"file": "clip.mp4", "frame": 0, "faces": [...]}
{"file": "clip.mp4", "frame": 5, "faces": [...]}
```

## 5. Data Transfer

```bash
# Upload datasets (from local machine)
rsync -avz --progress data/ user@cluster:~/anyface-plus-plus/data/

# Download trained weights
rsync -avz user@cluster:~/anyface-plus-plus/checkpoints/ ./checkpoints/

# Download inference results
rsync -avz user@cluster:~/anyface-plus-plus/results/ ./results/
```

For large datasets, consider using Globus or cluster scratch space first.

## 6. Transferring Trained Models Back to Local

After training on HPC:

```bash
# On local machine:
rsync -avz user@cluster:~/anyface-plus-plus/checkpoints/age.pth checkpoints/
rsync -avz user@cluster:~/anyface-plus-plus/checkpoints/mood.pth checkpoints/

# Now use locally with any device:
python demo.py image photo.jpg \
    --age-weights checkpoints/age.pth \
    --mood-weights checkpoints/mood.pth \
    --device mps
```

## 7. Quick-Reference Commands

```bash
# Setup (run once)
bash hpc/setup.sh

# Training
sbatch scripts/train_all.slurm
sbatch scripts/train_age.slurm
sbatch scripts/train_mood.slurm

# Inference
sbatch scripts/infer_headless.slurm -- --source data/ --output-dir results/

# Monitor
squeue -u $USER
tail -f hpc/logs/age_*.out

# Interactive GPU session
salloc --partition=gpu --gpus=1 --cpus-per-task=8 --mem=32G --time=8:00:00
```
