# Anyface++

[![CI](https://github.com/tasadilet-ctrl/anyface-plus-plus/actions/workflows/ci.yml/badge.svg)](https://github.com/tasadilet-ctrl/anyface-plus-plus/actions/workflows/ci.yml)

> **YOLO26-based face analysis pipeline** — real-time face detection, age estimation, and mood/emotion classification in a single unified framework.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-≥2.1-ee4c2c.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/yolo-yolo26-success.svg)](https://github.com/ultralytics/ultralytics)

## Face weights are required (and why)

`FaceDetector` needs weights that actually have a **face** class:

```bash
python3 scripts/download_face_model.py   # fetches yolov8n-face.pt (~6 MB)
```

This used to default to the stock COCO-pretrained `yolo26n.pt`, and `detect()`
returned *every* box the model produced without filtering by class. COCO has
no face class, so on Ultralytics' standard `bus.jpg` the detector returned
five "faces":

| what YOLO actually saw | returned as |
|---|---|
| bus | FaceBox 803×520 px, conf 0.88 |
| person | FaceBox 192×505 px, conf 0.87 |
| person | FaceBox 123×456 px, conf 0.86 |
| person | FaceBox 140×485 px, conf 0.85 |
| person | FaceBox 59×319 px, conf 0.65 |

Those crops were then handed to the age estimator and mood classifier — the
pipeline would confidently report the age and emotion of a bus. Nothing
crashed; the output was simply wrong, which is the harder failure to notice.

With proper face weights the same image yields two detections of 37×52 px and
36×53 px — face-sized, as expected.

Now the detector filters by class, and weights with no face class are rejected
with an actionable error rather than silently misused:

```
NotAFaceModelError: These weights have no 'face' class -- they look like a
general object detector (classes include: person, bicycle, car, ...).
Fix: run 'python3 scripts/download_face_model.py' ...
```

If you deliberately want to run against a non-face model, pass
`face_class_ids=[...]` so the decision is explicit in the code.


## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Anyface++ Pipeline                       │
│                                                             │
│  Input (image / video / webcam)                             │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────┐                                          │
│  │  Face Detect  │  YOLO26n (Ultralytics)                   │
│  │  (bounding    │  → FaceBox[x1,y1,x2,y2,conf]             │
│  │   boxes)      │                                          │
│  └──────┬───────┘                                          │
│         │ face crops                                       │
│         ├──────────────┬──────────────┐                    │
│         ▼              ▼              ▼                    │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐               │
│  │   Age    │  │   Mood    │  │  Visual- │               │
│  │ Estimator│  │ Classif.  │  │  izer    │               │
│  │          │  │           │  │          │               │
│  │ 8-bit    │  │ 7-class   │  │ B boxes  │               │
│  │ binary   │  │ MobileNet │  │ labels   │               │
│  │ code CNN │  │ V2        │  │ badges   │               │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘               │
│       │              │              │                     │
│       ▼              ▼              ▼                    │
│  FaceResult[x1,y1,x2,y2, conf, age, emotion, probs]       │
└─────────────────────────────────────────────────────────────┘
```

### Modules

| Module | Backbone | Task | Output |
|--------|----------|------|--------|
| **Face Detector** | YOLO26n (Ultralytics) | Object detection → faces | Bounding boxes + confidence |
| **Age Estimator** | Deep Residual CNN (Zhang & Sun, ACCV 2017) | 8-bit binary age code → age regression | Numeric age (0–127) |
| **Mood Classifier** | MobileNetV2 | 7-class emotion classification | `{angry, disgusted, fearful, happy, sad, surprised, neutral}` with probabilities |

## Quick Start (Local)

```bash
cd anyface-plus-plus
pip install -r requirements.txt
python3 scripts/download_face_model.py   # face weights (required)

# Image
python demo.py image your_photo.jpg --device mps    # Mac
python demo.py image your_photo.jpg --device cuda   # NVIDIA

# Webcam
python demo.py webcam

# Video
python demo.py video your_clip.mp4 -o output.mp4
```

## HPC / Cluster Usage (SSH, SLURM, no GUI)

The recommended workflow for training and inference on HPC clusters.

### 1. Set up environment

```bash
# SSH into cluster, clone repo, run setup:
bash hpc/setup.sh
```

This creates a conda environment (`anyface`), verifies CUDA, and pre-downloads YOLO26n weights.

### 2. Transfer datasets

```bash
# From your local machine:
rsync -avz --progress data/ user@cluster:~/anyface-plus-plus/data/
```

### 3. Train (via SLURM)

```bash
# Edit #SBATCH flags in scripts/*.slurm for your cluster
sbatch scripts/train_all.slurm          # both age + mood
# Or individually:
sbatch scripts/train_age.slurm
sbatch scripts/train_mood.slurm
```

### 4. Infer on HPC (headless, batch)

```bash
sbatch scripts/infer_headless.slurm \
    -- --source data/test_images/ --output-dir results/batch

# Or directly (interactive GPU node):
python scripts/infer_headless.py \
    --source data/test_images/ \
    --output-dir results/batch \
    --age-weights checkpoints/age.pth \
    --mood-weights checkpoints/mood.pth \
    --device cuda
```

Output is **annotated images** + a **JSONL results file** — no GUI needed.

See [hpc/README.md](hpc/README.md) for the full guide.

## Training Your Own Weights

The age and mood modules ship without pretrained weights. Train them:

### Age Estimation

CSV with columns `image_path` and `age`:

```bash
python train_age.py \
    --data data/ages.csv \
    --img-dir data/utkface \
    --output checkpoints/age.pth \
    --epochs 50 --batch-size 64 --device cuda
```

### Mood Classification

CSV with columns `image_path` and `emotion`:

```bash
python train_mood.py \
    --data data/emotions.csv \
    --img-dir data/fer2013 \
    --output checkpoints/mood.pth \
    --epochs 30 --batch-size 64 --device cuda
```

## API Usage

```python
from anyface_pp.pipeline import AnyfacePP

analyzer = AnyfacePP(device="cuda")

# Single image
results, annotated_img = analyzer.run_with_image("group_photo.jpg")
for r in results:
    print(f"age~{r.age:.0f}, mood={r.emotion}({r.emotion_conf:.1%})")

# Video stream (generator)
for frame_results, frame_img in analyzer.run_video("clip.mp4"):
    for r in frame_results:
        print(r.emotion)
```

## Project Structure

```
anyface-plus-plus/
├── README.md
├── requirements.txt
├── demo.py                 # CLI demo (image / video / webcam, needs display)
├── train_age.py            # Age estimator training script
├── train_mood.py           # Mood classifier training script
├── scripts/
│   ├── infer_headless.py   # Headless batch inference (HPC / SSH friendly)
│   ├── train_age.slurm     # SLURM job for age training
│   ├── train_mood.slurm    # SLURM job for mood training
│   ├── train_all.slurm     # SLURM job for both
│   └── infer_headless.slurm   # SLURM job for headless inference
├── hpc/
│   ├── environment.yml     # Conda env with CUDA for HPC
│   ├── setup.sh            # One-command HPC setup
│   └── README.md           # Full HPC usage guide
├── checkpoints/            # Trained model weights
├── output/                 # Demo output
└── anyface_pp/
    ├── __init__.py
    ├── pipeline.py          # Unified AnyfacePP pipeline orchestrator
    ├── models/
    │   ├── face_detector.py # YOLO26 face detection
    │   ├── age_estimator.py # 8-bit binary code age CNN
    │   └── mood_classifier.py  # MobileNetV2 emotion classifier
    └── utils/
        └── visualizer.py    # Bounding box + label drawing
```

## Datasets

| Task | Recommended datasets |
|------|---------------------|
| **Face Detection** | [WIDER Face](http://shuoyang1213.me/WIDERFACE/), [COCO-Face](https://github.com/facebookresearch/DenseFace/blob/master.datasets/COCO-Face.org.md) |
| **Age Estimation** | [UTKFace](https://susanqq.github.io/UTKFace/), [IMDB-WIKI](http://downloads.cs.stanford.edu/nlp/data/imdbwiki/imdb_wiki.tsv.gz), [FG-NET](http://www.expertsystemsgr.com/databases/datab1.php) |
| **Mood / Emotion** | [FER2013](https://github.com/grassfedcoder/Kaggle-FER2013), [AffectNet](https://sahibanyasar.github.io/affectnet/), [RAF-DB](http://www.whdeng.cn/RAF/) |

## References

- **YOLO26** — Ultralytics. <https://github.com/ultralytics/ultralytics>
- **Deep Residual Learning for Human Age Approximation** — Zhang & Sun, ACCV 2017. <https://arxiv.org/abs/1710.05181>
- **MobileNetV2** — Sandler et al., CVPR 2018. <https://arxiv.org/abs/1801.04381>
- **FER2013** — Goodfellow et al., 2013. <https://arxiv.org/abs/1308.0852>

## License

MIT
