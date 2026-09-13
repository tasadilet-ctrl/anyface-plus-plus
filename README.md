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


## Age coding, and two bugs that stopped the age branch running at all

The age branch had never executed. Two separate faults sat in front of it:

| fault | symptom |
|---|---|
| `age_estimator.py` called `cv2.cvtColor` but never imported `cv2` | `NameError` on the first face, before the network was reached |
| `_AgeNet` used one `channels` argument as both the input and output width of a residual stage | `RuntimeError`: stage 2 expected 128 channels and received 64 |

Either one is fatal, so `AnyfacePP.run()` could not return for any image
containing a face, and `train_age.py` died on its first batch. Both are now
fixed and covered by tests that run a real forward pass.

### The code that turns network outputs into a number

The head predicts independent binary units. How those are decoded decides
what one wrong unit costs.

The original decoding read the units as the **eight bits of the integer age**,
weighted 128...1. That is compact, but the place values are doing something
no classifier should be asked to underwrite:

```
unit target flips across ages 0..100
  binary   [0, 1, 3, 6, 12, 25, 50, 100]      <- the last bit is age parity
  ordinal  all 1                              <- every unit is one step
```

The weight-1 bit asks the network whether the person's age is odd. The
weight-128 bit never turns on below age 128, so one of the eight units is dead
over any realistic range. And because each unit carries its place value, a
single wrong unit can move the answer by **128 years**.

The default is now an **ordinal (thermometer) code**: unit *k* answers "older
than *k*?", and the age is the number of units that say yes. Every unit's
target is a single monotone step, and one wrong unit costs **one year**.

`benchmarks/age_encoding.py` flips each unit independently with probability
*p* and decodes. CI reruns it on every push:

| p | ordinal MAE | binary MAE | ordinal p99 | binary p99 | ordinal >10 y | binary >10 y |
|---|---|---|---|---|---|---|
| 0.001 | 0.10 | 0.23 | 1 | 0 | 0.0% | 0.4% |
| 0.01 | 0.77 | 2.57 | 3 | **128** | 0.0% | 4.0% |
| 0.05 | 2.98 | 12.12 | 9 | 128 | 0.2% | 18.5% |
| 0.10 | 5.48 | 23.68 | 15 | 160 | 9.8% | 34.7% |

At *p* = 0.01 — units that are individually 99% accurate — one binary
prediction in 25 is more than ten years out, and the 99th percentile error is
a century. The ordinal code cannot produce that error from a single wrong
unit.

Note that this comparison is set up in binary's favour: holding *p* equal
gives the binary code 8 units to the ordinal code's 100, so it suffers fewer
wrong units in absolute terms. It loses anyway.

### The same result under actual training

`benchmarks/age_encoding_train.py` trains both codes on one identical
synthetic task — same backbone, same schedule, same images, same seed, only
the decoding differs. Age is readable straight off image brightness, so
nothing here is limited by vision:

| encoding | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| ordinal | 1.19 | 3.06 | 3.19 | **2.48 y** |
| binary | 18.56 | 10.62 | 21.62 | **16.93 y** |

Ordinal wins on every seed with no overlap between the two sets of runs. The
binary runs are also far less repeatable: they span 10.6-21.6 y across the
three seeds, a range wider than the ordinal runs' entire mean, because one
bit changing its mind moves the prediction by decades.

This says nothing about accuracy on real faces; the task is a stand-in. It
says the decoding problem is real in training and not only on paper.

The binary code is still selectable (`--encoding binary`) so the comparison
can be rerun. A checkpoint records the code it was trained with, and
`AgeEstimator` decodes it that way regardless of what the caller asks for —
decoding a binary model as ordinal would return quiet nonsense.


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
│  │ ordinal  │  │ 7-class   │  │ B boxes  │               │
│  │ code CNN │  │ MobileNet │  │ labels   │               │
│  │ (0-100)  │  │ V2        │  │ badges   │               │
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
| **Age Estimator** | Pre-activation residual CNN | Ordinal (thermometer) code → age | Numeric age (0–100) |
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

Holds out `--val-frac` of the data (10% by default), reports validation MAE in
years each epoch, and keeps the best checkpoint rather than the last. Pass
`--encoding binary` to train the old code instead; the choice is stored in the
checkpoint.

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
├── benchmarks/
│   ├── age_encoding.py     # Cost of a wrong unit under each age code (CI)
│   └── age_encoding_train.py  # Both codes trained on one identical task
├── tests/                  # Offline: no weights, no dataset, no GPU
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
    │   ├── age_estimator.py # Ordinal-code age CNN
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
- **Ordinal Regression with Multiple Output CNN for Age Estimation** — Niu, Zhou, Wang, Gao & Hua, CVPR 2016.
  <https://openaccess.thecvf.com/content_cvpr_2016/html/Niu_Ordinal_Regression_With_CVPR_2016_paper.html>
  Source of the ordinal age coding used here, and of AFAD — the dataset
  `datasets/prep_afad.py` already prepares. A previous version of this file
  cited "Deep Residual Learning for Human Age Approximation, Zhang & Sun,
  ACCV 2017" at arXiv:1710.05181; that identifier belongs to a paper on
  neutrino spin oscillations, and the citation has been removed.
- **MobileNetV2** — Sandler et al., CVPR 2018. <https://arxiv.org/abs/1801.04381>
- **FER2013** — Goodfellow et al., 2013. <https://arxiv.org/abs/1308.0852>

## License

MIT
