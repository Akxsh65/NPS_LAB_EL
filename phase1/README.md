# Deep Learning-Based Encrypted Network Traffic Classification and Obfuscation

A full research pipeline for classifying encrypted QUIC network traffic using deep learning,
and defending against classification using deterministic obfuscation techniques.
Targets publication in IEEE/ACM Transactions on Networking or IEEE TIFS.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Methodology](#research-methodology)
- [Project Structure](#project-structure)
- [Environment and Requirements](#environment-and-requirements)
- [Phase 1 — Data Engineering](#phase-1--data-engineering)
- [Phase 2 — Model Training (A100)](#phase-2--model-training-a100)
- [Phase 3 — Obfuscation Defense](#phase-3--obfuscation-defense)
- [Phase 4 — Evaluation and Pareto Optimization](#phase-4--evaluation-and-pareto-optimization)
- [Phase 5 — Manuscript Preparation](#phase-5--manuscript-preparation)
- [Artifacts Reference](#artifacts-reference)
- [Configuration Reference](#configuration-reference)
- [Known Issues and Fixes](#known-issues-and-fixes)
- [Dataset Citation](#dataset-citation)

---

## Project Overview

Modern internet protocols like QUIC (used by YouTube, Google, HTTP/3) encrypt not just
the payload but also transport metadata. This makes traditional Deep Packet Inspection (DPI)
largely ineffective. This project builds a complete research pipeline that:

1. **Trains deep learning classifiers** (1D-CNN + BiLSTM and Transformer Encoder) to
   identify applications from encrypted QUIC traffic using only packet-level metadata
   (sizes, timing, directions) — no payload inspection.

2. **Builds deterministic obfuscators** that simulate anonymizing proxies by injecting
   timing jitter and packet size padding, deliberately degrading the classifier's accuracy.

3. **Evaluates the privacy vs. overhead tradeoff** using Pareto optimization curves,
   quantifying exactly how much bandwidth and latency cost is required to achieve a
   given level of privacy.

The result is a complete attack-defense framework suitable for a Q1 academic publication.

---

## Research Methodology

### The Attack (Phase 2)
Train models that can identify which application (YouTube, Facebook, Zoom, etc.) generated
a network flow, using only the first 30 packets' sizes, inter-packet times, and directions.
This represents what an ISP or eavesdropper could learn about users without breaking encryption.

### The Defense (Phase 3)
Build obfuscation algorithms that manipulate the same metadata features:
- **Size padding**: Round packet sizes up to fixed block sizes (128 bytes, MTU)
- **Timing jitter**: Inject random delays sampled from a Laplace distribution

### The Evaluation (Phase 4)
Measure the fundamental tradeoff:
- Privacy = how much the classifier's accuracy drops under obfuscation
- Cost = how much extra bandwidth and latency the obfuscation introduces
- Pareto curves show the optimal operating points between these two objectives

---

## Project Structure

```
phase1/
│
├── run_phase1.py               # Master script — run this to execute Phase 1
├── config.py                   # All configuration constants (edit here only)
├── dataset_loader.py           # Data ingestion via cesnet-datazoo
├── feature_engineering.py      # Normalization transforms + scaler fitting
├── dataset.py                  # PyTorch Dataset and DataLoader classes
├── validate_pipeline.py        # 7-check go/no-go validation suite
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── data/                       # Auto-created by cesnet-datazoo (gitignored)
│   └── cesnet_quic22/          # Downloaded dataset cache (~2.5 GB for XS)
│
├── artifacts/                  # Generated outputs (gitignored)
│   ├── train_tensors.pt        # Training data: shape (196948, 3, 30)
│   ├── val_tensors.pt          # Validation data: shape (19696, 3, 30)
│   ├── test_tensors.pt         # Test data (FROZEN): shape (49305, 3, 30)
│   ├── ipt_scaler.pkl          # IPT normalization stats (mean, std)
│   └── label_encoder.pkl       # App name ↔ integer index mapping
│
└── logs/                       # Diagnostic logs (gitignored)
```

### Files Added in Later Phases

```
phase2/                         # Created on A100 system
├── models.py                   # CNN-BiLSTM and Transformer architectures
├── train.py                    # AMP-optimized training loop
├── hyperparameter_sweep.py     # W&B or manual sweep script
└── artifacts/
    ├── cnn_bilstm_best.pt      # Best CNN-BiLSTM weights
    └── transformer_best.pt     # Best Transformer weights

phase3/                         # Obfuscation defense
├── obfuscator.py               # pad_sizes() and add_jitter() algorithms
└── artifacts/
    ├── obfuscated_linear128.pt
    ├── obfuscated_mtu.pt
    ├── obfuscated_jitter_low.pt
    └── obfuscated_jitter_high.pt

phase4/                         # Evaluation
├── evaluate.py                 # Accuracy + overhead measurement
├── pareto.py                   # Pareto curve generation
└── results/
    ├── accuracy_results.csv
    ├── confusion_matrices.png
    └── pareto_curves.png

phase5/                         # Manuscript
└── manuscript/
    └── paper.tex
```

---

## Environment and Requirements

### Local Machine (Windows CPU) — Phase 1 and Phase 3
- Python 3.9 or higher
- Windows 10/11
- Minimum 8 GB RAM (16 GB recommended)
- No GPU required

### Remote System (A100) — Phase 2 and Phase 4
- Python 3.9 or higher
- Linux (Ubuntu 20.04+ recommended)
- NVIDIA A100 GPU (40 GB or 80 GB VRAM)
- CUDA 11.8 or higher
- Minimum 32 GB system RAM

### Installation (Local Machine)

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows

# Install all dependencies
pip install -r requirements.txt
```

### requirements.txt Contents

```
cesnet-datazoo==0.2.0
torch>=2.2.0
torchvision
torchaudio
numpy>=1.26.0
pandas>=2.0.0
scikit-learn>=1.4.0
tqdm>=4.66.0
joblib>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

### A100 Additional Requirements

```bash
# Install CUDA-enabled PyTorch on the A100
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional packages for Phase 2
pip install wandb tensorboard
```

---

## Phase 1 — Data Engineering

**Duration**: Days 1–7  
**Hardware**: Local CPU (Windows)  
**Status**: ✅ Complete

### What It Does

1. Downloads the CESNET-QUIC22-XS dataset (~2.5 GB) from Zenodo
2. Splits into train (W-2022-44) and test (W-2022-45) periods
3. Applies three normalization transforms to PPI sequences:
   - **IPT** (Inter-Packet Time): log1p → Z-score standardization
   - **DIR** (Direction): ±1 encoding, 0 reserved for padding
   - **SIZE** (Packet Size): Min-Max normalization to [0, 1]
4. Builds PyTorch tensors of shape (N, 3, 30)
5. Saves everything to ./artifacts/
6. Runs 7 automated validation checks

### How to Run

```bash
cd phase1
python run_phase1.py
```

### Expected Output

```
Train samples :      200,000
Val   samples :       20,000
Test  samples :       50,000
Unique apps   :          101  (64 after rare-class filtering)

ALL 7 CHECKS PASSED
```

### Dataset Configuration

| Parameter | Value | Reason |
|---|---|---|
| Dataset | CESNET-QUIC22 | QUIC protocol, most modern challenge |
| Size tier | XS | ~2.5 GB, safe for Windows CPU RAM |
| Train period | W-2022-44 | Oct 31 – Nov 6, 2022 |
| Test period | W-2022-45 | Nov 7–13, 2022 (held out) |
| Train size | 200,000 flows | RAM-safe cap |
| Val size | 20,000 flows | 10% of train |
| Test size | 50,000 flows | Frozen until Phase 4 |
| Min class samples | 200 | Drops 37 rare classes |
| Final classes | 64 | After rare-class filtering |

### Tensor Structure

Each flow is represented as a (3, 30) float32 tensor:

```
tensor[0]  →  IPT channel    (log1p Z-scored inter-packet times)
tensor[1]  →  DIR channel    (+1 = client→server, -1 = server→client, 0 = padding)
tensor[2]  →  SIZE channel   (packet sizes normalized to [0, 1] by dividing by 1500)
```

Flows shorter than 30 packets are right-padded with zeros. The zero value in the DIR
channel is strictly reserved for padding — the model uses this to distinguish real packets
from absent ones.

### What to Transfer to A100

Copy the following to the A100 system before starting Phase 2:

```
artifacts/
├── train_tensors.pt
├── val_tensors.pt
├── test_tensors.pt
├── ipt_scaler.pkl
└── label_encoder.pkl

dataset.py              (SavedDataset class)
config.py               (shared constants)
feature_engineering.py  (needed by Phase 3 obfuscator)
```

---

## Phase 2 — Model Training (A100)

**Duration**: Days 8–30  
**Hardware**: NVIDIA A100 GPU  
**Status**: 🔲 Not started

### Architectures

**Model 1: 1D-CNN + BiLSTM**
- Two convolutional blocks extract local spatial patterns (burst fingerprints)
- Bidirectional LSTM captures long-range temporal dependencies
- Input: (B, 3, 30) → Output: (B, 64) class probabilities

**Model 2: 1D-Transformer Encoder**
- Multi-head self-attention (8 heads, d_model=128) captures global packet relationships
- Sinusoidal positional encodings preserve sequence order
- Global average pooling → linear classification head

### Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 1024 (A100 can handle 2048) |
| Optimizer | AdamW |
| Scheduler | Cosine Annealing |
| Precision | AMP float16 (bfloat16 if supported) |
| Early stopping | 10 epochs patience |
| Loss function | CrossEntropyLoss with class weights |
| num_workers | 4 |
| pin_memory | True |

### Loading Data on A100

```python
from dataset import SavedDataset
from torch.utils.data import DataLoader

train_ds = SavedDataset('./artifacts/train_tensors.pt')
val_ds   = SavedDataset('./artifacts/val_tensors.pt')

train_loader = DataLoader(train_ds, batch_size=1024,
                          shuffle=True, pin_memory=True, num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=1024,
                          shuffle=False, pin_memory=True, num_workers=4)
```

### Expected Outputs

- `artifacts/cnn_bilstm_best.pt` — frozen model weights
- `artifacts/transformer_best.pt` — frozen model weights
- Baseline accuracy: 80–95% on unobfuscated test set

---

## Phase 3 — Obfuscation Defense

**Duration**: Weeks 5–7  
**Hardware**: Local CPU  
**Status**: 🔲 Not started

### Obfuscation Strategies

**Size Padding**
- Linear padding: rounds each packet size up to nearest multiple of 128 bytes
  `S_padded = ceil(S / 128) × 128`
- MTU padding: pads every packet to 1500 bytes regardless of original size

**Timing Jitter**
- Samples artificial delays from a zero-mean Laplace distribution
- `f(x|μ,b) = (1/2b) exp(-|x-μ|/b)` where μ=0
- Negative delays are clipped to zero (causality constraint)

### Pipeline

The obfuscator:
1. Receives a normalized PPI tensor from the test set
2. Denormalizes using the saved ipt_scaler.pkl values
3. Applies padding and/or jitter transforms
4. Renormalizes using the same scaler values
5. Returns an obfuscated tensor ready for model inference

### Critical Dependency

The obfuscator MUST use the exact same `ipt_scaler.pkl` values from Phase 1.
Never refit the scaler. The mean and std must be identical across all phases.

---

## Phase 4 — Evaluation and Pareto Optimization

**Duration**: Weeks 8–9  
**Hardware**: A100 or local CPU  
**Status**: 🔲 Not started

### Metrics

**Privacy metric**: Absolute accuracy drop under obfuscation
- Baseline: 88–95% accuracy on raw test data
- Target after obfuscation: 15–25% (near random guessing for 64 classes = 1.6%)

**Cost metrics**:
- Bandwidth overhead: `(sum(S_padded) - sum(S_original)) / sum(S_original) × 100%`
- Latency overhead: cumulative sum of injected IPT delays per flow

### Visualizations

- Bar charts: baseline vs defended accuracy per model architecture
- Pareto frontier curves: bandwidth overhead (x) vs model accuracy (y)
- Confusion matrices: how obfuscation causes inter-class misclassification

### Key Research Question

Do Transformer models (global attention) resist obfuscation better than
BiLSTMs (sequential processing)? The hypothesis is yes — Transformers can
correlate distant packets directly, making local timing jitter less effective.

---

## Phase 5 — Manuscript Preparation

**Duration**: Week 10  
**Hardware**: Local machine  
**Status**: 🔲 Not started

### Target Journals

- IEEE/ACM Transactions on Networking (primary target)
- IEEE Transactions on Information Forensics and Security (TIFS)

### Template

Use the official IEEE double-column LaTeX template:
https://www.overleaf.com/gallery/tagged/ieee-official

### Key Contributions to Argue

1. First systematic comparison of CNN-BiLSTM vs Transformer robustness under
   deterministic obfuscation on the CESNET-QUIC22 backbone dataset
2. Pareto-optimal operating points for real-world proxy deployment
3. Evidence that 128-byte linear padding + moderate Laplace jitter is the
   most practical defense for bandwidth-constrained ISP environments

---

## Artifacts Reference

| File | Shape / Type | Created in | Used in | Description |
|---|---|---|---|---|
| train_tensors.pt | (196948, 3, 30) float32 | Phase 1 | Phase 2 | Normalized training flows |
| val_tensors.pt | (19696, 3, 30) float32 | Phase 1 | Phase 2 | Validation flows |
| test_tensors.pt | (49305, 3, 30) float32 | Phase 1 | Phase 4 | Held-out test flows |
| ipt_scaler.pkl | dict {mean, std} | Phase 1 | Phase 3 | IPT normalization stats |
| label_encoder.pkl | sklearn LabelEncoder | Phase 1 | Phase 4+5 | Class index mapping |
| cnn_bilstm_best.pt | model state_dict | Phase 2 | Phase 4 | Best BiLSTM weights |
| transformer_best.pt | model state_dict | Phase 2 | Phase 4 | Best Transformer weights |
| obfuscated_*.pt | (49305, 3, 30) float32 | Phase 3 | Phase 4 | Obfuscated test sets |
| accuracy_results.csv | DataFrame | Phase 4 | Phase 5 | All evaluation metrics |
| pareto_curves.png | Figure | Phase 4 | Phase 5 | Privacy-overhead tradeoff |

---

## Configuration Reference

All settings are in `config.py`. Change values there only.

| Parameter | Default | Description |
|---|---|---|
| DATASET_ROOT | ./data/cesnet_quic22 | Dataset download cache |
| ARTIFACTS_DIR | ./artifacts | Saved tensors and models |
| TRAIN_PERIOD | W-2022-44 | Training week |
| TEST_PERIOD | W-2022-45 | Held-out test week |
| DATASET_SIZE | XS | Size tier (XS/S/M) |
| SEQ_LEN | 30 | Packets per flow |
| MTU | 1500.0 | Max packet size in bytes |
| BATCH_SIZE | 512 | DataLoader batch size |
| NUM_WORKERS | 0 | Must be 0 on Windows |
| PIN_MEMORY | False | Must be False on CPU |
| MIN_CLASS_SAMPLES | 200 | Rare class threshold |
| SEED | 42 | Global random seed |

---

## Known Issues and Fixes

### ModuleNotFoundError: No module named 'files'
Python treats the folder name as a module. Fix: run from inside the project folder.
```bash
cd "D:\New folder\nps\phase1"
python run_phase1.py
```

### cesnet-datazoo version mismatch
Only version 0.2.0 is supported. The API changed significantly between versions.
```bash
pip install cesnet-datazoo==0.2.0
```

### MemoryError during index sorting (S size)
The S dataset size requires ~14 GB RAM. Use XS size instead.
Set `DATASET_SIZE = "XS"` in config.py.

### ValidationError: unexpected keyword argument train_val_split
Correct parameter name in 0.2.0 is `train_val_split_fraction` (not `train_val_split`).
Value is the fraction for validation (0.1 = 10% val), not for training.

### ValidationError: unexpected keyword argument val_size / test_size
Correct parameter names are `val_known_size` and `test_known_size`.

### MemoryError: Unable to allocate 605 MiB
Windows cannot find a contiguous memory block. Fix: set `train_workers=0`,
`val_workers=0`, `test_workers=0` in DatasetConfig, and use size limits.

### ChunkedEncodingError during download
Network dropped mid-download. Just re-run — the download is resumable.
Consider using a download manager for slow/unstable connections.

---

## Dataset Citation

```bibtex
@article{luxemburk2023cesnet,
  title     = {CESNET-QUIC22: A large one-month QUIC network traffic dataset
               from backbone lines},
  author    = {Luxemburk, Jan and Hynek, Karel and {\v{C}}ejka, Tom{\'a}{\v{s}}},
  journal   = {Data in Brief},
  year      = {2023},
  publisher = {Elsevier},
  doi       = {10.1016/j.dib.2023.108927}
}
```

Dataset available at: https://zenodo.org/record/10728760

---

## License

This project is for academic research purposes.
The CESNET-QUIC22 dataset is subject to its own terms of use at Zenodo.
