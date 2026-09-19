# GUARD: Generalizable Universal AI-generated & Real-image Detector

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GUARD is an end-to-end multi-modal deepfake and AI-generated image detection framework. By fusing spatial pixel-level forensics with frequency-domain spectral analysis through adaptive gated attention, GUARD delivers high generalization accuracy across GANs, Diffusion models, and varied social media degradations.

---

## 🏛️ Architecture Overview

```
                      +-----------------------------+
                      |         Input Image         |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Face Alignment & Extraction |
                      | (Margin Padding & Tilt Fix) |
                      +-----------------------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
   +---------------------------+           +---------------------------+
   |      SPATIAL STREAM       |           |     FREQUENCY STREAM      |
   |     EfficientNet-B4       |           |   2D FFT + Residual CNN   |
   |    (RGB Pixel Seams)      |           | (Spectral Grid Anomalies) |
   +---------------------------+           +---------------------------+
                 | (1792-d)                              | (512-d)
                 +-------------------+-------------------+
                                     |
                                     v
                      +-----------------------------+
                      |    DYNAMIC GATED FUSION     |
                      |  α * Spatial + (1-α) * Freq |
                      +-----------------------------+
                                     | (1024-d)
                                     v
                      +-----------------------------+
                      |     CLASSIFICATION HEAD     |
                      |   (BCE Logit & Sigmoid P)   |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Real vs Fake + Confidence  |
                      +-----------------------------+
```

1. **Input & Face Alignment (`src/preprocessing/face_aligner.py`)**:
   - Automated facial landmark tracking, eye-based tilt alignment, and 25% boundary expansion.
   - Preserves hairline, ears, and jaw blending seams; includes aspect-preserving center-crop fallback for non-facial images.
2. **Spatial Stream (`src/models/spatial_stream.py`)**:
   - Backbone: **EfficientNet-B4** via `timm` with Squeeze-and-Excitation attention.
   - Extracts 1792-d global feature vectors and 2D spatial feature maps.
3. **Frequency Stream (`src/models/frequency_stream.py`)**:
   - Differentiable in-graph 2D Fast Fourier Transform (FFT) with logarithmic magnitude compression.
   - 4-stage Residual CNN (`32 -> 64 -> 128 -> 256`) outputting 512-d spectral representations.
4. **Adaptive Gated Fusion (`src/models/fusion.py`)**:
   - Computes dynamic element-wise gating coefficients $\alpha \in [0, 1]$ to weight spatial vs spectral evidence.
5. **Classification Head (`src/models/classification_head.py`)**:
   - Multi-stage MLP with `LayerNorm` and `Dropout` producing calibrated probabilities.
6. **Training-Time Augmentations (`src/data/augmentations.py`)**:
   - Albumentations pipeline simulating JPEG recompression, optical blur, sensor noise, downsampling, and affine rotation.
7. **Evaluation Protocol (`src/engine/evaluator.py`)**:
   - In-distribution validation (FF++), cross-dataset generalization (Celeb-DF, DFDC), and degradation sweeps.
8. **Futuristic Tech UI Dashboard (`app.py`)**:
   - Glassmorphism, dark mode, and neon accents inspired by Uiverse.io.
   - Live forensic analysis, 2D FFT magnitude spectrum, spatial activation overlays, and real-time degradation testing.

---

## 📁 Repository Structure

```
guard-pbl/
├── configs/
│   ├── __init__.py               # Configuration exports
│   └── config.py                 # Central configuration dataclasses
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── face_aligner.py       # Face detection, eye alignment, margin padding
│   ├── models/
│   │   ├── __init__.py
│   │   ├── spatial_stream.py     # EfficientNet-B4 spatial backbone
│   │   ├── frequency_stream.py   # 2D FFT + Residual CNN frequency backbone
│   │   ├── fusion.py             # Dynamic Gated Attention Fusion block
│   │   ├── classification_head.py# MLP Real/Fake classification head
│   │   └── guard_model.py        # Master end-to-end GUARD detector
│   ├── data/
│   │   ├── __init__.py
│   │   ├── augmentations.py      # Albumentations training & sweep pipelines
│   │   └── dataset.py            # PyTorch Dataset & DataLoader abstractions
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Optimization, AMP, CosineAnnealing, Checkpoints
│   │   └── evaluator.py          # Benchmark evaluation suite
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── styles.py             # Uiverse.io futuristic glassmorphism CSS
│   │   └── visualizations.py     # 2D FFT & spatial activation overlays
│   └── utils/
│       ├── __init__.py
│       ├── seed.py               # Deterministic seed synchronization
│       ├── device.py             # Compute device detection & telemetry
│       ├── logger.py             # Formatted file & console logger
│       └── metrics.py            # AUC-ROC, EER, F1, Accuracy metrics
├── data/                         # Dataset directory (raw, processed, benchmarks)
├── checkpoints/                  # Trained model weights storage
├── outputs/                      # Saved diagnostic plots & JSON reports
├── train.py                      # Training CLI entry point
├── evaluate.py                   # Evaluation CLI entry point
├── app.py                        # Futuristic Streamlit Tech Dashboard
└── requirements.txt              # Project dependencies
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup
```bash
# Miniforge / Conda environment
conda activate guard-env
pip install -r requirements.txt
```

### 2. Launch the Futuristic Tech Dashboard
```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501` to perform live drag-and-drop forensic scans, view 2D Fourier spectra, inspect spatial activation heatmaps, and test transmission degradation robustness.*

### 3. Model Training
```bash
python train.py --data-dir data/processed --epochs 50 --batch-size 16 --lr 1e-4
```

### 4. Benchmark Evaluation
```bash
# In-distribution testing
python evaluate.py --mode in_distribution --data-dir data/processed/test

# Cross-dataset generalization testing
python evaluate.py --mode cross_dataset --benchmarks-dir data/benchmarks

# Robustness degradation sweep (JPEG 90-30, blur, scaling)
python evaluate.py --mode degradation_sweep --data-dir data/processed/test
```
## Model Weights
[Download guard_best.pth (116MB)](https://drive.google.com/file/d/1YhTbu3HncIBEllfH-vvzKN_kJEyu2hJV/view?usp=drive_link) and place it inside the `checkpoints/` folder.