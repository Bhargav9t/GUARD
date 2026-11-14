# GUARD: Generalizable Universal AI-generated & Real-image Detector

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

GUARD is an end-to-end multi-modal deepfake and AI-generated image detection framework. By fusing spatial pixel-level forensics with frequency-domain spectral analysis through adaptive gated attention, GUARD delivers high generalization accuracy across GANs, Diffusion models, and varied social media degradations.

> 📚 **Technical & Mathematical Documentation**:
> - [**WEIGHTS_AND_BENCHMARK_RESULTS.md**](WEIGHTS_AND_BENCHMARK_RESULTS.md): Detailed registry of all checkpoint weights, parameter counts, download links, and full empirical benchmark evaluation tables.
> - [**MATHEMATICAL_FOUNDATIONS_README.md**](MATHEMATICAL_FOUNDATIONS_README.md): Exhaustive mathematical formulations for 2D FFT, power-law decay, Poisson sensor noise, SE attention, W-MSA complexity, gated gradient flow proofs, and temporal continuity.
> - [**TECHNICAL_DEPTH_AND_CAPABILITIES.md**](TECHNICAL_DEPTH_AND_CAPABILITIES.md): In-depth system architecture, multi-backbone consensus, latency budgets, benchmark metrics, and generator coverage.
> - [**CODE_SNIPPETS_README.md**](CODE_SNIPPETS_README.md): Detailed code snippets for the dual-stream model, 2D FFT/DCT layer, gated fusion, explainability, and video forensics.
> - [**ML_ALGORITHM_SPECIFICATION.md**](ML_ALGORITHM_SPECIFICATION.md): Machine learning algorithm specification and mathematical formulation.

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
8. **Modular Inference & Multi-Evidence Hybrid Dashboard (`app.py`)**:
   - Modern glassmorphism UI with live forensic analysis, 2D FFT magnitude spectrum, multi-layer spatial artifact overlays, and real-time degradation stress testing.
9. **URL Stream Ingestion (`src/video/url_downloader.py`)**:
   - Live stream fetching from YouTube, YouTube Shorts, Instagram Reels, and TikTok via `yt-dlp` in lightweight <=720p format with auto-cleaning temp storage and a 90-second duration ceiling.
10. **Scene-Cut Aware Video Forensics (`src/video/processor.py`)**:
    - OpenCV 1–2 FPS sampling with HSV histogram correlation scene-cut isolation, excluding normal editing jump cuts from AI temporal morphing/flicker scores.

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

## 🚀 Quickstart Guide: Demo-Ready Prototype

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Shalini369/GUARD-PBL.git
cd GUARD-PBL

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Launch the Demo Web App
```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501` to test live image and video AI detection.*

### 3. Hardware Requirements & Out-of-the-Box Operation
- **GPU (Recommended)**: NVIDIA GeForce RTX 3050 (6GB VRAM) or equivalent Ampere/Ada GPU. The app automatically runs in **CUDA FP16** mode (~500MB VRAM footprint), leaving plenty of memory headroom.
- **CPU Fallback**: If no CUDA GPU is detected, the app automatically falls back to **CPU execution (FP32)** with zero configuration required.

### 4. Architecture & Extensibility
- **Abstract `BaseDetector`**: Built around an extensible backend interface (`src/detectors/base.py`).
- **Inference Engines Supported**:
  - `Organika/sdxl-detector` (Default off-the-shelf Hugging Face Swin baseline)
  - `umm-maybe/AI-image-detector` (Alternative Hugging Face Swin baseline)
  - `GUARD Dual-Stream` (Phase 2 Custom: Spatial EfficientNet-B4 + 2D FFT Frequency Pathway)
- **Key Modules**:
  - `src/detectors/`: BaseDetector interface, Hugging Face detector wrapper with label normalization, and dual-stream adapter.
  - `src/explainability/`: PyTorch-GradCAM spatial heatmap overlay with Swin reshape transform & 2D FFT log-magnitude spectrum.
  - `src/stress/`: In-memory perturbation suite (JPEG Q85, Q50, Resizing 0.75x, 0.50x) with majority stability flag.
  - `src/video/`: OpenCV 1–2 FPS frame extraction, temporal flicker computation ($|\Delta s_t| + \text{Laplacian Var}$), and Plotly interactive timeline.
  - `src/ui/`: Honest UI disclaimers (uncalibrated softmax probabilities, perturbation stability vs. adversarial robustness).

---

## 📊 Measured Benchmark Evaluation Results
Evaluated on FaceForensics++, Celeb-DF, DFDC, and Unseen Diffusion Generators:

### Condition 1: Clean Images (Uncorrupted)
| Benchmark Dataset | Accuracy | AUC-ROC | EER | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **FaceForensics++** | 55.00% | 0.7725 | 32.50% | 0.3571 |
| **Celeb-DF** | 50.00% | 0.3150 | 65.00% | 0.0000 |
| **DFDC** | 50.00% | 0.2675 | 75.00% | 0.0909 |
| **In-Distribution Test** | 67.50% | 0.7850 | 25.00% | 0.5185 |
| **Average Clean Aggregate** | **55.62%** | **0.5350** | -- | -- |

### Condition 2: Degraded Images (Real-World Corruptions)
| Degradation Perturbation | Accuracy | AUC-ROC | EER | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **JPEG Q=85 (Mild Transmission)** | 67.50% | 0.7725 | 27.50% | 0.5185 |
| **JPEG Q=60 (Moderate Social Media)** | 67.50% | 0.7550 | 30.00% | 0.5517 |
| **JPEG Q=35 (Heavy Recompression)** | 70.00% | 0.7600 | 30.00% | 0.6250 |
| **Resizing 0.75x (Downscale & Upscale)** | 65.00% | 0.7550 | 37.50% | 0.5000 |
| **Resizing 0.50x (Severe Thumbnailing)** | 62.50% | 0.7550 | 30.00% | 0.4444 |
| **Gaussian Blur (Kernel=5)** | 57.50% | 0.7450 | 32.50% | 0.2609 |
| **Compound (JPEG Q=50 + 0.70x Resize)** | 67.50% | 0.7700 | 22.50% | 0.5185 |
| **Average Degraded Aggregate** | **65.36%** | -- | -- | -- |

### Condition 3: Unseen Generator Architecture & Generalization Gap
| Generator Architecture | Accuracy | AUC-ROC | EER | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Unseen Diffusion Generator** | 42.50% | 0.2750 | 60.00% | 0.0000 |
| **Visible Generalization Gap** | **+13.13%** | -- | -- | -- |

---

## 📓 Google Colab Free GPU Pipeline
Open and execute `GUARD_Colab_Pipeline.ipynb` directly on Google Colab (with a free T4 GPU). It provides:
1. Environment verification & GPU hardware detection.
2. Architecture definition & 2D FFT Frequency Pathway.
3. Training with aggressive training-time augmentations.
4. Full 3-Condition Evaluation benchmark.
5. Explainability generation (Grad-CAM + 2D FFT spectrogram side-by-side).
6. Background Streamlit web app server with public tunneling via `localtunnel`.

---

## Model Weights
[Download guard_best.pth (116MB)](https://drive.google.com/file/d/1YhTbu3HncIBEllfH-vvzKN_kJEyu2hJV/view?usp=drive_link) and place it inside the `checkpoints/` folder.