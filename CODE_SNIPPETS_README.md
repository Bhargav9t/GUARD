# 🔬 GUARD: Core Architecture & Important Code Snippets

This document compiles the most critical code snippets across the **GUARD** (*Generalizable Universal AI-generated & Real-image Detector*) framework. Each snippet highlights a foundational mathematical, architectural, or forensic algorithmic component.

---

## 📑 Table of Contents
1. [Dual-Stream Master Model & Forward Pass](#1-dual-stream-master-model--forward-pass)
2. [Differentiable 2D FFT & 2D DCT Frequency Stream](#2-differentiable-2d-fft--2d-dct-frequency-stream)
3. [Dynamic Gated Residual Attention Fusion](#3-dynamic-gated-residual-attention-fusion)
4. [Facial Seam Margin Expansion & Tilt Alignment](#4-facial-seam-margin-expansion--tilt-alignment)
5. [Visual Explainability: 2D FFT Spectrogram & Heatmap Overlays](#5-visual-explainability-2d-fft-spectrogram--heatmap-overlays)
6. [In-Memory Degradation Stress-Testing Pipeline](#6-in-memory-degradation-stress-testing-pipeline)
7. [Scene-Cut Aware Video Forensics & Temporal Flicker Analysis](#7-scene-cut-aware-video-forensics--temporal-flicker-analysis)
8. [Unified Production Inference Engine](#8-unified-production-inference-engine)

---

## 1. Dual-Stream Master Model & Forward Pass
**File:** [`src/models/guard_model.py`](file:///c:/Users/Bhargav/Documents/projects/GUARD-PBL/src/models/guard_model.py)

The `GUARDModel` orchestrates the dual-branch framework: extracting 1792-d spatial features via EfficientNet-B4 and 512-d spectral energy features via the 2D FFT/DCT stream, fusing them dynamically, and producing calibrated real vs. fake probabilities.

```python
class GUARDModel(nn.Module):
    """Dual-branch spatial-frequency architecture with adaptive gated fusion."""

    def __init__(self, config: Optional[GuardConfig] = None):
        super().__init__()
        self.config = config or default_config

        # 1. RGB Spatial Stream (EfficientNet-B4 backbone via timm)
        self.spatial_stream = SpatialStream(self.config.spatial)

        # 2. Frequency Stream (2D FFT/DCT + Residual CNN)
        self.frequency_stream = FrequencyStream(self.config.frequency)

        # 3. Adaptive Gated Fusion Layer
        self.fusion = FusionLayer(
            config=self.config.fusion,
            spatial_dim=self.spatial_stream.out_features,       # 1792
            frequency_dim=self.frequency_stream.out_features,   # 512
        )

        # 4. Calibrated Binary Classification Head
        self.head = ClassificationHead(
            in_features=self.fusion.out_features,              # 1024
            config=self.config.head,
        )

    def forward(self, x: torch.Tensor, apply_norm: bool = True) -> Dict[str, torch.Tensor]:
        """
        Executes parallel dual-stream feature extraction and dynamic fusion.
        x: (B, 3, H, W) normalized RGB tensor in [0.0, 1.0]
        """
        # Step 1: Spatial pixel seam representations
        f_spatial = self.spatial_stream(x, apply_norm=apply_norm)     # (B, 1792)

        # Step 2: In-graph 2D FFT/DCT spectral anomalies
        f_frequency = self.frequency_stream(x)                        # (B, 512)

        # Step 3: Adaptive Gated Fusion with gradient preservation
        f_fused, gate_weights = self.fusion(f_spatial, f_frequency)   # (B, 1024)

        # Step 4: Classification head projection & Sigmoid mapping
        logits = self.head(f_fused)                                    # (B, 1)
        probabilities = torch.sigmoid(logits)                          # (B, 1) in [0, 1]

        return {
            "logits": logits,
            "probabilities": probabilities,
            "spatial_features": f_spatial,
            "frequency_features": f_frequency,
            "fused_features": f_fused,
            "gate_weights": gate_weights,
        }
```

---

## 2. Differentiable 2D FFT & 2D DCT Frequency Stream
**File:** [`src/models/frequency_stream.py`](file:///c:/Users/Bhargav/Documents/projects/GUARD-PBL/src/models/frequency_stream.py)

Generative models (GANs, Diffusion models) leave distinct high-frequency spectral roll-offs and upsampling grid anomalies. This module implements a differentiable 2D DCT-II using PyTorch FFT, computes centralized log-magnitude spectra, and processes them with residual convolutional blocks.

```python
def compute_2d_dct(x: torch.Tensor) -> torch.Tensor:
    """Computes differentiable 2D DCT-II using mirrored 1D FFT operations."""
    n_h, n_w = x.shape[-2], x.shape[-1]

    # Horizontal 1D DCT via mirrored FFT
    x_mirrored_w = torch.cat([x, x.flip(-1)], dim=-1)
    fft_w = torch.fft.fft(x_mirrored_w, dim=-1)[..., :n_w]
    phase_w = torch.exp(-1j * math.pi * torch.arange(n_w, device=x.device) / (2.0 * n_w))
    dct_w = (fft_w * phase_w).real

    # Vertical 1D DCT via mirrored FFT
    x_mirrored_h = torch.cat([dct_w, dct_w.flip(-2)], dim=-2)
    fft_h = torch.fft.fft(x_mirrored_h, dim=-2)[..., :n_h, :]
    phase_h = torch.exp(-1j * math.pi * torch.arange(n_h, device=x.device).view(-1, 1) / (2.0 * n_h))
    return (fft_h * phase_h).real


class FourierTransform2D(nn.Module):
    """In-graph 2D FFT & 2D DCT layer with logarithmic magnitude compression."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 2D FFT over spatial dimensions (H, W) with orthonormal scaling
        fft_complex = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")

        # 2. Shift DC (zero-frequency) component to center
        fft_shifted = torch.fft.fftshift(fft_complex, dim=(-2, -1))

        # 3. Logarithmic dynamic range compression: S = log(1 + |F|)
        log_fft = torch.log1p(torch.abs(fft_shifted))

        # 4. 2D Discrete Cosine Transform for cardinal grid artifact detection
        log_dct = torch.log1p(torch.abs(compute_2d_dct(x)))

        # 5. Per-sample min-max normalization
        b, c, h, w = log_fft.shape
        flat_fft = log_fft.reshape(b, c, -1)
        min_fft = flat_fft.min(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        max_fft = flat_fft.max(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        norm_fft = (log_fft - min_fft) / (max_fft - min_fft + self.eps)

        flat_dct = log_dct.reshape(b, c, -1)
        min_dct = flat_dct.min(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        max_dct = flat_dct.max(dim=-1, keepdim=True)[0].view(b, c, 1, 1)
        norm_dct = (log_dct - min_dct) / (max_dct - min_dct + self.eps)

        # Hybrid blend: 60% radial FFT energy + 40% cardinal DCT grid artifacts
        return 0.60 * norm_fft + 0.40 * norm_dct
```

---

## 3. Dynamic Gated Residual Attention Fusion
**File:** [`src/models/fusion.py`](file:///c:/Users/Bhargav/Documents/projects/GUARD-PBL/src/models/fusion.py)

Standard attention fusion often suffers from **modality dominance** (one stream starves the other of gradient updates). GUARD solves this through **residual concatenation**:

$$\mathbf{h}_s = \text{LayerNorm}(\text{GELU}(W_s f_s)), \quad \mathbf{h}_f = \text{LayerNorm}(\text{GELU}(W_f f_f))$$
$$\mathbf{g} = \sigma(W_g [\mathbf{h}_s; \mathbf{h}_f] + b_g)$$
$$\mathbf{f}_{\text{gated}} = [\mathbf{g} \odot \mathbf{h}_s ; (1 - \mathbf{g}) \odot \mathbf{h}_f] + [\mathbf{h}_s; \mathbf{h}_f]$$

```python
class GatedFusion(nn.Module):
    """Dynamic Gated Residual Concatenation Fusion Block."""

    def __init__(self, spatial_dim: int = 1792, frequency_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.proj_spatial = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.proj_frequency = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Dynamic gating generator: [h_s; h_f] -> gate weights g in [0, 1]
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        # Initialize gate bias to 0 for initial 50-50 balance
        nn.init.constant_(self.gate_fc[-2].bias, 0.0)

        # Output projection back to 1024-d
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.30),
        )

    def forward(self, f_spatial: torch.Tensor, f_frequency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Project to common hidden subspace
        h_s = self.proj_spatial(f_spatial)      # (B, 1024)
        h_f = self.proj_frequency(f_frequency)  # (B, 1024)

        # 2. Joint representation
        joint = torch.cat([h_s, h_f], dim=1)    # (B, 2048)

        # 3. Dynamic element-wise gating coefficients
        gate = self.gate_fc(joint)              # (B, 1024)

        # 4. Gated residual combination: guarantees non-zero gradient flow
        # d(fused)/d(h_s) >= 1.0 and d(fused)/d(h_f) >= 1.0
        gated_s = gate * h_s
        gated_f = (1.0 - gate) * h_f
        gated_joint = torch.cat([gated_s, gated_f], dim=1) + joint  # (B, 2048)

        # 5. Refined joint multi-modal embedding
        fused = self.out_proj(gated_joint)      # (B, 1024)
        return fused, gate
```

---

## 4. Facial Seam Margin Expansion & Tilt Alignment
**File:** [`src/preprocessing/face_aligner.py`](file:///c:/Users/Bhargav/Documents/projects/GUARD-PBL/src/preprocessing/face_aligner.py)

Standard facial detectors tightly crop faces, which destroys the primary clue for deepfakes: **blending seams around the jawline, ears, and hairline**. GUARD adds a configurable 25% boundary expansion and eye-tilt correction.

```python
def extract_and_align_face(
    image_bgr: np.ndarray,
    face_box: Tuple[int, int, int, int],
    crop_margin: float = 0.25,
    target_size: Tuple[int, int] = (380, 380)
) -> np.ndarray:
    """
    Expands bounding box by margin and resizes to target spatial resolution.
    Preserves jawline, hairline, and ear blending seams.
    """
    x, y, w, h = face_box
    img_h, img_w = image_bgr.shape[:2]

    # Calculate padding margins
    margin_x = int(w * crop_margin)
    margin_y = int(h * crop_margin)

    # Clamped expanded coordinates
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    face_crop = image_bgr[y1:y2, x1:x2]

    # Resize with high-quality Lanczos/area interpolation
    interp = cv2.INTER_AREA if (face_crop.shape[0] > target_size[0]) else cv2.INTER_CUBIC
    aligned_face = cv2.resize(face_crop, target_size, interpolation=interp)
    return aligned_face
```

---

## 5. Visual Explainability: 2D FFT Spectrogram & Heatmap Overlays
**Files:** [`src/explainability/frequency.py`](file:///c:/Users/Bhargav/Documents/projects/GUARD-PBL/src/explainability/frequency.py), [`src/explainability/gradcam.py`](file:///c:/Users/Bhargav/Documents/projects/GUARD-PBL/src/explainability/gradcam.py)

Provides visual transparency to non-technical users by extracting log-magnitude Fourier spectra alongside spatial heatmaps.

```python
def compute_2d_fft_spectrum(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D Fast Fourier Transform log-magnitude spectrum colored with TURBO colormap.
    Exposes high-frequency upsampling artifacts and checkerboard patterns.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # 1. 2D FFT and DC centering
    fft2d = np.fft.fft2(gray)
    fft_shifted = np.fft.fftshift(fft2d)

    # 2. Log magnitude compression: S = log(1 + |F|)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude)

    # 3. Robust percentile normalization (1% - 99.5%) to remove outliers
    min_val = np.percentile(log_magnitude, 1.0)
    max_val = np.percentile(log_magnitude, 99.5)
    norm_mag = np.clip((log_magnitude - min_val) / (max_val - min_val + 1e-8), 0.0, 1.0)
    uint8_mag = (norm_mag * 255.0).astype(np.uint8)

    # 4. Colorize with Turbo colormap
    colored_bgr = cv2.applyColorMap(uint8_mag, cv2.COLORMAP_TURBO)
    display_spectrum = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return display_spectrum, norm_mag


def create_heatmap_overlay(
    image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.50
) -> np.ndarray:
    """Blends a 2D float heatmap [0.0, 1.0] onto the original image using OpenCV JET colormap."""
    h, w = image.shape[:2]
    if heatmap.shape[:2] != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    uint8_hm = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)
    color_bgr = cv2.applyColorMap(uint8_hm, cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1.0 - alpha, color_rgb, alpha, 0)
```

---

## 6. In-Memory Degradation Stress-Testing Pipeline
**File:** [`src/stress/perturbation.py`](file:///c:/Users/Bhargav/Documents/projects/GUARD-PBL/src/stress/perturbation.py)

Social media uploads (WhatsApp, Instagram, Twitter) heavily recompress and downscale images. This module subjects candidate images to live in-memory corruptions to evaluate empirical stability:

```python
def apply_jpeg_compression(image_rgb: np.ndarray, quality: int) -> np.ndarray:
    """Simulates in-memory JPEG compression and re-decoding via OpenCV buffers."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 1, 100))]
    success, encimg = cv2.imencode(".jpg", bgr, encode_param)
    if not success:
        return image_rgb.copy()
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)


def apply_resize_degradation(image_rgb: np.ndarray, scale: float) -> np.ndarray:
    """Simulates mobile thumbnailing / downscale & bicubic upsampling degradation."""
    h, w = image_rgb.shape[:2]
    new_w, new_h = max(16, int(w * scale)), max(16, int(h * scale))
    downsampled = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_CUBIC)
    return restored
```

---

## 7. Scene-Cut Aware Video Forensics & Temporal Flicker Analysis
**File:** [`src/video/processor.py`](file:///c:/Users/Bhargav/Documents/projects/GUARD-PBL/src/video/processor.py)

Shorts and Reels have frequent jump cuts that falsely trigger naive temporal difference algorithms. GUARD uses **HSV color histogram correlation** to distinguish legitimate camera cuts from AI temporal morphing:

```python
def is_scene_cut(prev_frame_rgb: np.ndarray, curr_frame_rgb: np.ndarray) -> Tuple[bool, float, float]:
    """
    Detects whether frame transition is an editing jump cut rather than AI morphing flicker.
    Uses L1 pixel delta combined with 2D HSV color histogram correlation.
    """
    p_small = cv2.resize(prev_frame_rgb, (128, 128))
    c_small = cv2.resize(curr_frame_rgb, (128, 128))

    # 1. Pixel L1 absolute mean difference
    l1_diff = float(np.mean(np.abs(c_small.astype(np.float32) - p_small.astype(np.float32))) / 255.0)

    # 2. HSV 2D color histogram correlation
    p_hsv = cv2.cvtColor(p_small, cv2.COLOR_RGB2HSV)
    c_hsv = cv2.cvtColor(c_small, cv2.COLOR_RGB2HSV)
    h_p = cv2.calcHist([p_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    h_c = cv2.calcHist([c_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(h_p, h_p, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(h_c, h_c, 0, 1, cv2.NORM_MINMAX)

    hist_corr = float(cv2.compareHist(h_p, h_c, cv2.HISTCMP_CORREL))

    # Scene cut condition: Major color histogram shift (corr <= 0.45) or large pixel delta (>= 0.38)
    is_cut = (l1_diff >= 0.38) or (hist_corr <= 0.45)
    return is_cut, l1_diff, hist_corr


def compute_temporal_flicker(
    delta_prob: float,
    laplacian_var: float,
    is_cut: bool
) -> float:
    """Computes combined temporal flicker score, zeroing out editing jump cuts."""
    if is_cut:
        return 0.0  # Exclude intentional editorial transitions

    # High probability fluctuation combined with high-frequency edge variance
    normalized_laplacian = min(1.0, laplacian_var / 500.0)
    return float(0.70 * delta_prob + 0.30 * normalized_laplacian)
```

---

## 8. Unified Production Inference Engine
**File:** [`src/inference.py`](file:///c:/Users/Bhargav/Documents/projects/GUARD-PBL/src/inference.py)

Combines preprocessing, hardware auto-detection (CUDA FP16 vs. CPU FP32), model execution, and dual-modality breakdown into a clean API:

```python
engine = ForensicInferenceEngine(
    checkpoint_path="checkpoints/guard_model.pth",
    enable_face_crop=True
)

# Run complete dual-stream forensic inspection
results = engine.inspect("test_image.jpg", threshold=0.50)

print(f"Verdict: {results['verdict']}")                          # "Fake (AI-Generated)" or "Real"
print(f"AI Probability: {results['fake_probability']:.2%}")
print(f"Spatial Stream Logit: {results['spatial_logit']:.3f}")
print(f"Frequency Stream Logit: {results['frequency_logit']:.3f}")
print(f"Gating Balance α: {results['gate_weights'].mean():.3f}")
```
