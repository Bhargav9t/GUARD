# ==============================================================================
# File: app.py
# Description: Uiverse.io-Inspired Corporate Minimalist Web Dashboard for GUARD.
#
# Design System:
#   - Slate gray canvas (#f8fafc)
#   - Deep navy primary typography and accents (#0f172a) - zero white-on-white text
#   - Crisp tactile cards with subtle border highlights and soft elevation
#   - Clean enterprise typography and metrics
#
# Core Capabilities:
#   - Real-time forensic scan with tunable decision threshold
#   - Checkpoint status indicator (calibrated vs untrained fallback warning)
#   - Individual Stream Breakdown: Spatial Stream (EfficientNet-B4) vs.
#     Frequency Stream (2D FFT / DCT) confidence, logits, and gated weights
#   - Verdict Summary Card with large REAL / FAKE badge and confidence progress bar
#   - Tri-view visual forensics (Aligned Face & Bounding Box, 2D FFT Spectrum, Spatial Heatmap)
#   - Preprocessing transmission degradation simulation (JPEG + Gaussian blur)
#   - Telemetry and Debug expander showing tensor shapes, device allocations, and ranges
# ==============================================================================

# Import operating system and filesystem utilities
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Import scientific computing and image processing libraries
import numpy as np
import cv2
import torch
from PIL import Image

# Import Streamlit framework for interactive web interface
import streamlit as st

# Configure page layout and title (must be the first Streamlit command)
st.set_page_config(
    page_title="GUARD // AI Image Forensics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import GUARD core configuration and modules via package interfaces
from configs import default_config, GuardConfig
from src.model import GUARDModel, load_guard_checkpoint
from src.preprocessing import FaceAligner
from src.data import get_degradation_sweep_pipeline
from src.utils import get_device, get_device_info
from src.ui import (
    inject_custom_css,
    tensor_to_display_image,
    create_spectrum_display,
    create_spatial_heatmap_display,
)


@st.cache_resource(show_spinner="Initializing GUARD Deepfake Forensic Engine...")
def get_cached_guard_model(
    checkpoint_path: Optional[str] = None
) -> Tuple[GUARDModel, bool, Dict[str, Any], torch.device]:
    """
    Instantiates and caches the GUARD neural architecture and weights in memory.
    Uses st.cache_resource to prevent redundant model reloading across UI interactions.

    Parameters:
        checkpoint_path (Optional[str]): Filepath to the saved model checkpoint.
            If None, resolves to the configured checkpoints directory.

    Returns:
        Tuple[GUARDModel, bool, Dict[str, Any], torch.device]:
            - model: Initialized GUARD neural network.
            - is_trained: Boolean indicating whether calibrated weights were loaded.
            - status_info: Dictionary detailing checkpoint metadata or warning state.
            - device: Active PyTorch computing device (CUDA / CPU).
    """
    # 1. Resolve hardware device
    device = get_device()

    # 2. Resolve default checkpoint path dynamically from PathConfig
    if checkpoint_path is None:
        target_path = str(default_config.paths.checkpoints_dir / "guard_model.pth")
    else:
        target_path = checkpoint_path

    # 3. Safely load checkpoint with fallback handling
    model, is_trained, status_info = load_guard_checkpoint(
        checkpoint_path=target_path,
        device=device,
        config=default_config
    )

    return model, is_trained, status_info, device


@st.cache_resource(show_spinner=False)
def get_cached_face_aligner() -> FaceAligner:
    """
    Instantiates and caches the FaceAligner and its Haar Cascades in memory
    to avoid repeated filesystem disk I/O on each UI interaction.

    Returns:
        FaceAligner: Initialized face alignment engine.
    """
    return FaceAligner()


def generate_synthetic_demo_image(sample_type: str = "synthetic", sample_index: int = 0) -> np.ndarray:
    """
    Generates or retrieves benchmark demonstration portraits for instant testing without file upload.
    Checks for calibrated benchmark files on disk in data/train/ first, falling back to
    the procedural calibration generator matching the trained model distribution.

    Parameters:
        sample_type (str): 'authentic' (natural continuous gradients) or
                           'synthetic' (injected high-frequency checkerboard grid noise).
        sample_index (int): Deterministic index variation.

    Returns:
        np.ndarray: RGB image (380, 380, 3) in uint8 [0, 255].
    """
    # 1. Check if calibrated benchmark samples exist on disk
    base_dir = Path("data/train")
    target_sub = "fake" if sample_type == "synthetic" else "real"
    target_folder = base_dir / target_sub
    if target_folder.exists():
        disk_files = sorted(list(target_folder.glob("*.png")) + list(target_folder.glob("*.jpg")))
        if len(disk_files) > sample_index:
            disk_path = disk_files[sample_index % len(disk_files)]
            bgr = cv2.imread(str(disk_path))
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # 2. Fall back to exact procedural calibration generator matching train.py
    try:
        from train import generate_calibration_bootstrap_sample
        kind = "fake" if sample_type == "synthetic" else "real"
        return generate_calibration_bootstrap_sample(kind, index=sample_index, image_size=(380, 380))
    except ImportError:
        pass

    # 3. Native fallback generator with exact spectral noise profile
    h, w = 380, 380
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    face_radius = 110
    face_mask = np.exp(-(radius ** 2) / (2 * (face_radius ** 2)))

    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :, 0] = face_mask * 215.0 + 30.0
    base[:, :, 1] = face_mask * 175.0 + 20.0
    base[:, :, 2] = face_mask * 155.0 + 15.0

    if sample_type == "synthetic":
        # AI high-frequency periodic grid noise
        freq_grid = np.sin(2 * np.pi * x / 8.0) * np.cos(2 * np.pi * y / 8.0)
        base += freq_grid[:, :, None] * 25.0
        # Radial spectral harmonic rings
        harmonic_ring = np.sin(2 * np.pi * radius / 16.0) * 15.0
        base += harmonic_ring[:, :, None] * face_mask[:, :, None]
    else:
        # Authentic camera continuous lighting gradient
        lighting = (x / w) * 25.0
        base += lighting[:, :, None]
        noise = np.random.normal(0, 3.0, (h, w, 3)).astype(np.float32)
        base += noise

    return np.clip(base, 0, 255).astype(np.uint8)


def render_sidebar(
    is_trained: bool,
    status_info: Dict[str, Any],
    device_info: Dict[str, Any]
) -> Tuple[float, bool, bool, int, int]:
    """
    Renders the corporate minimalist sidebar controls, calibration slider,
    and checkpoint health monitor.

    Parameters:
        is_trained (bool): Whether a trained checkpoint is active.
        status_info (Dict[str, Any]): Checkpoint metadata dictionary.
        device_info (Dict[str, Any]): Compute hardware details.

    Returns:
        Tuple[float, bool, bool, int, int]:
            - decision_thresh: Selected classification threshold.
            - enable_face_crop: Face alignment toggle state.
            - enable_degradation: Transmission simulation toggle state.
            - jpeg_quality: JPEG degradation quality factor.
            - blur_kernel: Gaussian blur kernel size.
    """
    # 1. Checkpoint Status Section
    st.sidebar.markdown("### Model Checkpoint")
    if is_trained:
        st.sidebar.markdown(
            """
            <div class="status-pill-calibrated">
                <span>●</span> Calibrated Weights Loaded
            </div>
            """,
            unsafe_allow_html=True
        )
        st.sidebar.caption(f"File: `{Path(status_info['checkpoint_path']).name}`")
        if status_info.get("accuracy") is not None:
            st.sidebar.caption(f"Calibration Accuracy: **{status_info['accuracy']:.1f}%**")
        if status_info.get("epoch") is not None:
            st.sidebar.caption(f"Trained Epochs: **{status_info['epoch']}**")
    else:
        st.sidebar.markdown(
            """
            <div class="status-pill-untrained">
                <span>▲</span> Untrained Fallback Mode
            </div>
            """,
            unsafe_allow_html=True
        )
        st.sidebar.warning(
            "Untrained backbone loaded.\n\n"
            "To calibrate the model on your local images and eliminate false positives, run:\n"
            "`python train.py --epochs 10`"
        )

    st.sidebar.markdown("---")

    # 2. Tunable Decision Boundary Threshold Slider
    st.sidebar.markdown("### Detection Sensitivity")
    decision_thresh = st.sidebar.slider(
        "Fake Decision Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Images with a fake probability greater than or equal to this threshold are flagged as AI-Generated."
    )
    st.sidebar.caption(
        f"Active boundary: **{decision_thresh:.2f}** "
        f"(Lower = more sensitive to fakes, Higher = reduces false fakes)"
    )

    st.sidebar.markdown("---")

    # 3. Preprocessing and Alignment Controls
    st.sidebar.markdown("### Preprocessing Pipeline")
    enable_face_crop = st.sidebar.checkbox(
        "Automated Face Alignment",
        value=True,
        help="Detects facial landmarks, corrects rotation, and extracts a canonical 380x380 crop."
    )

    enable_degradation = st.sidebar.checkbox(
        "Simulate Transmission Degradation",
        value=False,
        help="Injects real-world compression or blur before inspection to assess model resilience."
    )

    jpeg_quality = 100
    blur_kernel = 1
    if enable_degradation:
        jpeg_quality = st.sidebar.slider("JPEG Compression Quality", 20, 95, 60)
        blur_kernel = st.sidebar.select_slider("Gaussian Blur Kernel", options=[1, 3, 5, 7], value=1)

    st.sidebar.markdown("---")

    # 4. System Telemetry Information
    st.sidebar.markdown("### System Telemetry")
    st.sidebar.text(f"Device: {device_info['device_name']}")
    st.sidebar.text(f"PyTorch: {device_info['pytorch_version']}")
    if device_info.get("cuda_available"):
        st.sidebar.text(f"VRAM: {device_info['total_memory_gb']} GB")

    return decision_thresh, enable_face_crop, enable_degradation, jpeg_quality, blur_kernel


def main() -> None:
    """
    Main Streamlit application entry point rendering the corporate minimalist dashboard.
    """
    # 1. Apply Corporate Minimalist CSS Styling System (Uiverse.io-inspired)
    inject_custom_css()

    # 2. Initialize and Cache Model
    model, is_trained, status_info, device = get_cached_guard_model()
    device_info = get_device_info()

    # 3. Render Sidebar Controls and Checkpoint Health Monitor
    (
        decision_thresh,
        enable_face_crop,
        enable_degradation,
        jpeg_quality,
        blur_kernel,
    ) = render_sidebar(is_trained, status_info, device_info)

    # 4. Render Corporate Header Banner
    st.markdown(
        """
        <div class="corporate-header">
            <div class="corporate-title">
                🛡️ GUARD Forensic Intelligence
            </div>
            <div class="corporate-subtitle">
                Dual-Stream Spatial-Frequency Deepfake & AI-Generated Image Forensics System
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5. Image Acquisition Section (Container with Clean Mode Selector)
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-title">📁 Image Acquisition</div>
            """,
            unsafe_allow_html=True
        )

        input_mode = st.radio(
            "Select Input Source Mode:",
            options=["Upload Image File", "Benchmark Preset Samples"],
            horizontal=True,
            label_visibility="collapsed"
        )

        active_image = None

        if input_mode == "Upload Image File":
            uploaded_file = st.file_uploader(
                "Upload Image for Forensic Inspection",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                help="Supports high-resolution images up to 25MB."
            )
            if uploaded_file is not None:
                try:
                    pil_image = Image.open(uploaded_file).convert("RGB")
                    active_image = np.array(pil_image)
                except Exception as img_err:
                    st.error(f"Failed to read image: {img_err}")
                    return
        else:
            preset_choice = st.radio(
                "Benchmark Demonstration Presets:",
                options=["Authentic Camera Image", "AI-Generated Diffusion Image"],
                index=0,
                horizontal=True
            )
            if preset_choice == "Authentic Camera Image":
                active_image = generate_synthetic_demo_image("authentic")
                st.caption("ℹ️ Loaded authentic synthetic portrait with natural continuous gradients.")
            else:
                active_image = generate_synthetic_demo_image("synthetic")
                st.caption("ℹ️ Loaded generative synthetic portrait with high-frequency lattice noise.")

    # Stop rendering if no input image is available
    if active_image is None:
        st.info("💡 Please upload an image or choose a demo preset above to begin forensic analysis.")
        return

    # 6. Apply Optional Transmission Degradation Simulator
    if enable_degradation:
        active_jpeg = jpeg_quality if jpeg_quality < 100 else None
        active_blur = blur_kernel if blur_kernel > 1 else None

        if active_jpeg is not None or active_blur is not None:
            sweep_pipeline = get_degradation_sweep_pipeline(
                jpeg_quality=active_jpeg,
                blur_limit=active_blur,
                target_size=(active_image.shape[0], active_image.shape[1])
            )
            active_image = sweep_pipeline(image=active_image)["image"]
            st.caption(f"⚠️ Applied transmission degradation: JPEG Quality={jpeg_quality}, Blur Kernel={blur_kernel}")

    # 7. Face Detection and Canonical Alignment Preprocessing
    aligner = get_cached_face_aligner()
    aligner.config.enable_face_crop = enable_face_crop

    proc_result = aligner.process(active_image)
    aligned_rgb = proc_result["aligned_image"]
    face_detected = proc_result["face_detected"]
    face_bbox = proc_result.get("bbox")

    # 8. Format Image into PyTorch Tensor
    # Convert uint8 (H, W, 3) to normalized float32 tensor (1, 3, 380, 380) in [0.0, 1.0]
    tensor_input = aligned_rgb.astype(np.float32) / 255.0
    tensor_input = np.transpose(tensor_input, (2, 0, 1))
    tensor_input = torch.from_numpy(tensor_input).unsqueeze(0).to(device)

    # 9. Execute Two-Stream Forensic Model Inference
    with torch.no_grad():
        explain_data = model.extract_explainability_data(tensor_input, threshold=decision_thresh)
        pred_info = explain_data["prediction"]

    # Extract primary verdict and telemetry values
    is_fake = pred_info["is_fake"]
    fake_prob = pred_info["fake_probability"]
    confidence = pred_info["confidence"]
    spatial_ratio = pred_info["spatial_gating_ratio"]
    freq_ratio = pred_info["frequency_gating_ratio"]
    stream_data = pred_info.get("stream_breakdown", {})

    # Extract individual stream metrics
    spatial_logit = stream_data.get("spatial_logit", 0.0)
    spatial_prob = stream_data.get("spatial_prob", fake_prob)
    spatial_conf = stream_data.get("spatial_confidence", confidence)

    frequency_logit = stream_data.get("frequency_logit", 0.0)
    frequency_prob = stream_data.get("frequency_prob", fake_prob)
    frequency_conf = stream_data.get("frequency_confidence", confidence)

    # 10. Render Verdict Summary Card (Uiverse.io Style High-Visibility)
    st.markdown("### Forensic Verdict")

    if is_fake:
        st.markdown(
            f"""
            <div class="verdict-card-fake">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="verdict-badge-fake">⚠️ FAKE / AI-GENERATED</span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: #b91c1c;">
                        Certainty: {confidence * 100:.1f}%
                    </span>
                </div>
                <div style="margin-top: 14px; font-size: 0.95rem; color: #4b5563; line-height: 1.5;">
                    The forensic system detected synthetic generative artifacts. 
                    Calculated Fake Probability: <strong>{fake_prob * 100:.2f}%</strong> 
                    (Active Decision Threshold: {decision_thresh:.2f}).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-card-real">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span class="verdict-badge-real">✅ VERIFIED AUTHENTIC</span>
                    <span style="font-size: 1.15rem; font-weight: 800; color: #15803d;">
                        Certainty: {confidence * 100:.1f}%
                    </span>
                </div>
                <div style="margin-top: 14px; font-size: 0.95rem; color: #4b5563; line-height: 1.5;">
                    The image exhibits natural sensor characteristics and continuous gradients without generative noise. 
                    Calculated Authentic Probability: <strong>{(1.0 - fake_prob) * 100:.2f}%</strong> 
                    (Active Decision Threshold: {decision_thresh:.2f}).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Progress bar illustrating confidence certainty
    st.progress(
        min(max(float(confidence), 0.0), 1.0),
        text=f"Forensic Decision Confidence: {confidence * 100:.1f}%"
    )

    # 11. Stream Breakdown Section
    st.markdown("### Stream Breakdown")
    col_str1, col_str2 = st.columns(2)

    with col_str1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">🔹 Spatial Stream (EfficientNet-B4)</div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="stream-block">
                    <div class="stream-metric-row">
                        <span class="stream-metric-label">Raw Spatial Logit</span>
                        <span class="stream-metric-value">{spatial_logit:+.4f}</span>
                    </div>
                    <div class="stream-metric-row">
                        <span class="stream-metric-label">Stream Fake Probability</span>
                        <span class="stream-metric-value">{spatial_prob * 100:.1f}%</span>
                    </div>
                    <div class="stream-metric-row">
                        <span class="stream-metric-label">Stream Confidence Score</span>
                        <span class="stream-metric-value">{spatial_conf * 100:.1f}%</span>
                    </div>
                    <div class="stream-metric-row">
                        <span class="stream-metric-label">Gated Fusion Weight</span>
                        <span class="stream-metric-value">{spatial_ratio * 100:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Visual progress bar for spatial fake probability
            st.progress(
                min(max(float(spatial_prob), 0.0), 1.0),
                text=f"Spatial Fake Likelihood: {spatial_prob * 100:.1f}%"
            )
            st.caption("Focuses on pixel-level facial geometry, warping, and boundary blending artifacts.")

    with col_str2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">🔸 Frequency Stream (FFT / DCT)</div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(
                f"""
                <div class="stream-block">
                    <div class="stream-metric-row">
                        <span class="stream-metric-label">Raw Frequency Logit</span>
                        <span class="stream-metric-value">{frequency_logit:+.4f}</span>
                    </div>
                    <div class="stream-metric-row">
                        <span class="stream-metric-label">Stream Fake Probability</span>
                        <span class="stream-metric-value">{frequency_prob * 100:.1f}%</span>
                    </div>
                    <div class="stream-metric-row">
                        <span class="stream-metric-label">Stream Confidence Score</span>
                        <span class="stream-metric-value">{frequency_conf * 100:.1f}%</span>
                    </div>
                    <div class="stream-metric-row">
                        <span class="stream-metric-label">Gated Fusion Weight</span>
                        <span class="stream-metric-value">{freq_ratio * 100:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            # Visual progress bar for frequency fake probability
            st.progress(
                min(max(float(frequency_prob), 0.0), 1.0),
                text=f"Frequency Fake Likelihood: {frequency_prob * 100:.1f}%"
            )
            st.caption("Focuses on spectral anomalies, periodic lattice patterns, and generative upsampling noise.")

    # 12. Multi-Modal Forensic Tri-View Visual Inspection
    st.markdown("### Multi-Modal Forensic Tri-View")
    col_v1, col_v2, col_v3 = st.columns(3)

    # Panel 1: Input & Alignment (Canonical Crop and Bounding Box Tabs)
    with col_v1:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">1. Input & Alignment</div>
                """,
                unsafe_allow_html=True
            )

            annotated_display = active_image.copy()
            if face_detected and face_bbox is not None:
                bx, by, bw, bh = face_bbox
                cv2.rectangle(annotated_display, (bx, by), (bx + bw, by + bh), (34, 197, 94), 3)

            tab_crop, tab_bbox = st.tabs(["Canonical Crop", "Face Bounding Box"])

            with tab_crop:
                if face_detected:
                    st.caption("🟢 Aligned Canonical 380×380 Model View")
                else:
                    st.caption("ℹ️ Center-Cropped 380×380 (No Face Detected)")
                st.image(aligned_rgb, use_container_width=True)

            with tab_bbox:
                if face_detected and face_bbox is not None:
                    st.caption(f"Detected BBox: (x={face_bbox[0]}, y={face_bbox[1]}, w={face_bbox[2]}, h={face_bbox[3]})")
                else:
                    st.caption("ℹ️ Full-Scene Source Image")
                st.image(annotated_display, use_container_width=True)

    # Panel 2: 2D FFT & DCT Magnitude Spectrum
    with col_v2:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">2. 2D FFT/DCT Magnitude Spectrum</div>
                """,
                unsafe_allow_html=True
            )
            spectrum_display = create_spectrum_display(explain_data["spectrum_tensor"])
            st.caption("Spectral Energy Distribution (Centralized Log-Magnitude)")
            st.image(spectrum_display, use_container_width=True)

    # Panel 3: Spatial Activation Heatmap
    with col_v3:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title">3. Spatial Activation Heatmap</div>
                """,
                unsafe_allow_html=True
            )
            spatial_overlay = create_spatial_heatmap_display(
                image_rgb=aligned_rgb,
                heatmap_tensor=explain_data["spatial_heatmap_tensor"],
                alpha=0.50
            )
            st.caption("High-Activation Forensic Discrepancy Regions")
            st.image(spatial_overlay, use_container_width=True)

    # 13. Debug & Tensor Telemetry Expander
    with st.expander("🛠️ Debug Information & Tensor Telemetry"):
        st.markdown("**Preprocessing & Tensor Diagnostics**")
        st.json({
            "input_shape": list(tensor_input.shape),
            "input_dtype": str(tensor_input.dtype),
            "input_min": float(tensor_input.min().item()),
            "input_max": float(tensor_input.max().item()),
            "device": str(device),
            "face_detected": bool(face_detected),
            "face_bbox": list(face_bbox) if face_bbox is not None else None,
            "active_threshold": float(decision_thresh),
            "raw_logit": float(pred_info.get("logit", 0.0)),
            "fake_probability": float(fake_prob),
            "decision_confidence": float(confidence),
            "spatial_gating_ratio": float(spatial_ratio),
            "frequency_gating_ratio": float(freq_ratio),
            "checkpoint_status": status_info,
        })


# Standard entry point execution
if __name__ == "__main__":
    main()
