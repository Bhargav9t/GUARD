"""
GUARD: AI-Generated Image & Video Forensic Detection System
Modular forensic intelligence platform integrating multi-backbone neural consensus,
2D Fourier high-frequency physics, camera sensor noise residuals,
in-memory degradation stress-testing, and scene-cut filtered temporal flicker.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Configure page layout and title (must be the first Streamlit command)
st.set_page_config(
    page_title="GUARD — AI Media Forensics & Verification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import modular backend components
from src.detectors import get_detector, SUPPORTED_ENGINES, BaseDetector, DetectionResult
from src.explainability import create_heatmap_overlay, compute_2d_fft_spectrum
from src.stress import run_degradation_stress_test, StressTestSummary
from src.video import (
    process_video_pipeline,
    create_plotly_video_timeline,
    create_matplotlib_video_timeline,
    VideoAnalysisResult,
    download_video_from_url,
    VideoDownloadError,
    VideoDurationExceededError,
    identify_platform,
)
from src.ui.styles import inject_custom_css
from src.ui.disclaimers import render_honest_disclaimers, render_engine_badge


# -----------------------------------------------------------------------------
# Caching Model Engine
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_detector(
    engine_name: str,
    force_cpu: bool,
    use_fp16: bool,
) -> BaseDetector:
    """Instantiates and caches the detector backend in memory."""
    return get_detector(
        engine_name=engine_name,
        force_cpu=force_cpu,
        use_fp16=use_fp16,
    )


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # 1. Inject modern editorial styling
    inject_custom_css()

    # 2. Sidebar Configuration
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.12);">
                <div style="width: 38px; height: 38px; border-radius: 6px; background: #ffffff; border: 1px solid rgba(255, 255, 255, 0.3); display: flex; align-items: center; justify-content: center; font-weight: 700; color: #00539b; font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif; font-size: 1.35rem; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25);">G</div>
                <div>
                    <div style="font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif; font-size: 1.15rem; font-weight: 700; color: #ffffff; letter-spacing: 0.08em; font-variant: small-caps;">GUARD</div>
                    <div style="font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif; font-size: 0.66rem; color: #82b9e9; letter-spacing: 0.08em; font-variant: small-caps;">Forensic Media Intelligence</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Architecture & Engine")
        selected_engine = st.selectbox(
            "Backbone Architecture",
            options=list(SUPPORTED_ENGINES.keys()),
            index=0,
            help="Select the inference model. Default uses multi-evidence hybrid consensus. You can switch to Phase 2 Dual-Stream without changing the UI.",
        )

        st.markdown("### Runtime Hardware")
        cuda_available = False
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                st.caption(f"CUDA Acceleration Active: **{gpu_name}**")
            else:
                st.caption("No CUDA GPU detected. Running on CPU.")
        except Exception:
            st.caption("PyTorch CPU fallback active.")

        force_cpu = st.checkbox(
            "Force CPU Execution",
            value=not cuda_available,
            help="Force inference on CPU even if CUDA is available.",
        )

        use_fp16 = st.checkbox(
            "Enable CUDA FP16 (Half Precision)",
            value=True,
            disabled=force_cpu or not cuda_available,
            help="Runs FP16 on NVIDIA RTX 3050 (conserves VRAM, operates safely within memory budget).",
        )

        st.markdown("### Domain & Calibration")
        domain_mode = st.selectbox(
            "Media Domain Mode",
            ["Auto-Detect", "AI Poster / Graphic Art", "Photographic / Portrait", "Social Media / Compressed"],
            index=0,
            help="Select the media domain. For AI posters or stylized graphics, select 'AI Poster / Graphic Art' to activate diffusion upsampling grain analysis.",
        )

        sensitivity_choice = st.radio(
            "Detection Sensitivity Profile",
            ["Balanced (0.50)", "Aggressive AI (0.35)", "Strict Realism (0.65)", "Custom"],
            index=0,
            help="Balanced: standard decision boundary. Aggressive AI: catches subtle AI art, posters, and Midjourney. Strict Realism: prioritizes preventing false alarms on authentic photos.",
        )

        if sensitivity_choice == "Custom":
            threshold = st.slider("Custom Classification Threshold", 0.10, 0.90, 0.50, 0.05)
        else:
            threshold = {"Balanced (0.50)": 0.50, "Aggressive AI (0.35)": 0.35, "Strict Realism (0.65)": 0.65}[sensitivity_choice]

        st.markdown("### Spatial Evidence Layer")
        heatmap_layer = st.selectbox(
            "Spatial Evidence Map Type",
            ["Combined Multi-Evidence", "Vision Transformer Attention", "Artifact Edge Discontinuities", "Noise Residual & Diffusion Texture"],
            index=0,
            help="Switch between different forensic evidence representations.",
        )
        layer_method = {
            "Combined Multi-Evidence": "combined",
            "Vision Transformer Attention": "vision",
            "Artifact Edge Discontinuities": "edges",
            "Noise Residual & Diffusion Texture": "noise",
        }[heatmap_layer]

        overlay_alpha = st.slider("Heatmap Blending Alpha", 0.10, 0.90, 0.50, 0.05)
        colormap_choice = st.selectbox("Heatmap Colormap", ["Jet", "Turbo", "Inferno"])
        cv2_colormap = {
            "Jet": cv2.COLORMAP_JET,
            "Turbo": cv2.COLORMAP_TURBO,
            "Inferno": cv2.COLORMAP_INFERNO,
        }.get(colormap_choice, cv2.COLORMAP_JET)

        st.markdown("---")
        st.markdown(
            "<div style='color: #82b9e9; font-size: 0.70rem; line-height: 1.5; font-family: \"Copperplate Gothic\", \"Cinzel\", serif; font-variant: small-caps;'>GUARD Forensic Platform – Verifiable Media Intelligence</div>",
            unsafe_allow_html=True,
        )

    # 3. Load Detector Backend
    with st.spinner("Initializing forensic inference engine..."):
        try:
            detector = load_cached_detector(
                engine_name=selected_engine,
                force_cpu=force_cpu,
                use_fp16=use_fp16,
            )
        except Exception as e:
            st.error(f"Failed to load detector engine: {e}")
            return

    # 4. Editorial Header & Engine Telemetry Ribbon
    st.markdown(
        """
        <div class="hero-header">
            <div class="hero-eyebrow">Forensic Media Intelligence</div>
            <h1 class="hero-title">Synthetic media verification with <em>multimodal evidence</em></h1>
            <p class="hero-subtitle">
                Comprehensive forensic inspection integrating vision transformer consensus, 2D Fourier spectral physics, sensor noise residuals, and temporal continuity analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_engine_badge(
        model_name=detector.model_name,
        device=str(detector.device),
        dtype="FP16" if detector.is_fp16 else "FP32",
    )

    # 5. Scientific Honest UI Disclaimers
    render_honest_disclaimers()

    # 6. Main Operational Tabs: Single Image vs Video Pipeline
    tab_image, tab_video = st.tabs([
        "Image Forensics & Stress Test",
        "Video Forensics & Temporal Flicker",
    ])

    # =========================================================================
    # TAB 1: SINGLE IMAGE FORENSICS
    # =========================================================================
    with tab_image:
        col_input, col_preset = st.columns([2, 1])

        with col_input:
            uploaded_file = st.file_uploader(
                "Upload Image for Forensic Inspection",
                type=["png", "jpg", "jpeg", "webp"],
                help="Supports standard formats. Media is processed entirely in-memory.",
            )

        with col_preset:
            st.markdown("<div style='font-size: 0.82rem; font-weight: 500; color: #a1a1aa; margin-bottom: 6px;'>Benchmark Reference Samples</div>", unsafe_allow_html=True)
            preset_options = [
                "Select a sample...",
                "Celeb-DF (Real Camera Photo)",
                "FaceForensics++ (Real Video Frame)",
                "Latent Diffusion (AI-Generated Photo)",
                "DFDC (AI Face-Swap Deepfake)",
                "AI Graphic Poster Sample",
            ]
            chosen_preset = st.selectbox("Sample Preset", preset_options, label_visibility="collapsed")

        # Resolve image source
        active_image_rgb = None
        source_label = "Uploaded Image"

        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            active_image_rgb = np.array(pil_img)
            source_label = uploaded_file.name
        elif chosen_preset != "Select a sample...":
            preset_map = {
                "Celeb-DF (Real Camera Photo)": "data/benchmarks/Celeb-DF/real/celebdf_real_0000.jpg",
                "FaceForensics++ (Real Video Frame)": "data/benchmarks/FaceForensics++/real/ffpp_real_000.jpg",
                "Latent Diffusion (AI-Generated Photo)": "data/benchmarks/Unseen_Generator/fake/unseen_gen_fake_000.jpg",
                "DFDC (AI Face-Swap Deepfake)": "data/benchmarks/DFDC/fake/dfdc_fake_0000.jpg",
                "AI Graphic Poster Sample": "data/benchmarks/ai_poster_sample.jpg",
            }
            target_path = preset_map.get(chosen_preset)
            if target_path and os.path.exists(target_path):
                active_image_rgb = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)
                source_label = chosen_preset
            else:
                st.warning(f"Preset image not found at: {target_path}")

        # If image is ready, run inference and explainability with session-state caching
        if active_image_rgb is not None:
            st.markdown("---")
            cache_key = f"{source_label}_{active_image_rgb.shape}_{selected_engine}_{domain_mode}"

            if "image_cache" not in st.session_state or st.session_state.get("current_image_key") != cache_key:
                with st.spinner("Executing forensic neural analysis and spectral transforms..."):
                    if hasattr(detector, "predict_image") and "domain_hint" in detector.predict_image.__code__.co_varnames:
                        det_res = detector.predict_image(active_image_rgb, threshold=threshold, domain_hint=domain_mode)
                    else:
                        det_res = detector.predict_image(active_image_rgb, threshold=threshold)

                    spatial_hm = detector.get_spatial_heatmap(active_image_rgb, method=layer_method)
                    fft_display, raw_fft = compute_2d_fft_spectrum(active_image_rgb)

                    st.session_state["image_cache"] = {
                        "det_res": det_res,
                        "spatial_hm": spatial_hm,
                        "spatial_method": layer_method,
                        "fft_display": fft_display,
                        "raw_fft": raw_fft,
                    }
                    st.session_state["current_image_key"] = cache_key
            else:
                cached = st.session_state["image_cache"]
                det_res = cached["det_res"]
                if cached.get("spatial_method") != layer_method:
                    with st.spinner(f"Extracting {layer_method} evidence map..."):
                        spatial_hm = detector.get_spatial_heatmap(active_image_rgb, method=layer_method)
                        cached["spatial_hm"] = spatial_hm
                        cached["spatial_method"] = layer_method
                else:
                    spatial_hm = cached["spatial_hm"]
                fft_display = cached["fft_display"]
                raw_fft = cached["raw_fft"]

            # Blending overlay is computed instantly (<0.005s) without re-running neural network
            hm_overlay = create_heatmap_overlay(active_image_rgb, spatial_hm, alpha=overlay_alpha, colormap=cv2_colormap)

            # 1. Editorial Verdict Card (dynamically updated with threshold in real-time)
            verdict_is_ai = det_res.ai_probability >= threshold
            card_class = "verdict-card-synthetic" if verdict_is_ai else "verdict-card-authentic"
            tag_class = "synthetic" if verdict_is_ai else "authentic"
            dot_class = "rose" if verdict_is_ai else "green"
            status_color = "#e11d48" if verdict_is_ai else "#16a34a"
            verdict_title = "Synthetic Artifacts Detected" if verdict_is_ai else "Authentic Media Confirmed"
            tag_label = "Synthetic Classification" if verdict_is_ai else "Authentic Classification"
            confidence_val = max(det_res.ai_probability, 1.0 - det_res.ai_probability)

            domain_desc = det_res.metadata.get("domain_detected", "Standard Photographic")

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
                        <div style="max-width: 60%;">
                            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                                <span class="status-dot {dot_class}"></span>
                                <span class="editorial-tag {tag_class}">{tag_label}</span>
                                <span style="font-size: 0.74rem; color: #627d98; font-family: 'Copperplate Gothic', 'Cinzel', serif;">Domain: {domain_desc}</span>
                            </div>
                            <h2 class="verdict-heading" style="color: {status_color};">{verdict_title}</h2>
                            <div style="font-size: 0.88rem; color: #334e68; line-height: 1.5; margin-top: 6px;">
                                Inspected Source: <strong style="color: #0a2346;">{source_label}</strong> – Decision threshold calibrated at <code>{threshold:.2f}</code>.
                            </div>
                        </div>
                        <div style="text-align: right; min-width: 180px;">
                            <div style="font-size: 0.72rem; font-weight: 700; color: #00539b; letter-spacing: 0.06em; text-transform: uppercase; font-family: 'Copperplate Gothic', 'Cinzel', serif;">AI Probability Estimate</div>
                            <div style="font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; font-size: 2.6rem; font-weight: 700; color: #0a2346; line-height: 1.1; margin: 4px 0;">{det_res.ai_probability * 100:.1f}%</div>
                            <div style="font-size: 0.78rem; color: #627d98;">
                                Confidence: <span style="color: #00539b; font-weight: 600;">{confidence_val * 100:.1f}%</span> – Latency: <span style="color: #00539b; font-weight: 600;">{det_res.inference_time_ms:.1f}ms</span>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Multi-Evidence Forensic Breakdown Panel
            st.markdown("### Forensic Evidence Decomposition")
            ev_col1, ev_col2, ev_col3, ev_col4 = st.columns(4)

            raw_scores = det_res.raw_scores
            p_umm = raw_scores.get("vision_umm_maybe_ai", det_res.ai_probability)
            p_org = raw_scores.get("vision_organika_ai", None)
            spk_rat = raw_scores.get("frequency_spike_ratio", 0.35)
            r2_fit = raw_scores.get("fft_r_squared", 0.96)
            n_var = raw_scores.get("noise_variance", 5.0)
            is_poster = raw_scores.get("is_poster_domain", 0.0) > 0.5
            has_spikes = spk_rat > 0.40 or r2_fit < 0.92
            has_sensor_noise = n_var >= 1.5

            with ev_col1:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-tile-label">Vision Transformer</div>
                        <div class="metric-tile-value" style="color: {'#e11d48' if det_res.ai_probability > 0.5 else '#16a34a'};">{det_res.ai_probability * 100:.1f}%</div>
                        <div class="metric-tile-caption">Ensemble consensus score</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with ev_col2:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-tile-label">Fourier Spectral Physics</div>
                        <div class="metric-tile-value" style="color: {'#e11d48' if has_spikes else '#16a34a'};">{r2_fit:.2f} R²</div>
                        <div class="metric-tile-caption">{'Anomalous roll-off detected' if has_spikes else 'Natural 1/f decay confirmed'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with ev_col3:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-tile-label">Camera Sensor Noise</div>
                        <div class="metric-tile-value" style="color: {'#16a34a' if has_sensor_noise else '#d97706'};">{n_var:.1f} Var</div>
                        <div class="metric-tile-caption">{'Organic optical grain' if has_sensor_noise else 'Synthetic texture smoothing'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with ev_col4:
                st.markdown(
                    f"""
                    <div class="metric-tile">
                        <div class="metric-tile-label">Domain Classification</div>
                        <div class="metric-tile-value" style="color: #00539b;">{'Graphic Poster' if is_poster else 'Photographic'}</div>
                        <div class="metric-tile-caption">{'Diffusion upsampling profile' if is_poster else 'Optical camera sensor profile'}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Optional Cross-Model Consensus Matrix Expander
            with st.expander("Multi-Model Cross-Examination Matrix (Individual Backbone Estimates)"):
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                with m_c1:
                    st.metric("General AI & Art (umm-maybe)", f"{raw_scores.get('vision_umm_maybe_ai', 0.0)*100:.1f}%")
                with m_c2:
                    st.metric("SDXL Photorealism (Organika)", f"{raw_scores.get('vision_organika_ai', 0.0)*100:.1f}%")
                with m_c3:
                    st.metric("Midjourney Classifier", f"{raw_scores.get('vision_midjourney_ai', 0.0)*100:.1f}%")
                with m_c4:
                    st.metric("Phase 2 Dual-Stream", f"{raw_scores.get('guard_dual_stream_prob', 0.0)*100:.1f}%")

            # 3. Triple Inspection View: Original Image | Grad-CAM Evidence Map | 2D FFT Spectrogram
            st.markdown("### Spatial & Spectral Artifact Inspection")
            col_orig, col_cam, col_fft = st.columns(3)

            with col_orig:
                st.markdown("<div style='font-size: 0.84rem; font-weight: 600; color: #f4f4f6; margin-bottom: 6px;'>Original Image Input</div>", unsafe_allow_html=True)
                st.image(active_image_rgb, use_container_width=True)
                st.caption(f"Dimensions: {active_image_rgb.shape[1]} × {active_image_rgb.shape[0]} px – RGB")

            with col_cam:
                st.markdown("<div style='font-size: 0.84rem; font-weight: 600; color: #f4f4f6; margin-bottom: 6px;'>Spatial Evidence Map</div>", unsafe_allow_html=True)
                st.image(hm_overlay, use_container_width=True)
                if verdict_is_ai:
                    st.markdown(
                        """
                        <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #be123c; margin-top: 8px;">
                            <span class="status-dot rose"></span><strong>Generative Artifact Hotspots:</strong> Highlighted regions correspond to latent diffusion upsampling seams, facial blend boundaries, or synthetic texture smoothing.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem; color: #15803d; margin-top: 8px;">
                            <span class="status-dot green"></span><strong>Authentic Optical Coherence:</strong> Natural depth-of-field, organic physical textures, and uniform Poisson camera sensor grain confirmed across the frame.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with col_fft:
                st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #0a2346; margin-bottom: 6px; font-family: \"Copperplate Gothic\", \"Cinzel\", serif;'>2D Fourier Magnitude Spectrum</div>", unsafe_allow_html=True)
                st.image(fft_display, use_container_width=True)
                st.caption("Log-magnitude Fourier representation: natural photos exhibit isotropic radial decay; generative models exhibit high-frequency flattening or periodic grid artifacts.")

            # 4. Degradation Stress Test Panel
            st.markdown("---")
            st.markdown("### Perturbation Stability Analysis")
            st.caption("Evaluates classification stability across 4 common transmission transforms (lossy JPEG compression, Gaussian noise, and downscale-upscale resampling).")

            with st.spinner("Simulating perturbation transforms in-memory..."):
                stress_summary: StressTestSummary = run_degradation_stress_test(
                    active_image_rgb,
                    detector=detector,
                    threshold=threshold,
                )

            # Stress test table
            table_rows = []
            for v in stress_summary.variant_results:
                badge = '<span class="editorial-tag authentic">Consistent</span>' if v.verdict_matches_baseline else '<span class="editorial-tag synthetic">Shifted</span>'
                sign = "+" if v.score_delta >= 0 else ""
                table_rows.append({
                    "Perturbation Variant": v.variant_name,
                    "Parameters": v.parameters,
                    "Predicted Label": v.predicted_label,
                    "AI Likelihood": f"{v.ai_probability * 100:.1f}%",
                    "Delta": f"{sign}{v.score_delta * 100:.1f}%",
                    "Stability": badge,
                })

            # Display table using clean editorial styling
            tbody_html = "".join([
                f"<tr>"
                f"<td style='font-weight: 600; color: #0a2346;'>{r['Perturbation Variant']}</td>"
                f"<td style='color: #627d98; font-family: \"JetBrains Mono\", monospace; font-size: 0.76rem;'>{r['Parameters']}</td>"
                f"<td><code>{r['Predicted Label']}</code></td>"
                f"<td style='font-weight: 700; color: #0a2346; font-family: \"JetBrains Mono\", monospace;'>{r['AI Likelihood']}</td>"
                f"<td style='color: #00539b; font-family: \"JetBrains Mono\", monospace; font-weight: 700;'>{r['Delta']}</td>"
                f"<td>{r['Stability']}</td>"
                f"</tr>"
                for r in table_rows
            ])
            table_html = (
                f'<table class="editorial-table">'
                f'<thead><tr>'
                f'<th>Perturbation Variant</th>'
                f'<th>Parameters</th>'
                f'<th>Classification</th>'
                f'<th>AI Likelihood</th>'
                f'<th>Delta</th>'
                f'<th>Stability</th>'
                f'</tr></thead>'
                f'<tbody>{tbody_html}</tbody>'
                f'</table>'
            )
            st.markdown(table_html, unsafe_allow_html=True)

            # Stability summary banner
            is_stable = stress_summary.is_stable
            stat_color = "#15803d" if is_stable else "#be123c"
            stat_bg = "#f0fdf4" if is_stable else "#fff1f2"
            stat_border = "#86efac" if is_stable else "#fecdd3"
            stat_dot = "green" if is_stable else "rose"
            stat_title = "Perturbation Stable Decision" if is_stable else "Classification Shift Under Perturbation"

            st.markdown(
                f"""
                <div style="background: {stat_bg}; border: 1px solid {stat_border}; border-radius: 8px; padding: 16px 22px; display: flex; justify-content: space-between; align-items: center; margin-top: 16px; flex-wrap: wrap; gap: 12px; box-shadow: 0 2px 6px rgba(10, 35, 70, 0.04);">
                    <div>
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="status-dot {stat_dot}"></span>
                            <span style="font-weight: 700; font-size: 0.98rem; color: {stat_color}; font-family: 'Copperplate Gothic', 'Cinzel', serif;">{stat_title}</span>
                        </div>
                        <div style="font-size: 0.86rem; color: #334e68; margin-top: 4px;">
                            Verdict maintained consistency across <strong>{stress_summary.matching_perturbations}/{stress_summary.total_perturbations}</strong> degraded variants ({stress_summary.stability_score_pct:.0f}% Stability).
                        </div>
                    </div>
                    <div style="font-size: 0.78rem; color: #627d98; max-width: 380px; text-align: right;">
                        Measures empirical stability under transmission recompression, not adversarial robustness.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.info("Upload an image or choose a reference benchmark preset to begin forensic analysis.")

    # =========================================================================
    # TAB 2: VIDEO FORENSICS & TEMPORAL FLICKER
    # =========================================================================
    with tab_video:
        st.markdown("### Video Deepfake & Temporal Continuity Analysis")
        st.markdown(
            """
            <div style="font-size: 0.90rem; color: #a1a1aa; margin-bottom: 18px; max-width: 68ch; text-wrap: pretty;">
                Extract frames at 1–2 FPS, evaluate frame-by-frame synthetic artifact likelihoods, and quantify temporal continuity (first-order confidence variance and Laplacian texture stability) to detect deepfake manipulations.
            </div>
            """,
            unsafe_allow_html=True,
        )

        video_mode = st.radio(
            "Video Ingestion Mode",
            ["Live Stream URL (YouTube / Shorts / Reels / TikTok)", "Local Video File Upload", "Benchmark Sample Video"],
            horizontal=True,
            help="Select how you wish to ingest video media. URL mode downloads a lightweight <=720p stream in seconds via yt-dlp.",
        )

        active_video_source = None
        video_filename = "video_input.mp4"
        is_stream = False
        stream_meta = {}

        if video_mode == "Live Stream URL (YouTube / Shorts / Reels / TikTok)":
            url_col, btn_col = st.columns([3, 1])
            with url_col:
                input_url = st.text_input(
                    "Video Stream URL",
                    placeholder="https://www.youtube.com/shorts/... or https://www.instagram.com/reel/...",
                    help="Supports YouTube Videos, YouTube Shorts, Instagram Reels, and TikTok clips under 90s.",
                )
                st.markdown(
                    "<div style='font-size: 0.76rem; color: #627d98; margin-top: -10px; margin-bottom: 10px;'>"
                    "Supports public YouTube Shorts, YouTube videos, Instagram Reels, and TikTok clips (&le;90s). "
                    "If Meta login gates an Instagram Reel, drop the file into <em>Local Video File Upload</em>."
                    "</div>",
                    unsafe_allow_html=True,
                )
            with btn_col:
                st.write("")
                st.write("")
                fetch_btn = st.button("Fetch Stream", type="primary", use_container_width=True)

            if fetch_btn and input_url.strip():
                clean_url = input_url.strip()
                platform = identify_platform(clean_url)
                with st.spinner(f"Connecting to {platform} and extracting stream (≤720p)..."):
                    try:
                        filepath, temp_dir, meta = download_video_from_url(
                            clean_url,
                            max_duration_sec=90,
                            max_height=720,
                        )
                        st.session_state["stream_video_path"] = filepath
                        st.session_state["stream_temp_dir"] = temp_dir
                        st.session_state["stream_meta"] = meta
                        st.success(f"Stream fetched: **{meta['title'][:45]}** ({meta['duration_sec']:.1f}s – {meta['platform']})")
                    except VideoDurationExceededError as e:
                        st.warning(f"Duration limit exceeded: {e}")
                    except VideoDownloadError as e:
                        st.error(f"Access error: {e}")
                        if "instagram" in clean_url.lower():
                            st.info(
                                "💡 **Quick Solution for Instagram Reels**: Meta frequently blocks unauthenticated server requests. "
                                "You can save the Reel to your device (via any standard downloader) and switch to the "
                                "**Local Video File Upload** tab above to analyze it with the full temporal deepfake model, "
                                "or test with a public **YouTube Short**."
                            )
                    except Exception as e:
                        st.error(f"Failed to fetch video stream: {e}")

            if "stream_video_path" in st.session_state and os.path.exists(st.session_state["stream_video_path"]):
                active_video_source = st.session_state["stream_video_path"]
                stream_meta = st.session_state.get("stream_meta", {})
                video_filename = f"{stream_meta.get('platform', 'Stream')}_{stream_meta.get('title', 'video')[:20]}.mp4"
                is_stream = True

        elif video_mode == "Local Video File Upload":
            uploaded_video = st.file_uploader(
                "Upload Video File",
                type=["mp4", "mov", "avi", "webm"],
                help="Recommended: Clips under 45 seconds for swift local inference.",
            )
            if uploaded_video is not None:
                active_video_source = uploaded_video.read()
                video_filename = uploaded_video.name

        else:  # Benchmark Sample Video
            sample_path = "data/benchmarks/sample_test_video.mp4"
            if os.path.exists(sample_path):
                active_video_source = sample_path
                video_filename = "sample_test_video.mp4"
                st.caption(f"Loaded reference benchmark video: `{sample_path}` (45 frames, 30 FPS)")
            else:
                st.warning(f"Benchmark test video not found at: {sample_path}")

        # Controls & Execution
        if active_video_source is not None:
            # Social Media Compression Academic Notice
            if is_stream:
                st.markdown(
                    """
                    <div style="background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 12px 18px; margin: 14px 0; font-size: 0.85rem; color: #fbbf24;">
                        <span class="status-dot amber"></span><strong>Social Media Stream Detected:</strong> Platform re-compression (H.264/VP9 at reduced bitrates) attenuates high-frequency pixel artifacts. The detection pipeline automatically places higher weighting on the <strong>Scene-Cut Aware Temporal Continuity</strong> signal.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.video(active_video_source)

            ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 1])
            with ctrl_col1:
                target_fps = st.slider("Sampling Rate (FPS)", min_value=1.0, max_value=2.0, value=1.5, step=0.5)
            with ctrl_col2:
                max_frames = st.slider("Max Frames Analyzed", min_value=15, max_value=90, value=45, step=15)
            with ctrl_col3:
                st.write("")
                st.write("")
                run_video_btn = st.button("Run Temporal Forensic Analysis", type="primary", use_container_width=True)

            if run_video_btn:
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(val, text):
                    progress_bar.progress(val)
                    status_text.markdown(f"**Status:** {text}")

                try:
                    video_res: VideoAnalysisResult = process_video_pipeline(
                        video_source=active_video_source,
                        detector=detector,
                        video_filename=video_filename,
                        target_fps=target_fps,
                        max_frames=max_frames,
                        threshold=threshold,
                        is_social_media_stream=is_stream,
                        progress_callback=update_progress,
                    )
                    st.session_state["last_video_result"] = video_res
                except Exception as e:
                    st.error(f"Video analysis failed: {e}")

        # Render video results if available in session state
        if "last_video_result" in st.session_state:
            res: VideoAnalysisResult = st.session_state["last_video_result"]
            st.markdown("---")

            # 1. Video Summary Verdict Card (dynamically updated with threshold)
            top_k_score = getattr(res, "top_k_ai_probability", res.mean_ai_probability)
            faces_audited = getattr(res, "faces_detected_count", 0)
            v_is_ai = bool(
                (top_k_score >= threshold) or
                (res.mean_ai_probability >= threshold) or
                (res.ai_frame_percentage >= 25.0) or
                (res.peak_flicker_score >= 0.28 and top_k_score >= 0.38)
            )
            card_class = "verdict-card-synthetic" if v_is_ai else "verdict-card-authentic"
            tag_class = "synthetic" if v_is_ai else "authentic"
            dot_class = "rose" if v_is_ai else "green"
            status_color = "#f43f5e" if v_is_ai else "#10b981"
            v_title = "Synthetic Video Artifacts Detected" if v_is_ai else "Authentic Video Stream Confirmed"

            st.markdown(
                f"""
                <div class="{card_class}">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 20px;">
                        <div style="max-width: 50%;">
                            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                                <span class="status-dot {dot_class}"></span>
                                <span class="editorial-tag {tag_class}">Aggregated Video Verdict</span>
                                {f'<span class="editorial-tag neutral">Facial Audit ({faces_audited} Frames)</span>' if faces_audited > 0 else '<span class="editorial-tag neutral">Scene Audit</span>'}
                            </div>
                            <h2 class="verdict-heading" style="color: {status_color};">{v_title}</h2>
                            <div style="font-size: 0.85rem; color: #a1a1aa; line-height: 1.5;">
                                Source: <strong style="color: #f4f4f6;">{res.video_filename}</strong> ({res.duration_sec:.1f}s at {res.sampling_fps} FPS, {res.total_frames_analyzed} frames) – Latency: {res.processing_time_sec:.2f}s – Threshold: <code>{threshold:.2f}</code>
                            </div>
                            {f'<div style="font-size: 0.78rem; color: #71717a; margin-top: 8px;">{res.scene_cuts_detected} editing jump cuts isolated via histogram correlation and excluded from flicker computation.</div>' if res.scene_cuts_detected > 0 else ''}
                        </div>
                        <div style="display: flex; gap: 24px; text-align: right; flex-wrap: wrap;">
                            <div>
                                <div style="font-size: 0.69rem; font-weight: 500; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Top-K Anomaly</div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; font-size: 1.85rem; font-weight: 600; color: {'#f43f5e' if top_k_score >= threshold else '#10b981'};">{top_k_score * 100:.1f}%</div>
                                <div style="font-size: 0.70rem; color: #71717a;">Top 20% Peak Frames</div>
                            </div>
                            <div>
                                <div style="font-size: 0.69rem; font-weight: 500; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Mean Likelihood</div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; font-size: 1.85rem; font-weight: 600; color: #f4f4f6;">{res.mean_ai_probability * 100:.1f}%</div>
                                <div style="font-size: 0.70rem; color: #71717a;">Overall Average</div>
                            </div>
                            <div>
                                <div style="font-size: 0.69rem; font-weight: 500; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Flagged Frames</div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; font-size: 1.85rem; font-weight: 600; color: {'#f43f5e' if res.ai_frame_percentage > 25 else '#10b981'};">{res.ai_frame_percentage:.1f}%</div>
                                <div style="font-size: 0.70rem; color: #71717a;">Above Threshold</div>
                            </div>
                            <div>
                                <div style="font-size: 0.69rem; font-weight: 500; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Peak Flicker</div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; font-size: 1.85rem; font-weight: 600; color: {'#f43f5e' if res.peak_flicker_score > 0.25 else '#38bdf8'};">{res.peak_flicker_score:.3f}</div>
                                <div style="font-size: 0.70rem; color: #71717a;">Continuity Index</div>
                            </div>
                            <div>
                                <div style="font-size: 0.69rem; font-weight: 500; color: #71717a; text-transform: uppercase; letter-spacing: 0.05em;">Scene Cuts</div>
                                <div style="font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; font-size: 1.85rem; font-weight: 600; color: #a1a1aa;">{res.scene_cuts_detected}</div>
                                <div style="font-size: 0.70rem; color: #71717a;">Transitions Isolated</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # 2. Interactive Plotly Timeline
            st.markdown("### Frame-by-Frame Forensic Timeline")
            st.caption("Inspect temporal AI confidence variance alongside the continuity flicker metric. Shaded intervals designate anomalous frame ranges.")

            plotly_fig = create_plotly_video_timeline(res, threshold=threshold)
            if plotly_fig is not None:
                st.plotly_chart(plotly_fig, use_container_width=True)
            else:
                mpl_fig = create_matplotlib_video_timeline(res, threshold=threshold)
                st.pyplot(mpl_fig)

            # 3. Top Flagged Frames Gallery with Spatial Heatmaps
            st.markdown("### Peak Anomaly Frames")
            st.caption("Frames exhibiting the highest AI confidence or temporal discontinuity, presented with their respective spatial artifact heatmaps.")

            top_cols = st.columns(3)
            for idx, top_frame in enumerate(res.top_flagged_frames):
                with top_cols[idx]:
                    st.markdown(f"<div style='font-size: 0.80rem; font-weight: 600; color: #f4f4f6; margin-bottom: 6px;'>Flagged Frame {idx + 1} (T = {top_frame.timestamp_sec:.2f}s)</div>", unsafe_allow_html=True)
                    if top_frame.heatmap_overlay_rgb is not None:
                        st.image(top_frame.heatmap_overlay_rgb, use_container_width=True)
                    else:
                        st.image(top_frame.frame_rgb, use_container_width=True)

                    st.markdown(
                        f"""
                        <div style="background: #13151a; border: 1px solid rgba(255, 255, 255, 0.07); border-radius: 6px; padding: 12px 14px; margin-top: 8px; font-size: 0.78rem; color: #a1a1aa;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                                <span>AI Probability:</span>
                                <strong style="font-family: 'JetBrains Mono', monospace; color: {'#f43f5e' if top_frame.ai_probability > 0.5 else '#10b981'};">{top_frame.ai_probability * 100:.1f}%</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                                <span>Flicker Metric:</span>
                                <strong style="font-family: 'JetBrains Mono', monospace; color: #38bdf8;">{top_frame.temporal_flicker_score:.3f}</strong>
                            </div>
                            <div style="display: flex; justify-content: space-between;">
                                <span>Laplacian Variance:</span>
                                <strong style="font-family: 'JetBrains Mono', monospace; color: #e4e4e7;">{top_frame.laplacian_variance:.1f}</strong>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
