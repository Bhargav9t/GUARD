"""
Honest UI & Scientific Disclaimers Module for GUARD.
Ensures transparent communication of machine learning limitations:
  1. Confidence scores are uncalibrated softmax probabilities.
  2. Degradation stress test measures heuristic perturbation stability, not adversarial robustness.
  3. Inference engine baseline characteristics and dual-stream support.
  4. Dynamic empirical evaluation on uploaded inputs without fabricated benchmark claims.
"""

import streamlit as st


def render_honest_disclaimers():
    """Renders persistent scientific notice and model limitations in an understated editorial expander."""
    with st.expander("Scientific Notice & Evidentiary Limitations", expanded=False):
        st.markdown(
            r"""
            <div style="font-size: 0.88rem; color: #334e68; line-height: 1.65; padding: 4px 0; font-family: 'Plus Jakarta Sans', sans-serif;">
                <p style="margin-bottom: 12px;">
                    <strong style="color: #0a2346; font-weight: 700; font-family: 'Copperplate Gothic', 'Cinzel', serif;">Uncalibrated Softmax Probabilities:</strong>
                    The confidence figures reported by this prototype represent raw softmax output distributions ($p \in [0, 1]$). In deep neural architectures, softmax outputs are frequently overconfident under domain shift and do not correspond directly to empirical Bayes error rates or legal evidentiary thresholds.
                </p>
                <p style="margin-bottom: 12px;">
                    <strong style="color: #0a2346; font-weight: 700; font-family: 'Copperplate Gothic', 'Cinzel', serif;">Perturbation Stability vs. Adversarial Robustness:</strong>
                    The Degradation Stress Test evaluates heuristic resilience against common transmission transforms—specifically lossy JPEG recompression and downscale-upscale resampling. This should not be conflated with formal mathematical robustness against gradient-based adversarial perturbations (such as FGSM or PGD).
                </p>
                <p style="margin-bottom: 12px;">
                    <strong style="color: #0a2346; font-weight: 700; font-family: 'Copperplate Gothic', 'Cinzel', serif;">Inference Architecture:</strong>
                    The active engine utilizes a multi-evidence consensus model combining vision transformer representations (<code>Organika/sdxl-detector</code> and <code>umm-maybe/AI-image-detector</code>) with high-frequency Fourier spectral physics and sensor noise analysis. A custom dual-stream architecture (Spatial EfficientNet + in-graph 2D FFT Frequency Residuals) is accessible via the <code>BaseDetector</code> adapter.
                </p>
                <p style="margin-bottom: 0;">
                    <strong style="color: #0a2346; font-weight: 700; font-family: 'Copperplate Gothic', 'Cinzel', serif;">Verification Policy:</strong>
                    All metrics displayed across this interface are computed dynamically on your uploaded inputs. No synthetic or unverified benchmark figures are claimed.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_engine_badge(model_name: str, device: str, dtype: str):
    """Renders understated technical telemetry ribbon with Signal Blue and Porcelain contrast."""
    is_hf = "sdxl" in model_name.lower() or "umm-maybe" in model_name.lower()
    engine_type_text = "Baseline Transformer" if is_hf else "Multi-Evidence Consensus"
    tag_style = "amber" if is_hf else "authentic"

    st.markdown(
        f"""
        <div class="meta-ribbon">
            <div class="meta-ribbon-item">
                <span class="editorial-tag {tag_style}">{engine_type_text}</span>
            </div>
            <div class="meta-ribbon-item">
                <span class="meta-ribbon-label">Model:</span>
                <span class="meta-ribbon-value" style="color: #00539b;">{model_name}</span>
            </div>
            <div class="meta-ribbon-item">
                <span class="meta-ribbon-label">Hardware:</span>
                <span class="meta-ribbon-value" style="color: #0a2346;">{device}</span>
            </div>
            <div class="meta-ribbon-item">
                <span class="meta-ribbon-label">Precision:</span>
                <span class="meta-ribbon-value" style="color: #0a2346;">{dtype}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
