"""
Interactive Timeline Visualization Module for GUARD Video Forensics.
Renders interactive Plotly timeline plots tracking per-frame AI confidence
and temporal flicker metrics, marking anomalous frame ranges.
Calibrated for Signal Blue & Porcelain editorial theme with Copperplate typography.
"""

from typing import Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.video.processor import VideoAnalysisResult


def create_plotly_video_timeline(analysis: VideoAnalysisResult, threshold: float = 0.50) -> Any:
    """
    Constructs an interactive Plotly timeline figure for video forensics with Signal Blue & Porcelain styling.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return None

    timestamps = [f.timestamp_sec for f in analysis.frames]
    ai_probs = [f.ai_probability * 100.0 for f in analysis.frames]
    flicker_scores = [f.temporal_flicker_score for f in analysis.frames]

    # Create subplots: Top = AI Probability, Bottom = Temporal Flicker
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        subplot_titles=(
            "FRAME-BY-FRAME AI CONFIDENCE & ANOMALY DETECTION",
            "TEMPORAL FLICKER INDEX (|Δ CONFIDENCE| + LAPLACIAN DISCONTINUITY)"
        ),
        row_heights=[0.60, 0.40],
    )

    # 1. AI Probability Trace in Signal Blue
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=ai_probs,
            mode="lines+markers",
            name="AI Probability (%)",
            line=dict(color="#00539b", width=2.5),
            marker=dict(
                size=6,
                color=["#e11d48" if p >= threshold * 100 else "#16a34a" for p in ai_probs],
                symbol="circle",
                line=dict(color="#ffffff", width=1),
            ),
            hovertemplate="<b>Timestamp:</b> %{x:.2f}s<br><b>AI Probability:</b> %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Threshold horizontal reference line
    fig.add_hline(
        y=threshold * 100.0,
        line_dash="dash",
        line_color="#00539b",
        annotation_text=f"Threshold ({threshold*100:.0f}%)",
        annotation_position="bottom right",
        annotation_font=dict(color="#00539b", family="Copperplate Gothic, Cinzel, serif", size=11),
        row=1,
        col=1,
    )

    # 2. Temporal Flicker Trace in Soft Signal Blue
    fig.add_trace(
        go.Bar(
            x=timestamps,
            y=flicker_scores,
            name="Temporal Flicker Metric",
            marker=dict(
                color=["#e11d48" if fs > 0.25 else "#00539b" for fs in flicker_scores],
                line=dict(color="#ffffff", width=0.5),
            ),
            hovertemplate="<b>Timestamp:</b> %{x:.2f}s<br><b>Flicker Metric:</b> %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Highlight anomalous ranges with semi-transparent shading
    for start_t, end_t in analysis.anomalous_timestamp_ranges:
        x_end = max(end_t, start_t + 0.5)
        fig.add_vrect(
            x0=start_t,
            x1=x_end,
            fillcolor="#e11d48",
            opacity=0.12,
            layer="below",
            line_width=0,
            row="all",
            col=1,
        )

    # Signal Blue & Porcelain layout styling
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family="Copperplate Gothic, Cinzel, serif", color="#0a2346"),
        height=500,
        margin=dict(l=48, r=32, t=50, b=36),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1,
            font=dict(family="Copperplate Gothic, Cinzel, serif", size=11),
        ),
        hovermode="x unified",
    )

    fig.update_xaxes(gridcolor="rgba(0, 83, 155, 0.08)", linecolor="rgba(0, 83, 155, 0.15)")
    fig.update_yaxes(gridcolor="rgba(0, 83, 155, 0.08)", linecolor="rgba(0, 83, 155, 0.15)")
    fig.update_yaxes(title_text="AI Confidence (%)", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Flicker Index", row=2, col=1)
    fig.update_xaxes(title_text="Timeline (seconds)", row=2, col=1)

    return fig


def create_matplotlib_video_timeline(analysis: VideoAnalysisResult, threshold: float = 0.50) -> plt.Figure:
    """
    Constructs a Matplotlib figure fallback for environments where Plotly is absent.
    """
    timestamps = [f.timestamp_sec for f in analysis.frames]
    ai_probs = [f.ai_probability * 100.0 for f in analysis.frames]
    flicker_scores = [f.temporal_flicker_score for f in analysis.frames]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#ffffff")
    ax1.set_facecolor("#f8fafc")
    ax2.set_facecolor("#f8fafc")

    ax1.plot(timestamps, ai_probs, color="#00539b", linewidth=2, marker="o", markersize=4, label="AI Probability (%)")
    ax1.axhline(threshold * 100.0, color="#d97706", linestyle="--", label=f"Threshold ({threshold*100:.0f}%)")
    ax1.set_ylabel("AI Probability (%)", color="#0a2346")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    ax1.legend(loc="upper right")

    ax2.bar(timestamps, flicker_scores, color="#00539b", width=0.4, label="Temporal Flicker")
    ax2.set_ylabel("Flicker", color="#0a2346")
    ax2.set_xlabel("Timeline (seconds)", color="#0a2346")
    ax2.grid(True, linestyle=":", alpha=0.3, color="#94a3b8")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    return fig
