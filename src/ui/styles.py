# ==============================================================================
# File: src/ui/styles.py
# Description: Uiverse.io-Inspired Corporate Minimalist Styling & CSS System.
#
# Design Aesthetics:
#   - Canvas: Soft clean slate surface (#f8fafc)
#   - Primary Typography: Deep navy (#0f172a) - strictly avoiding white-on-white text
#   - Secondary Typography: Muted slate (#475569) for high legibility
#   - Tactile Cards: Crisp white (#ffffff) with subtle multi-layer inset highlights
#     and soft depth elevation inspired by Uiverse.io
#   - Micro-Interactions: Smooth hover transitions, interactive elevation
#   - Status Badges: High-visibility Emerald Authentic & Crimson Fake badges
# ==============================================================================

# Import Streamlit for custom stylesheet injection
import streamlit as st


def inject_custom_css() -> None:
    """
    Injects high-grade Uiverse.io-inspired corporate minimalist CSS styling into Streamlit.
    Strictly enforces high-contrast dark typography on white surfaces to eliminate
    invisible white-on-white text bugs.
    """
    custom_css = """
    <style>
    /* Google Fonts: Inter (Body), Plus Jakarta Sans (Headings), JetBrains Mono (Code/Metrics) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Slate Surface Canvas */
    .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Headings Styling - Strictly Deep Navy (#0f172a) */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
    }

    /* Ensure Streamlit Markdown, Paragraphs, and Labels Never Render White Text */
    .stMarkdown p, .stMarkdown span, .stMarkdown label, div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
    }

    /* Primary Corporate Header Banner */
    .corporate-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 2px 4px -2px rgba(15, 23, 42, 0.02);
        display: flex;
        flex-direction: column;
        gap: 6px;
        position: relative;
        overflow: hidden;
    }

    .corporate-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0ea5e9, #3b82f6, #6366f1);
    }

    .corporate-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.75rem;
        font-weight: 800;
        color: #0f172a !important;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .corporate-subtitle {
        font-size: 0.92rem;
        color: #64748b !important;
        margin: 0;
    }

    /* Uiverse.io-Inspired Tactile Card Containers */
    .uiverse-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 2px 4px -2px rgba(15, 23, 42, 0.02);
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.25s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.25s ease;
    }

    .uiverse-card:hover {
        transform: translateY(-2px);
        border-color: #cbd5e1;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.07), 0 4px 6px -4px rgba(15, 23, 42, 0.04);
    }

    /* Streamlit Native Container Border Wrapper Styling */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        padding: 20px 24px !important;
        box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 2px 4px -2px rgba(15, 23, 42, 0.02) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div:hover {
        border-color: #cbd5e1 !important;
        box-shadow: 0 10px 15px -3px rgba(15, 23, 42, 0.06), 0 4px 6px -4px rgba(15, 23, 42, 0.03) !important;
    }

    /* Section Card Header Title */
    .card-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.90rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #475569 !important;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Uiverse.io-Inspired Verdict Cards */
    .verdict-card-real {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 14px;
        padding: 24px 28px;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.08);
    }

    .verdict-badge-real {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #16a34a;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.30rem;
        font-weight: 800;
        padding: 6px 18px;
        border-radius: 9999px;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 10px rgba(22, 163, 74, 0.25);
    }

    .verdict-card-fake {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fca5a5;
        border-radius: 14px;
        padding: 24px 28px;
        margin: 16px 0;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.08);
    }

    .verdict-badge-fake {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #dc2626;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.30rem;
        font-weight: 800;
        padding: 6px 18px;
        border-radius: 9999px;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 10px rgba(220, 38, 38, 0.25);
    }

    /* Checkpoint Status Indicator Pills */
    .status-pill-calibrated {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #dcfce7;
        color: #15803d !important;
        border: 1px solid #86efac;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .status-pill-untrained {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #fef3c7;
        color: #b45309 !important;
        border: 1px solid #fcd34d;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
    }

    /* Stream Metric Sub-Cards */
    .stream-block {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-top: 8px;
    }

    .stream-metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        font-size: 0.85rem;
        border-bottom: 1px dashed #e2e8f0;
    }

    .stream-metric-row:last-child {
        border-bottom: none;
    }

    .stream-metric-label {
        color: #64748b !important;
        font-weight: 500;
    }

    .stream-metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #0f172a !important;
    }

    /* Streamlit Metrics Corporate Styling */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.03) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 1.65rem !important;
    }

    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }

    /* Sidebar Corporate Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 0.82rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #475569 !important;
        margin-top: 16px !important;
        margin-bottom: 8px !important;
    }

    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
        color: #0f172a !important;
    }

    /* Clean Corporate File Uploader Container */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 18px;
    }

    /* Radio Button Text High-Contrast Overrides */
    div[data-testid="stRadio"] label, div[data-testid="stRadio"] span {
        color: #0f172a !important;
        font-weight: 500 !important;
    }

    /* Tabs Styling - Enterprise Minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 38px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b !important;
        padding: 0 14px;
        border-radius: 6px 6px 0 0;
    }

    .stTabs [aria-selected="true"] {
        color: #0f172a !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #0f172a !important;
    }

    /* Progress Bar Corporate Accent */
    .stProgress > div > div > div > div {
        background-color: #0f172a !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
