# ==============================================================================
# File: src/ui/styles.py
# Description: Bespoke Signal Blue & Porcelain Editorial Design System with
#              Copperplate Typography for the GUARD Forensic Platform.
#
# Color Architecture:
#   - Canvas: Porcelain (#F4F6F9) / Pure Glazed Porcelain Surfaces (#FFFFFF)
#   - Primary Signature: Signal Blue (#00539B / RAL 5005)
#   - Deep Structure / Sidebar: Deep Signal Blue (#091C36)
#   - Foreground Text: Deep Signal Ink (#0A2346) & Slate (#486581)
#   - Typography: Copperplate Gothic / Cinzel Display + Plus Jakarta Sans + JetBrains Mono
# ==============================================================================

import streamlit as st


def inject_custom_css() -> None:
    """
    Injects high-fidelity Signal Blue & Porcelain styling with Copperplate typography:
    Classic Copperplate Gothic display headers, sharp ceramic porcelain surfaces,
    vibrant Signal Blue buttons and indicators, and restrained editorial hierarchy.
    """
    custom_css = """
    <style>
    /* Google Fonts: Cinzel (Classical Copperplate equivalent webfont) + Plus Jakarta Sans + JetBrains Mono */
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    /* ==========================================================================
       1. FLUSH TOP COMMAND BAR & PORCELAIN CONTAINER MEASURES
       ========================================================================== */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
        height: 1.25rem !important;
        min-height: 0 !important;
        padding: 0 !important;
        z-index: 99 !important;
    }

    header[data-testid="stHeader"] > div:first-child {
        height: 1.25rem !important;
    }

    .block-container,
    [data-testid="stMainBlockContainer"],
    div[data-testid="stAppViewBlockContainer"] > div.block-container,
    div[data-testid="stAppViewBlockContainer"] {
        padding-top: 0.5rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 1320px !important;
        margin: 0 auto !important;
    }

    /* Sidebar Flush Alignment */
    [data-testid="stSidebarHeader"],
    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        line-height: 0 !important;
        overflow: visible !important;
    }

    [data-testid="stLogoSpacer"],
    section[data-testid="stSidebar"] [data-testid="stLogoSpacer"] {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        position: absolute !important;
        top: 0.6rem !important;
        right: 0.6rem !important;
        z-index: 100 !important;
        color: #f7f9fc !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 0.75rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div:first-child,
    section[data-testid="stSidebar"] div.block-container,
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* ==========================================================================
       2. GLOBAL CANVAS: PORCELAIN CERAMIC (#F4F6F9) & COPPERPLATE TYPOGRAPHY
       ========================================================================== */
    html, body, .stApp, [data-testid="stAppViewContainer"], .main, [data-testid="stMain"], [data-testid="stAppViewBlockContainer"] {
        background-color: #f4f6f9 !important;
        background-image: none !important;
        color: #0a2346 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        font-feature-settings: "cv02", "cv03", "cv04", "cv11", "ss01", "kern", "liga" 1;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }

    /* Copperplate Typography: Native Copperplate Gothic with Cinzel Google Font fallback */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Copperplate Gothic', 'Copperplate Gothic Bold', 'Copperplate Gothic Light', 'Copperplate', 'Cinzel', Georgia, serif !important;
        font-weight: 700 !important;
        color: #0a2346 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        font-variant: small-caps !important;
        text-wrap: balance !important;
        line-height: 1.25 !important;
    }

    h1 {
        font-size: 2.15rem !important;
        color: #00539b !important; /* Signal Blue signature headline */
    }

    h2 {
        font-size: 1.65rem !important;
        color: #0a2346 !important;
    }

    h3 {
        font-size: 1.15rem !important;
        color: #0a2346 !important;
        letter-spacing: 0.06em !important;
    }

    /* Reading Rhythm & Measure */
    p, .stMarkdown p, div[data-testid="stMarkdownContainer"] p {
        color: #334e68 !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        font-size: 0.935rem;
        line-height: 1.65;
        letter-spacing: normal;
        text-wrap: pretty;
        max-width: 68ch;
    }

    /* Strict Monospace Restriction: code, tabular data, hashes only */
    code, pre, .font-mono, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace !important;
        font-variant-numeric: tabular-nums !important;
        font-feature-settings: "tnum" 1 !important;
    }

    code {
        background: #e8eef5 !important;
        color: #00539b !important;
        border: 1px solid #c9d8e8 !important;
        padding: 0.15em 0.45em !important;
        border-radius: 4px !important;
        font-size: 0.85em !important;
        font-weight: 600 !important;
    }

    /* ==========================================================================
       3. EDITORIAL HERO HEADER & EYEBROW IN COPPERPLATE & SIGNAL BLUE
       ========================================================================== */
    .hero-header {
        background: #ffffff !important;
        border: 1px solid #d9e2ec !important;
        border-top: 4px solid #00539b !important; /* Signal Blue top accent */
        border-radius: 8px !important;
        padding: 24px 30px 26px 30px;
        margin-top: 0 !important;
        margin-bottom: 22px;
        box-shadow: 0 2px 8px rgba(10, 35, 70, 0.05) !important;
        position: relative;
    }

    .hero-eyebrow {
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif !important;
        font-size: 0.74rem;
        font-weight: 700;
        color: #00539b !important;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        font-family: 'Copperplate Gothic', 'Copperplate Gothic Bold', 'Copperplate', 'Cinzel', serif !important;
        font-size: 2.15rem;
        font-weight: 700;
        color: #0a2346 !important;
        margin: 4px 0 10px 0;
        letter-spacing: 0.04em;
        line-height: 1.2;
        text-transform: uppercase;
        font-variant: small-caps;
        text-wrap: balance;
    }

    .hero-title em {
        font-style: normal;
        color: #00539b;
        border-bottom: 2px solid rgba(0, 83, 155, 0.35);
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #486581 !important;
        margin: 0;
        line-height: 1.65;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        max-width: 65ch;
        text-wrap: pretty;
    }

    /* Metadata Ribbon */
    .meta-ribbon {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 6px;
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 10px 18px;
        margin-bottom: 22px;
        font-size: 0.82rem;
        color: #486581;
        flex-wrap: wrap;
        box-shadow: 0 1px 3px rgba(10, 35, 70, 0.04);
    }

    .meta-ribbon-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .meta-ribbon-label {
        color: #627d98;
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        font-size: 0.70rem;
    }

    .meta-ribbon-value {
        color: #0a2346;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ==========================================================================
       4. GLAZED PORCELAIN CARDS WITH SIGNAL BLUE ACCENTS
       ========================================================================== */
    .spotlight-card {
        background: #ffffff;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 22px 26px;
        margin-bottom: 20px;
        box-shadow: 0 2px 6px rgba(10, 35, 70, 0.04);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .spotlight-card:hover {
        border-color: #00539b;
        box-shadow: 0 4px 14px rgba(0, 83, 155, 0.12);
    }

    /* Streamlit Native Border Block Wrappers */
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: #ffffff !important;
        border: 1px solid #d9e2ec !important;
        border-radius: 8px !important;
        padding: 20px 24px !important;
        box-shadow: 0 2px 6px rgba(10, 35, 70, 0.04) !important;
    }

    /* ==========================================================================
       5. VERDICT CARDS: SIGNAL BLUE & PORCELAIN CALIBRATION
       ========================================================================== */
    .verdict-card-authentic {
        background: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        border-left: 4px solid #16a34a !important;
        border-radius: 8px !important;
        padding: 24px 30px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 3px 12px rgba(22, 163, 74, 0.08) !important;
    }

    .verdict-card-synthetic {
        background: #fff1f2 !important;
        border: 1px solid #fecdd3 !important;
        border-left: 4px solid #e11d48 !important;
        border-radius: 8px !important;
        padding: 24px 30px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 3px 12px rgba(225, 29, 72, 0.08) !important;
    }

    .verdict-eyebrow {
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .verdict-heading {
        font-family: 'Copperplate Gothic', 'Copperplate Gothic Bold', 'Copperplate', 'Cinzel', serif !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.03em !important;
        margin: 4px 0 8px 0 !important;
        line-height: 1.25 !important;
        text-transform: uppercase !important;
        font-variant: small-caps !important;
    }

    /* Understated Status Indicators */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        vertical-align: middle;
    }

    .status-dot.emerald, .status-dot.green {
        background-color: #16a34a;
        box-shadow: 0 0 6px rgba(22, 163, 74, 0.4);
    }

    .status-dot.crimson, .status-dot.rose {
        background-color: #e11d48;
        box-shadow: 0 0 6px rgba(225, 29, 72, 0.4);
    }

    .status-dot.amber {
        background-color: #d97706;
        box-shadow: 0 0 6px rgba(217, 119, 6, 0.4);
    }

    /* Minimal Porcelain & Signal Blue Tags */
    .editorial-tag {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 4px;
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .editorial-tag.authentic {
        background: #dcfce7;
        color: #15803d !important;
        border: 1px solid #86efac;
    }

    .editorial-tag.synthetic {
        background: #ffe4e6;
        color: #be123c !important;
        border: 1px solid #fda4af;
    }

    .editorial-tag.neutral {
        background: #e8eef5;
        color: #00539b !important;
        border: 1px solid #c9d8e8;
    }

    .editorial-tag.amber {
        background: #fef3c7;
        color: #b45309 !important;
        border: 1px solid #fde68a;
    }

    /* Legacy compatibility tag */
    .cyber-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 10px;
        border-radius: 4px;
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif;
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid #c9d8e8;
        background: #e8eef5;
        color: #00539b;
    }
    .cyber-badge.authentic { background: #dcfce7; color: #15803d !important; border-color: #86efac; }
    .cyber-badge.synthetic { background: #ffe4e6; color: #be123c !important; border-color: #fda4af; }
    .cyber-badge.amber { background: #fef3c7; color: #b45309 !important; border-color: #fde68a; }
    .cyber-badge.neutral { background: #e8eef5; color: #00539b !important; border-color: #c9d8e8; }

    /* ==========================================================================
       6. FORENSIC METRIC TILES: PORCELAIN & SIGNAL BLUE
       ========================================================================== */
    .metric-tile {
        background: #ffffff !important;
        border: 1px solid #d9e2ec !important;
        border-radius: 8px !important;
        padding: 18px 20px !important;
        text-align: left !important;
        box-shadow: 0 2px 6px rgba(10, 35, 70, 0.04) !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }

    .metric-tile:hover {
        border-color: #00539b !important;
        box-shadow: 0 4px 12px rgba(0, 83, 155, 0.1) !important;
    }

    .metric-tile-label {
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif;
        font-size: 0.70rem;
        font-weight: 700;
        color: #00539b !important; /* Signal Blue label */
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .metric-tile-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
        font-size: 1.65rem;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: -0.02em;
    }

    .metric-tile-caption {
        font-size: 0.76rem;
        color: #627d98 !important;
        margin-top: 6px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 500;
    }

    /* Streamlit Native Metric Overrides */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #d9e2ec !important;
        border-radius: 8px !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 6px rgba(10, 35, 70, 0.04) !important;
    }

    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
        color: #00539b !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
        letter-spacing: -0.02em !important;
    }

    [data-testid="stMetricLabel"] {
        color: #0a2346 !important;
        font-size: 0.74rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif !important;
    }

    /* ==========================================================================
       7. SIDEBAR: DEEP SIGNAL BLUE (#091C36) & CRISP PORCELAIN
       ========================================================================== */
    section[data-testid="stSidebar"] {
        background-color: #091c36 !important;
        border-right: 1px solid #061324 !important;
        box-shadow: 3px 0 16px rgba(9, 28, 54, 0.25) !important;
    }

    section[data-testid="stSidebar"] h3 {
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.09em !important;
        color: #82b9e9 !important; /* Soft Signal Blue */
        margin-top: 24px !important;
        margin-bottom: 10px !important;
        border-bottom: 1px solid rgba(130, 185, 233, 0.2) !important;
        padding-bottom: 6px;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #d9e2ec !important; /* Crisp Porcelain text */
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: #102a4e !important;
        color: #ffffff !important;
        border: 1px solid #244b7a !important;
    }

    /* ==========================================================================
       8. EDITORIAL SEGMENTED TABS: SIGNAL BLUE & PORCELAIN
       ========================================================================== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #e2e8f0;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 4px;
        box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .stTabs [data-baseweb="tab"] {
        height: 38px;
        font-size: 0.84rem;
        font-weight: 700;
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif !important;
        color: #475569 !important;
        padding: 0 18px;
        border-radius: 6px;
        border: none !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        transition: color 0.15s ease, background 0.15s ease;
    }

    .stTabs [aria-selected="true"] {
        background: #00539b !important; /* Signal Blue Active Tab */
        color: #ffffff !important;      /* Porcelain Text */
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 2px 6px rgba(0, 83, 155, 0.25);
    }

    /* ==========================================================================
       9. BESPOKE BUTTONS: SIGNAL BLUE (#00539B) ON PORCELAIN
       ========================================================================== */
    .stButton > button {
        background: #00539b !important;
        color: #ffffff !important;
        border: 1px solid #004580 !important;
        border-radius: 6px !important;
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif !important;
        font-weight: 700 !important;
        font-size: 0.86rem !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 8px 22px !important;
        box-shadow: 0 2px 6px rgba(0, 83, 155, 0.2) !important;
        transition: background 0.15s ease, transform 0.1s ease !important;
    }

    .stButton > button:hover {
        background: #0066bd !important;
        border-color: #00539b !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 83, 155, 0.3) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: #ffffff !important;
        color: #00539b !important;
        border: 1px solid #c9d8e8 !important;
        box-shadow: 0 1px 3px rgba(10, 35, 70, 0.05) !important;
    }

    .stButton > button[kind="secondary"]:hover {
        background: #f0f4f9 !important;
        border-color: #00539b !important;
    }

    /* ==========================================================================
       10. FORM CONTROLS & PORCELAIN INPUTS
       ========================================================================== */
    .stTextInput > div > div > input, 
    .stSelectbox > div > div,
    div[data-baseweb="select"] > div {
        background: #ffffff !important;
        color: #0a2346 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #00539b !important;
        box-shadow: 0 0 0 2px rgba(0, 83, 155, 0.15) !important;
    }

    /* File Uploader Dropzone */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #94a3b8;
        border-radius: 8px;
        padding: 22px;
        box-shadow: 0 1px 4px rgba(10, 35, 70, 0.03);
        transition: border-color 0.2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #00539b;
    }

    /* Radio Inputs */
    div[data-testid="stRadio"] label, div[data-testid="stRadio"] span {
        color: #334e68 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }

    /* Expander Box */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border: 1px solid #d9e2ec !important;
        border-radius: 6px !important;
        color: #0a2346 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        box-shadow: 0 1px 3px rgba(10, 35, 70, 0.04) !important;
    }

    .streamlit-expanderContent {
        background: #f8fafc !important;
        border: 1px solid #d9e2ec !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
        box-shadow: none !important;
    }

    /* Progress Bar in Signal Blue */
    .stProgress > div > div > div > div {
        background: #00539b !important;
        border-radius: 4px !important;
        border: none !important;
    }

    /* Tables: Editorial Signal Blue Header on Porcelain Grid */
    .editorial-table, .cyber-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 16px 0;
        font-size: 0.85rem;
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(10, 35, 70, 0.04);
    }

    .editorial-table th, .cyber-table th {
        background: #00539b !important;
        color: #ffffff !important;
        font-family: 'Copperplate Gothic', 'Copperplate', 'Cinzel', serif;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        padding: 12px 16px;
        border-bottom: 1px solid #004580;
        text-align: left;
    }

    .editorial-table td, .cyber-table td {
        padding: 11px 16px;
        border-bottom: 1px solid #e2e8f0;
        color: #1e293b;
        background: #ffffff;
    }

    .editorial-table tr:last-child td, .cyber-table tr:last-child td {
        border-bottom: none;
    }

    .editorial-table tr:hover td, .cyber-table tr:hover td {
        background: #f1f5f9;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
