"""FlashStudio — Custom CSS Styles matching the professional mockup (light theme)."""

import streamlit as st


def inject_custom_css():
    """Inject custom CSS to match the professional mockup design — light theme, dark sidebar."""
    st.markdown("""
    <style>
    /* ═══════ Global ═══════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ═══════ Sidebar ═══════ */
    section[data-testid="stSidebar"] {
        background: #FAFBFC !important;
        border-right: 1px solid #E8E8EF;
        width: 240px !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 240px !important;
        padding: 1rem 0.8rem !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #E8E8EF !important;
        margin: 0.4rem 0 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        font-size: 0.82rem !important;
        padding: 0.4rem 0.7rem !important;
        border-radius: 6px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        min-height: 0 !important;
        height: auto !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        color: #374151 !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: #F5F3FF !important;
        color: #7C3AED !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #F5F3FF !important;
        border: 1px solid #EDE9FE !important;
        color: #7C3AED !important;
        font-weight: 600 !important;
    }

    /* Hide Streamlit's auto multipage nav */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* ═══════ Main Content ═══════ */
    .stApp > header {
        background: transparent;
    }

    /* ═══════ Metric Cards ═══════ */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8E8EF;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s, box-shadow 0.2s;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.08);
    }

    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6B7280 !important;
        font-weight: 500;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700;
        color: #1A1A2E !important;
    }

    /* ═══════ Primary Buttons ═══════ */
    .main .stButton > button[kind="primary"] {
        background: #7C3AED !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        color: #FFFFFF !important;
        letter-spacing: 0.02em;
        transition: all 0.2s;
    }

    .main .stButton > button[kind="primary"]:hover {
        background: #6D28D9 !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
        transform: translateY(-1px);
    }

    .main .stButton > button[kind="secondary"] {
        border-radius: 8px;
        border: 1px solid #E8E8EF !important;
        font-weight: 500;
        color: #1A1A2E !important;
    }

    .main .stButton > button[kind="secondary"]:hover {
        border-color: #7C3AED !important;
        color: #7C3AED !important;
    }

    /* ═══════ Containers / Cards ═══════ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        border-color: #E8E8EF !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    /* ═══════ Step Indicator ═══════ */
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 0.3rem;
        padding: 0.5rem 0.5rem;
        margin-bottom: 1rem;
        background: #FAFAFA;
        border-radius: 8px;
        border: 1px solid #F0F0F5;
    }

    .step-item {
        text-align: center;
        flex: 1;
        position: relative;
    }

    .step-icon { font-size: 1.3rem; display: block; }
    .step-label { font-size: 0.7rem; margin-top: 0.2rem; font-weight: 500; color: #6B7280; }
    .step-bar { height: 3px; border-radius: 2px; margin-top: 0.4rem; }

    .step-active .step-label { color: #7C3AED; font-weight: 700; }
    .step-active .step-bar { background: #7C3AED; }
    .step-done .step-bar { background: #10B981; }
    .step-pending .step-bar { background: #E5E7EB; }
    .step-pending .step-label { opacity: 0.5; }

    /* ═══════ Progress Bar ═══════ */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7C3AED, #A78BFA) !important;
        border-radius: 4px;
    }

    /* ═══════ Tabs ═══════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #F0F0F5;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        color: #6B7280;
    }

    .stTabs [aria-selected="true"] {
        color: #7C3AED !important;
        font-weight: 600;
    }

    /* ═══════ Info Bar (bottom) ═══════ */
    .info-bar {
        background: #F5F3FF;
        border: 1px solid #EDE9FE;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        text-align: center;
        font-size: 0.8rem;
        margin-top: 1.5rem;
        color: #4C1D95;
    }

    .info-bar b {
        color: #1A1A2E;
    }

    /* ═══════ File Uploader ═══════ */
    [data-testid="stFileUploader"] section {
        border-radius: 12px;
        border: 2px dashed #D4D4D8 !important;
        transition: border-color 0.2s;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #7C3AED !important;
    }

    /* ═══════ Sliders ═══════ */
    .stSlider > div > div > div > div {
        background: #7C3AED !important;
    }

    /* ═══════ Selectbox ═══════ */
    .main .stSelectbox > div > div {
        border-radius: 8px;
        border-color: #E8E8EF;
    }

    /* ═══════ Dataframe ═══════ */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #E8E8EF;
    }

    /* ═══════ Expander ═══════ */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1A1A2E;
    }

    /* ═══════ Download Buttons ═══════ */
    .stDownloadButton > button[kind="primary"] {
        background: #7C3AED !important;
        color: white !important;
        border-radius: 8px;
    }

    .stDownloadButton > button {
        border-radius: 8px;
        border: 1px solid #E8E8EF;
    }
    </style>
    """, unsafe_allow_html=True)


def render_info_bar(items: dict):
    """Render a professional info bar at the bottom (like mockup)."""
    inner = "  •  ".join(f"{k}: <b>{v}</b>" for k, v in items.items())
    st.markdown(
        f'<div class="info-bar">ℹ️ {inner}</div>',
        unsafe_allow_html=True,
    )


def render_page_header(icon: str, title: str, subtitle: str):
    """Render a professional page header with icon and subtitle."""
    st.markdown(
        f"""<div style="margin-bottom: 1.5rem;">
            <h1 style="margin:0; display:flex; align-items:center; gap:0.6rem;
                font-size:1.8rem; font-weight:700; color:#1A1A2E;">
                <span style="font-size:2rem;">{icon}</span> {title}
            </h1>
            <p style="margin:0.4rem 0 0; color:#6B7280; font-size:0.95rem;">{subtitle}</p>
        </div>""",
        unsafe_allow_html=True,
    )
