"""FlashStudio — Professional CSS (readable, clean layout)."""

import streamlit as st


def inject_custom_css():
    """Professional-grade CSS — clean, readable, well-spaced."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ═══════ GLOBAL — NO SCROLL ═══════ */
    .stApp {
        font-family: 'Inter', -apple-system, sans-serif;
        overflow: hidden !important;
        height: 100vh !important;
    }

    .main {
        overflow: hidden !important;
        height: 100vh !important;
    }

    .main .block-container {
        padding: 0.8rem 1.5rem 0.5rem !important;
        max-width: 1500px;
        height: calc(100vh - 2rem) !important;
        overflow: hidden !important;
    }

    /* Allow internal scroll on specific containers */
    .scrollable-panel {
        overflow-y: auto !important;
        scrollbar-width: thin;
        scrollbar-color: #D1D5DB transparent;
    }
    .scrollable-panel::-webkit-scrollbar { width: 4px; }
    .scrollable-panel::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 2px; }

    .main .stMarkdown { margin-bottom: 0 !important; }
    .main .element-container { margin-bottom: 0.15rem !important; }
    .main div[data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
    .main hr { margin: 0.3rem 0 !important; border-color: #E5E7EB !important; }

    /* ═══════ SIDEBAR ═══════ */
    section[data-testid="stSidebar"] {
        background: #FAFBFC !important;
        border-right: 1px solid #E5E7EB;
        width: 210px !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 210px !important;
        padding: 0.5rem 0.5rem !important;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #E5E7EB !important;
        margin: 0.4rem 0 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        font-size: 0.82rem !important;
        padding: 0.38rem 0.7rem !important;
        border-radius: 6px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        min-height: 0 !important;
        height: auto !important;
        transition: all 0.15s ease;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: none !important;
        color: #4B5563 !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        background: #EEF2FF !important;
        color: #7C3AED !important;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #EEF2FF !important;
        border: none !important;
        color: #7C3AED !important;
        font-weight: 600 !important;
        border-left: 3px solid #7C3AED !important;
        border-radius: 0 6px 6px 0 !important;
    }

    [data-testid="stSidebarNav"] { display: none !important; }

    /* ═══════ METRIC CARDS ═══════ */
    div[data-testid="stMetric"] {
        background: #FFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 0.6rem 0.8rem !important;
    }

    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #6B7280 !important;
        font-weight: 600;
        margin-bottom: 0.1rem !important;
    }

    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.15rem !important;
        font-weight: 700;
        color: #1A1A2E !important;
        line-height: 1.3;
    }

    /* ═══════ BUTTONS ═══════ */
    .main .stButton > button {
        font-size: 0.82rem !important;
        padding: 0.3rem 0.7rem !important;
        min-height: 0 !important;
        border-radius: 6px;
        transition: all 0.15s ease;
    }

    .main .stButton > button[kind="primary"] {
        background: #7C3AED !important;
        border: none;
        font-weight: 600;
        color: #FFF !important;
    }

    .main .stButton > button[kind="primary"]:hover {
        background: #6D28D9 !important;
        box-shadow: 0 2px 8px rgba(124, 58, 237, 0.2);
    }

    .main .stButton > button[kind="secondary"] {
        border: 1px solid #E5E7EB !important;
        color: #374151 !important;
    }

    /* ═══════ CONTAINERS ═══════ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px !important;
        border-color: #E5E7EB !important;
    }

    div[data-testid="stExpander"] {
        border-radius: 8px !important;
        border: 1px solid #E5E7EB !important;
    }

    /* ═══════ TABS ═══════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        border-bottom: 1px solid #E5E7EB;
    }

    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        font-size: 0.82rem;
        color: #6B7280;
        padding: 0.35rem 0.6rem;
    }

    .stTabs [aria-selected="true"] {
        color: #7C3AED !important;
        font-weight: 600;
        border-bottom: 2px solid #7C3AED !important;
    }

    /* Tab content area: constrain height to prevent page scroll */
    .stTabs [data-baseweb="tab-panel"] {
        max-height: calc(100vh - 12rem) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scrollbar-width: thin;
        scrollbar-color: #D1D5DB transparent;
    }
    .stTabs [data-baseweb="tab-panel"]::-webkit-scrollbar { width: 4px; }
    .stTabs [data-baseweb="tab-panel"]::-webkit-scrollbar-thumb { background: #D1D5DB; border-radius: 2px; }

    /* ═══════ FORM INPUTS ═══════ */
    .main .stSelectbox > div > div,
    .main .stTextInput > div > div > input,
    .main .stNumberInput > div > div > input {
        font-size: 0.84rem !important;
        padding: 0.3rem 0.5rem !important;
        border-radius: 6px;
    }

    .main .stSelectbox label,
    .main .stTextInput label,
    .main .stNumberInput label,
    .main .stSlider label,
    .main .stCheckbox label,
    .main .stRadio label,
    .main .stFileUploader label,
    .main .stTextArea label {
        font-size: 0.82rem !important;
        margin-bottom: 0 !important;
    }

    .main .stCheckbox { margin-bottom: 0 !important; }
    .main .stSlider > div { padding-top: 0 !important; }
    .stSlider > div > div > div > div { background: #7C3AED !important; }

    /* ═══════ HEADINGS ═══════ */
    .main h1 { font-size: 1.25rem !important; margin: 0.1rem 0 !important; font-weight: 700; }
    .main h2 { font-size: 1.05rem !important; margin: 0.15rem 0 !important; font-weight: 600; }
    .main h3 { font-size: 0.92rem !important; margin: 0.1rem 0 !important; font-weight: 600; color: #1A1A2E; }
    .main h4 { font-size: 0.85rem !important; margin: 0.1rem 0 !important; font-weight: 600; color: #374151; }

    /* ═══════ EXPANDER ═══════ */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.82rem;
        padding: 0.3rem 0.5rem !important;
    }

    /* ═══════ STEP INDICATOR ═══════ */
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 0.1rem;
        padding: 0.25rem 0;
        margin-bottom: 0.3rem;
    }

    .step-item { text-align: center; flex: 1; }
    .step-icon { font-size: 0.9rem; display: block; }
    .step-label { font-size: 0.68rem; font-weight: 500; color: #6B7280; }
    .step-bar { height: 2px; border-radius: 1px; margin-top: 0.15rem; }
    .step-active .step-label { color: #7C3AED; font-weight: 700; }
    .step-active .step-bar { background: #7C3AED; }
    .step-done .step-bar { background: #10B981; }
    .step-pending .step-bar { background: #E5E7EB; }
    .step-pending .step-label { opacity: 0.5; }

    /* ═══════ PIPELINE ═══════ */
    .pipeline-steps {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.15rem;
        padding: 0.25rem 0;
        margin-bottom: 0.2rem;
    }

    .pipeline-step {
        display: inline-flex;
        align-items: center;
        gap: 0.2rem;
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .pipeline-arrow { color: #D1D5DB; font-size: 0.65rem; }
    .ps-done { background: #ECFDF5; color: #065F46; }
    .ps-active { background: #FEF3C7; color: #92400E; }
    .ps-pending { background: #F9FAFB; color: #9CA3AF; }

    /* ═══════ INFO BAR ═══════ */
    .info-bar {
        background: #F8F7FF;
        border: 1px solid #EDE9FE;
        border-radius: 6px;
        padding: 0.35rem 0.8rem;
        text-align: center;
        font-size: 0.78rem;
        color: #6B21A8;
    }
    .info-bar b { color: #1A1A2E; }

    /* ═══════ STAT ROW ═══════ */
    .ds-card-stats {
        display: flex;
        gap: 0.6rem;
        font-size: 0.78rem;
        color: #6B7280;
        flex-wrap: wrap;
        line-height: 1.4;
    }
    .ds-card-stats span { white-space: nowrap; }

    /* ═══════ ALERTS ═══════ */
    .stAlert { padding: 0.35rem 0.6rem !important; font-size: 0.82rem !important; }

    /* ═══════ CAPTION ═══════ */
    .main .stCaption, .main small { font-size: 0.75rem !important; }

    /* ═══════ PROGRESS ═══════ */
    .stProgress > div > div {
        background: linear-gradient(90deg, #7C3AED, #A78BFA) !important;
        border-radius: 3px;
    }

    /* ═══════ FILE UPLOADER ═══════ */
    [data-testid="stFileUploader"] section {
        border-radius: 6px;
        border: 2px dashed #D1D5DB !important;
        padding: 0.6rem !important;
    }
    [data-testid="stFileUploader"] section:hover { border-color: #7C3AED !important; }

    /* ═══════ DATAFRAME ═══════ */
    .stDataFrame { border-radius: 6px; border: 1px solid #E5E7EB; }

    /* ═══════ DOWNLOAD ═══════ */
    .stDownloadButton > button {
        border-radius: 6px;
        font-size: 0.82rem !important;
        padding: 0.3rem 0.7rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_info_bar(items: dict):
    inner = " · ".join(f"{k}: <b>{v}</b>" for k, v in items.items())
    st.markdown(f'<div class="info-bar">{inner}</div>', unsafe_allow_html=True)


def render_page_header(icon: str, title: str, subtitle: str = ""):
    st.markdown(
        f'<div style="margin-bottom:0.4rem;">'
        f'<span style="font-size:1.2rem;font-weight:700;color:#1A1A2E;">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
