"""FlashStudio — shared visual system and page chrome."""

import html

import streamlit as st


_PAGE_META = {
    "Dashboard": ("01", "Overview"),
    "Data": ("02", "Prepare"),
    "Model": ("03", "Build"),
    "Training": ("04", "Train"),
    "Export": ("05", "Ship"),
    "Inference": ("06", "Run"),
}


def inject_custom_css():
    """Install the FlashStudio design system."""
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --fs-primary: #6D28D9;
        --fs-primary-2: #8B5CF6;
        --fs-primary-soft: #F1ECFF;
        --fs-bg: #F4F5FA;
        --fs-card: #FFFFFF;
        --fs-border: #E5E7EF;
        --fs-border-soft: #EEF0F6;
        --fs-text: #111827;
        --fs-body: #374151;
        --fs-muted: #6B7280;
        --fs-faint: #9CA3AF;
        --fs-success: #059669;
        --fs-warning: #B45309;
        --fs-danger: #DC2626;
        --fs-radius: 16px;
        --fs-radius-sm: 10px;
        --fs-shadow: 0 1px 2px rgba(16, 24, 40, .04), 0 8px 24px rgba(16, 24, 40, .04);
        /* Type scale — every UI font size uses one of these four steps. */
        --fs-text-xs: .7rem;
        --fs-text-sm: .78rem;
        --fs-text-md: .86rem;
        --fs-text-lg: .95rem;
    }

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--fs-text);
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(900px 420px at 10% -10%, #EDE9FE 0%, transparent 55%),
            radial-gradient(700px 380px at 100% 0%, #EEF2FF 0%, transparent 55%),
            var(--fs-bg);
    }

    /* Keep Streamlit's own scroll container intact: forcing stMain to
       ``overflow: visible`` makes everything below the fold unreachable on
       short screens. */

    section[data-testid="stMain"] .block-container {
        padding: 3.6rem 2rem 2.6rem !important;
        max-width: 1480px;
        height: auto !important;
        overflow: visible !important;
    }

    section[data-testid="stMain"] .stMarkdown { margin-bottom: 0 !important; }
    section[data-testid="stMain"] .element-container { margin-bottom: 0.15rem !important; }
    section[data-testid="stMain"] div[data-testid="stVerticalBlock"] > div { gap: 0.55rem !important; }
    section[data-testid="stMain"] hr { margin: 0.6rem 0 !important; border-color: var(--fs-border) !important; }

    /* ── Sidebar ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: #FFFFFF !important;
        border-right: 1px solid var(--fs-border);
        width: 276px !important;
        box-shadow: 8px 0 30px rgba(17, 24, 39, .03);
    }
    section[data-testid="stSidebar"] > div {
        width: 276px !important;
        padding: 1rem 0.85rem 1.2rem !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--fs-border-soft) !important;
        margin: 0.75rem 0 !important;
    }

    .fs-brand {
        display: flex; align-items: center; gap: .7rem;
        padding: .45rem .45rem .8rem;
    }
    .fs-brand-mark {
        width: 42px; height: 42px; border-radius: 12px;
        display: grid; place-items: center; flex: 0 0 42px;
        background: linear-gradient(135deg, #6D28D9, #A78BFA);
        color: #fff; font-weight: 800; font-size: var(--fs-text-lg);
        box-shadow: 0 6px 16px rgba(109, 40, 217, .25);
    }
    .fs-brand-name { font-weight: 800; font-size: 1.02rem; line-height: 1.1; color: var(--fs-text); }
    .fs-brand-tag { font-size: var(--fs-text-sm); color: var(--fs-muted); margin-top: .12rem; }

    .fs-nav-label {
        font-size: var(--fs-text-xs); font-weight: 700; letter-spacing: .11em;
        text-transform: uppercase; color: var(--fs-faint);
        padding: .7rem .5rem .28rem;
    }

    section[data-testid="stSidebar"] .stButton > button {
        font-size: var(--fs-text-md) !important;
        padding: .5rem .7rem !important;
        border-radius: 10px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        min-height: 2.35rem !important;
        height: auto !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: var(--fs-body) !important;
        font-weight: 600 !important;
        transition: background .15s ease, color .15s ease, transform .15s ease;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: #F7F5FF !important;
        color: var(--fs-primary) !important;
        transform: translateX(1px);
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: var(--fs-primary-soft) !important;
        color: var(--fs-primary) !important;
        border-color: #DDD3FF !important;
        box-shadow: inset 3px 0 0 var(--fs-primary);
    }

    .fs-project-card {
        background: linear-gradient(135deg, #FBFAFF, #F4F1FF);
        border: 1px solid #E9E2FF;
        border-radius: 14px;
        padding: .75rem .8rem;
        margin: .2rem 0 .45rem;
    }
    .fs-project-kicker { font-size: var(--fs-text-xs); font-weight: 700; letter-spacing: .1em; color: #8B5CF6; text-transform: uppercase; }
    .fs-project-name { font-size: var(--fs-text-lg); font-weight: 700; color: var(--fs-text); margin-top: .15rem; }
    .fs-project-meta { font-size: var(--fs-text-sm); color: var(--fs-muted); margin-top: .1rem; }

    .fs-side-progress {
        display: flex; align-items: center; gap: .55rem;
        padding: .55rem .5rem .2rem;
    }
    .fs-side-bar { flex: 1; height: 5px; border-radius: 999px; background: #ECEAF4; overflow: hidden; }
    .fs-side-bar > span { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6D28D9, #A78BFA); }
    .fs-side-step { font-size: var(--fs-text-xs); color: var(--fs-muted); font-weight: 600; white-space: nowrap; }

    .fs-status-card {
        border: 1px solid var(--fs-border-soft);
        background: #FAFBFC;
        border-radius: 14px;
        padding: .75rem .8rem;
        font-size: var(--fs-text-sm);
    }
    .fs-status-row { display: flex; align-items: center; justify-content: space-between; gap: .5rem; padding: .18rem 0; }
    .fs-status-row span:first-child { color: var(--fs-muted); }
    .fs-status-row span:last-child { color: var(--fs-body); font-weight: 600; text-align: right; }
    .fs-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: .35rem; }

    [data-testid="stSidebarNav"] { display: none !important; }

    /* ── Page header ─────────────────────────────────────────── */
    .page-header {
        display: flex; align-items: flex-end; justify-content: space-between;
        gap: 1rem; padding: .2rem 0 .9rem;
        border-bottom: 1px solid var(--fs-border);
        margin-bottom: .9rem;
    }
    .page-kicker {
        font-size: var(--fs-text-xs); font-weight: 700; letter-spacing: .12em;
        text-transform: uppercase; color: var(--fs-primary);
        display: flex; align-items: center; gap: .45rem;
    }
    .page-kicker::before {
        content: ""; width: 18px; height: 2px; border-radius: 2px;
        background: linear-gradient(90deg, #6D28D9, #A78BFA);
    }
    .page-title { font-size: 1.85rem !important; font-weight: 800 !important; color: var(--fs-text) !important; margin: .18rem 0 .1rem !important; letter-spacing: -.03em; }
    .page-subtitle { font-size: var(--fs-text-md) !important; color: var(--fs-muted) !important; max-width: 760px; line-height: 1.5; }
    .page-chip {
        display: inline-flex; align-items: center; gap: .4rem;
        padding: .42rem .75rem; border-radius: 999px;
        background: #fff; border: 1px solid var(--fs-border);
        font-size: var(--fs-text-sm); font-weight: 600; color: var(--fs-body);
        box-shadow: var(--fs-shadow); white-space: nowrap;
    }

    /* ── Cards, metrics, containers ──────────────────────────── */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--fs-card);
        border: 1px solid var(--fs-border) !important;
        border-radius: var(--fs-radius) !important;
        box-shadow: var(--fs-shadow);
        transition: border-color .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #DCD8EE !important;
    }

    div[data-testid="stMetric"] {
        background: var(--fs-card);
        border: 1px solid var(--fs-border);
        border-radius: 14px;
        padding: .85rem 1rem !important;
        box-shadow: var(--fs-shadow);
        min-height: 92px;
    }
    div[data-testid="stMetric"] label {
        font-size: var(--fs-text-xs) !important;
        text-transform: uppercase;
        letter-spacing: .09em;
        color: var(--fs-muted) !important;
        font-weight: 700;
        margin-bottom: .15rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        font-weight: 800;
        color: var(--fs-text) !important;
        line-height: 1.15;
        letter-spacing: -.02em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: var(--fs-text-sm) !important; color: var(--fs-muted) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────── */
    section[data-testid="stMain"] .stButton > button {
        font-size: var(--fs-text-md) !important;
        font-weight: 600 !important;
        padding: .48rem .85rem !important;
        min-height: 2.45rem !important;
        border-radius: var(--fs-radius-sm);
        border: 1px solid var(--fs-border) !important;
        background: #fff !important;
        color: var(--fs-body) !important;
        transition: all .15s ease;
    }
    section[data-testid="stMain"] .stButton > button:hover {
        border-color: #CFC6E9 !important;
        color: var(--fs-primary) !important;
        box-shadow: 0 4px 12px rgba(109, 40, 217, .08);
    }
    section[data-testid="stMain"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6D28D9, #8B5CF6) !important;
        border: none !important;
        color: #fff !important;
        box-shadow: 0 8px 20px rgba(109, 40, 217, .22);
    }
    section[data-testid="stMain"] .stButton > button[kind="primary"]:hover {
        filter: brightness(1.04);
        box-shadow: 0 10px 24px rgba(109, 40, 217, .28);
    }

    /* ── Tabs ────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: .3rem;
        border-bottom: none !important;
        background: #ECECF4;
        padding: .28rem;
        border-radius: 12px;
        width: fit-content;
        max-width: 100%;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: var(--fs-text-md);
        color: var(--fs-muted);
        padding: .45rem .85rem;
        border-radius: 9px;
        border-bottom: none !important;
        height: auto;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--fs-primary); background: rgba(255,255,255,.6); }
    .stTabs [aria-selected="true"] {
        background: #fff !important;
        color: var(--fs-primary) !important;
        box-shadow: 0 2px 8px rgba(17, 24, 39, .08);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none !important; }
    .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: .85rem !important;
        max-height: none !important;
        overflow: visible !important;
    }

    /* ── Page-level view switcher (segmented control) ────────── */
    section[data-testid="stMain"] div[data-testid="stButtonGroup"] {
        background: #ECECF4;
        padding: .28rem;
        border-radius: 12px;
        width: fit-content;
        max-width: 100%;
    }
    section[data-testid="stMain"] div[data-testid="stButtonGroup"] > div[role="radiogroup"] {
        gap: .3rem;
        flex-wrap: wrap;
    }
    section[data-testid="stMain"] div[data-testid="stButtonGroup"] button[kind="segmented_control"],
    section[data-testid="stMain"] div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] {
        font-size: var(--fs-text-md) !important;
        font-weight: 600 !important;
        color: var(--fs-muted) !important;
        background: transparent !important;
        border: none !important;
        border-radius: 9px !important;
        min-height: 2.3rem !important;
        padding: .45rem .95rem !important;
        box-shadow: none !important;
    }
    section[data-testid="stMain"] div[data-testid="stButtonGroup"] button[kind="segmented_control"]:hover {
        color: var(--fs-primary) !important;
        background: rgba(255, 255, 255, .6) !important;
    }
    section[data-testid="stMain"] div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] {
        background: #fff !important;
        color: var(--fs-primary) !important;
        box-shadow: 0 2px 8px rgba(17, 24, 39, .08) !important;
    }

    /* ── Inputs ──────────────────────────────────────────────── */
    section[data-testid="stMain"] .stSelectbox > div > div,
    section[data-testid="stMain"] .stTextInput > div > div > input,
    section[data-testid="stMain"] .stNumberInput > div > div > input,
    section[data-testid="stMain"] .stTextArea textarea {
        font-size: var(--fs-text-md) !important;
        border-radius: var(--fs-radius-sm) !important;
        border-color: var(--fs-border) !important;
        background: #fff !important;
    }
    section[data-testid="stMain"] .stTextInput > div > div > input:focus,
    section[data-testid="stMain"] .stNumberInput > div > div > input:focus,
    section[data-testid="stMain"] .stTextArea textarea:focus {
        border-color: var(--fs-primary-2) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, .12) !important;
    }
    section[data-testid="stMain"] .stSelectbox label,
    section[data-testid="stMain"] .stTextInput label,
    section[data-testid="stMain"] .stNumberInput label,
    section[data-testid="stMain"] .stSlider label,
    section[data-testid="stMain"] .stCheckbox label,
    section[data-testid="stMain"] .stRadio label,
    section[data-testid="stMain"] .stFileUploader label,
    section[data-testid="stMain"] .stTextArea label,
    section[data-testid="stMain"] .stMultiSelect label {
        font-size: var(--fs-text-md) !important;
        font-weight: 600 !important;
        color: var(--fs-body) !important;
    }
    section[data-testid="stMain"] .stCheckbox { margin-bottom: 0 !important; }
    .stSlider > div > div > div > div { background: var(--fs-primary) !important; }

    section[data-testid="stMain"] div[role="radiogroup"] { gap: .4rem; }
    section[data-testid="stMain"] div[role="radiogroup"] label[data-baseweb="radio"] {
        background: #F4F4FA !important;
        border: 1px solid transparent !important;
        border-radius: 999px !important;
        padding: .28rem .7rem !important;
        margin-right: .1rem !important;
        transition: background .15s ease, border-color .15s ease;
    }
    section[data-testid="stMain"] div[role="radiogroup"] label[data-baseweb="radio"]:hover {
        background: #EFEBFF !important;
    }
    section[data-testid="stMain"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) {
        background: var(--fs-primary-soft) !important;
        border-color: #DDD3FF !important;
    }
    section[data-testid="stMain"] div[role="radiogroup"] label[data-baseweb="radio"] p {
        color: var(--fs-body) !important;
        font-weight: 600 !important;
    }

    [data-testid="stCode"] {
        border-radius: 12px !important;
        border: 1px solid var(--fs-border) !important;
        background: #F8F9FC !important;
    }
    [data-testid="stCode"] code, section[data-testid="stMain"] pre code {
        color: #1F2937 !important;
        font-size: var(--fs-text-md) !important;
    }

    /* ── Headings inside pages ───────────────────────────────── */
    section[data-testid="stMain"] h1 { font-size: 1.5rem !important; margin: .15rem 0 !important; font-weight: 800; letter-spacing: -.02em; }
    section[data-testid="stMain"] h1.page-title { font-size: 1.85rem !important; }
    section[data-testid="stMain"] h2 { font-size: 1.28rem !important; margin: .15rem 0 !important; font-weight: 700; }
    section[data-testid="stMain"] h3 { font-size: 1.08rem !important; margin: .1rem 0 !important; font-weight: 700; color: var(--fs-text); }
    section[data-testid="stMain"] h4 {
        font-size: var(--fs-text-md) !important; margin: .1rem 0 .35rem !important;
        font-weight: 700; color: var(--fs-body);
        text-transform: uppercase; letter-spacing: .07em;
    }

    /* ── Expanders ───────────────────────────────────────────── */
    div[data-testid="stExpander"] {
        border-radius: 12px !important;
        border: 1px solid var(--fs-border) !important;
        background: #fff !important;
        box-shadow: none;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        font-size: var(--fs-text-md) !important;
        color: var(--fs-body) !important;
        padding: .55rem .8rem !important;
    }

    /* ── Alerts ──────────────────────────────────────────────── */
    .stAlert {
        padding: .6rem .8rem !important;
        font-size: var(--fs-text-md) !important;
        border-radius: 11px !important;
        border: 1px solid var(--fs-border) !important;
    }

    /* ── Info bar / custom components ────────────────────────── */
    .info-bar {
        background: linear-gradient(135deg, #F8F5FF, #F1ECFF);
        border: 1px solid #E5DBFF;
        border-radius: 12px;
        padding: .55rem .9rem;
        font-size: var(--fs-text-md);
        color: #5B21B6;
        display: flex; flex-wrap: wrap; gap: .35rem .9rem;
    }
    .info-bar b { color: var(--fs-text); }

    .ds-card-stats {
        display: flex; gap: .5rem .9rem; flex-wrap: wrap;
        font-size: var(--fs-text-md); color: var(--fs-muted); line-height: 1.5;
    }
    .ds-card-stats span { white-space: nowrap; }
    .ds-card-stats b { color: var(--fs-text); }

    .ds-check-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .15rem 0 .5rem; }
    .ds-check {
        display: inline-flex; align-items: center;
        padding: .26rem .62rem; border-radius: 999px;
        font-size: var(--fs-text-sm); font-weight: 600; white-space: nowrap;
    }
    .ds-ok { background: #ECFDF5; color: #047857; }
    .ds-fail { background: #FEF2F2; color: #B91C1C; }
    .ds-note { background: #FFFBEB; color: #92400E; }
    .ds-badge {
        display: inline-block; padding: .12rem .45rem; border-radius: 999px;
        background: #ECFDF5; color: #047857; font-size: var(--fs-text-xs); font-weight: 700;
        vertical-align: middle;
    }

    .pipeline-steps { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem; }
    .pipeline-step {
        display: inline-flex; align-items: center; gap: .3rem;
        padding: .3rem .6rem; border-radius: 999px;
        font-size: var(--fs-text-sm); font-weight: 600;
    }
    .pipeline-arrow { color: var(--fs-faint); font-size: var(--fs-text-xs); }
    .ps-done { background: #ECFDF5; color: #065F46; }
    .ps-active { background: #FEF3C7; color: #92400E; }
    .ps-pending { background: #F4F5FA; color: #9CA3AF; }

    /* Only paint the fill — styling every nested div also painted the progress
       label and the whole track purple. */
    .stProgress > div > div:last-child > div {
        background: #ECEAF4 !important;
        border-radius: 999px;
    }
    .stProgress > div > div:last-child > div > div {
        background: linear-gradient(90deg, #6D28D9, #A78BFA) !important;
        border-radius: 999px;
    }

    [data-testid="stFileUploader"] section {
        border-radius: 12px;
        border: 1.5px dashed #D8D3E8 !important;
        background: #FBFAFF !important;
        padding: .7rem !important;
    }
    [data-testid="stFileUploader"] section:hover { border-color: var(--fs-primary-2) !important; }

    .stDataFrame { border-radius: 12px; border: 1px solid var(--fs-border); overflow: hidden; }

    .stDownloadButton > button {
        border-radius: var(--fs-radius-sm) !important;
        font-size: var(--fs-text-md) !important;
        font-weight: 600 !important;
        min-height: 2.3rem !important;
    }

    [data-testid="stAppDeployButton"] { display: none !important; }

    .app-footer {
        text-align: center;
        color: var(--fs-faint);
        font-size: var(--fs-text-sm);
        padding: 2rem 0 .5rem;
        letter-spacing: .02em;
    }

    /* Custom components (drawable canvases) size themselves. */
    div[data-testid="stCustomComponentV1"] {
        max-width: 100% !important;
        margin: 0 !important;
    }
    iframe[data-testid="stCustomComponentV1"] {
        max-width: 100% !important;
        margin: 0 !important;
        display: block !important;
    }

    section[data-testid="stMain"] .stCaption, section[data-testid="stMain"] small { font-size: var(--fs-text-sm) !important; color: var(--fs-muted) !important; }

    /* Streamlit wraps most control text in <p>/<span> with its own 14-16px
       defaults. Force those onto the shared scale so buttons, tabs, labels,
       captions, and markdown text match. */
    section[data-testid="stMain"] .stMarkdown p,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p {
        font-size: var(--fs-text-md) !important;
    }
    section[data-testid="stMain"] div[data-testid="stMetric"] [data-testid="stMarkdownContainer"] p {
        font-size: var(--fs-text-xs) !important;
    }
    section[data-testid="stMain"] div[data-testid="stMetric"] div[data-testid="stMetricValue"] p {
        font-size: 1.35rem !important;
        font-weight: 800;
        letter-spacing: -.02em;
    }
    section[data-testid="stMain"] div[data-testid="stMetric"] div[data-testid="stMetricDelta"] p {
        font-size: var(--fs-text-sm) !important;
    }
    section[data-testid="stMain"] .stCaption p,
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] p {
        font-size: var(--fs-text-sm) !important;
    }
    section[data-testid="stMain"] .stCaption code,
    section[data-testid="stMain"] [data-testid="stCaptionContainer"] code,
    section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p code {
        font-size: var(--fs-text-sm) !important;
    }
    section[data-testid="stMain"] [data-baseweb="tab"] p { font-size: var(--fs-text-md) !important; }
    section[data-testid="stMain"] button span,
    section[data-testid="stMain"] button div,
    section[data-testid="stSidebar"] button span,
    section[data-testid="stSidebar"] button div,
    section[data-testid="stMain"] button p,
    section[data-testid="stSidebar"] button p { font-size: inherit !important; }
    section[data-testid="stMain"] div[data-testid="stExpander"] summary div,
    section[data-testid="stMain"] div[data-testid="stExpander"] summary p {
        font-size: inherit !important;
    }
    section[data-testid="stMain"] label p { font-size: inherit !important; }
    section[data-testid="stMain"] [data-testid="stFileUploader"] p,
    section[data-testid="stMain"] [data-testid="stFileUploader"] small {
        font-size: var(--fs-text-md) !important;
    }

    /* ── Workflow footer ─────────────────────────────────────── */
    .workflow-footer {
        display: flex; align-items: center; justify-content: space-between;
        gap: .8rem; margin-top: 1.4rem; padding-top: .9rem;
        border-top: 1px solid var(--fs-border);
    }
    .workflow-divider {
        margin-top: 1.6rem;
        border-top: 1px solid var(--fs-border);
    }
    .workflow-step { font-size: var(--fs-text-sm); color: var(--fs-muted); font-weight: 600; }
    .workflow-hint { font-size: var(--fs-text-sm); color: var(--fs-faint); }

    .step-indicator { display: flex; gap: .35rem; padding: .2rem 0 .6rem; }
    .step-item { flex: 1; text-align: center; }
    .step-icon { font-size: var(--fs-text-md); display: block; color: var(--fs-faint); }
    .step-label { font-size: var(--fs-text-sm); font-weight: 600; color: var(--fs-muted); }
    .step-bar { height: 3px; border-radius: 999px; margin-top: .2rem; background: #E9E7F1; }
    .step-active .step-label { color: var(--fs-primary); }
    .step-active .step-bar { background: var(--fs-primary); }
    .step-done .step-bar { background: #10B981; }
    .step-done .step-label { color: var(--fs-body); }

    /* ── Responsive layout ───────────────────────────────────── */
    @media (max-width: 1200px) {
        section[data-testid="stMain"] .block-container {
            padding: 2.8rem 1.1rem 2rem !important;
        }
        /* Metric / action rows: two per row instead of crushing 4-7 cards. */
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4)) {
            flex-wrap: wrap !important;
            row-gap: .55rem !important;
        }
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4)) > div[data-testid="stColumn"] {
            /* Two per row: subtract half of Streamlit's 16px column gap so the
               items actually fit on one line instead of each taking a row. */
            flex: 1 1 calc(50% - .5rem) !important;
            min-width: calc(50% - .5rem) !important;
            width: auto !important;
        }
    }

    @media (max-width: 1024px) {
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div {
            width: 248px !important;
        }
        section[data-testid="stMain"] h1.page-title { font-size: 1.55rem !important; }
    }

    @media (max-width: 900px) {
        .page-header { align-items: flex-start; flex-wrap: wrap; }
        .page-chip { margin-top: .2rem; }
        /* Main content columns (setup + launch, side-by-side panels) stack. */
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:not(:has(> div[data-testid="stColumn"]:nth-child(3))) {
            flex-wrap: wrap !important;
        }
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:not(:has(> div[data-testid="stColumn"]:nth-child(3))) > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }
    }

    @media (max-width: 640px) {
        section[data-testid="stMain"] .block-container {
            padding: 2.4rem .7rem 1.5rem !important;
        }
        section[data-testid="stMain"] h1.page-title { font-size: 1.35rem !important; }
        .page-subtitle { font-size: var(--fs-text-md) !important; }
        .page-header { gap: .4rem; padding-bottom: .6rem; }
        /* Everything stacks once space is really tight. */
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
        section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(> div[data-testid="stColumn"]:nth-child(4)) > div[data-testid="stColumn"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }
        .stTabs [data-baseweb="tab-list"] { width: 100% !important; }
        .stTabs [data-baseweb="tab"] { flex: 1 1 auto; justify-content: center; }
        section[data-testid="stMain"] div[data-testid="stButtonGroup"],
        section[data-testid="stMain"] div[data-testid="stButtonGroup"] > div[role="radiogroup"] {
            width: 100% !important;
        }
        section[data-testid="stMain"] div[data-testid="stButtonGroup"] button {
            flex: 1 1 auto;
            justify-content: center;
        }
        .ds-card-stats span { white-space: normal; }
        div[data-testid="stMetric"] { min-height: 72px !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 1.15rem !important;
        }
    }

    /* streamlit-autorefresh is an invisible timer component; without this
       rule its iframe reserves a large blank area in some Streamlit versions. */
    iframe[src*="streamlit_autorefresh.st_autorefresh"] {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
        border: 0 !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_info_bar(items: dict):
    inner = " · ".join(
        f"{html.escape(str(k))}: <b>{html.escape(str(v))}</b>" for k, v in items.items()
    )
    st.markdown(f'<div class="info-bar">{inner}</div>', unsafe_allow_html=True)


def render_page_header(icon: str, title: str, subtitle: str = ""):
    """Render the page header with workflow context."""
    step_no, stage = _PAGE_META.get(str(title), ("", ""))
    step_index = st.session_state.get("current_step")
    total = 6
    if step_no and step_index is not None:
        kicker = f"Step {int(step_index) + 1} of {total} · {stage}"
    elif step_no:
        kicker = f"Step {step_no} · {stage}"
    else:
        kicker = "Workspace"

    icon_html = (
        f'<span style="margin-right:.45rem;">{html.escape(str(icon))}</span>' if icon else ""
    )
    subtitle_html = (
        f'<div class="page-subtitle">{html.escape(str(subtitle))}</div>'
        if subtitle else ""
    )
    chip = (
        f'<span class="page-chip"><span class="fs-dot" '
        f'style="background:#8B5CF6;"></span>{html.escape(stage)}</span>'
        if stage else ""
    )
    st.markdown(
        f'<div class="page-header">'
        f'<div><div class="page-kicker">{html.escape(kicker)}</div>'
        f'<h1 class="page-title">{icon_html}{html.escape(str(title))}</h1>'
        f'{subtitle_html}</div>'
        f'<div>{chip}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
