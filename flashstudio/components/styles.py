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

    .stApp, section[data-testid="stMain"] { overflow: visible !important; height: auto !important; }

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
        color: #fff; font-weight: 800; font-size: .95rem;
        box-shadow: 0 6px 16px rgba(109, 40, 217, .25);
    }
    .fs-brand-name { font-weight: 800; font-size: 1.02rem; line-height: 1.1; color: var(--fs-text); }
    .fs-brand-tag { font-size: .72rem; color: var(--fs-muted); margin-top: .12rem; }

    .fs-nav-label {
        font-size: .67rem; font-weight: 700; letter-spacing: .11em;
        text-transform: uppercase; color: var(--fs-faint);
        padding: .7rem .5rem .28rem;
    }

    section[data-testid="stSidebar"] .stButton > button {
        font-size: .9rem !important;
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
    .fs-project-kicker { font-size: .64rem; font-weight: 700; letter-spacing: .1em; color: #8B5CF6; text-transform: uppercase; }
    .fs-project-name { font-size: .95rem; font-weight: 700; color: var(--fs-text); margin-top: .15rem; }
    .fs-project-meta { font-size: .72rem; color: var(--fs-muted); margin-top: .1rem; }

    .fs-side-progress {
        display: flex; align-items: center; gap: .55rem;
        padding: .55rem .5rem .2rem;
    }
    .fs-side-bar { flex: 1; height: 5px; border-radius: 999px; background: #ECEAF4; overflow: hidden; }
    .fs-side-bar > span { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6D28D9, #A78BFA); }
    .fs-side-step { font-size: .7rem; color: var(--fs-muted); font-weight: 600; white-space: nowrap; }

    .fs-status-card {
        border: 1px solid var(--fs-border-soft);
        background: #FAFBFC;
        border-radius: 14px;
        padding: .75rem .8rem;
        font-size: .78rem;
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
        font-size: .68rem; font-weight: 700; letter-spacing: .12em;
        text-transform: uppercase; color: var(--fs-primary);
        display: flex; align-items: center; gap: .45rem;
    }
    .page-kicker::before {
        content: ""; width: 18px; height: 2px; border-radius: 2px;
        background: linear-gradient(90deg, #6D28D9, #A78BFA);
    }
    .page-title { font-size: 1.85rem !important; font-weight: 800 !important; color: var(--fs-text) !important; margin: .18rem 0 .1rem !important; letter-spacing: -.03em; }
    .page-subtitle { font-size: .9rem !important; color: var(--fs-muted) !important; max-width: 760px; line-height: 1.5; }
    .page-chip {
        display: inline-flex; align-items: center; gap: .4rem;
        padding: .42rem .75rem; border-radius: 999px;
        background: #fff; border: 1px solid var(--fs-border);
        font-size: .74rem; font-weight: 600; color: var(--fs-body);
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
        font-size: .7rem !important;
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
        font-size: .74rem !important; color: var(--fs-muted) !important;
    }

    /* ── Buttons ─────────────────────────────────────────────── */
    section[data-testid="stMain"] .stButton > button {
        font-size: .88rem !important;
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
        font-size: .86rem;
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

    /* ── Inputs ──────────────────────────────────────────────── */
    section[data-testid="stMain"] .stSelectbox > div > div,
    section[data-testid="stMain"] .stTextInput > div > div > input,
    section[data-testid="stMain"] .stNumberInput > div > div > input,
    section[data-testid="stMain"] .stTextArea textarea {
        font-size: .88rem !important;
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
        font-size: .8rem !important;
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
        font-size: .8rem !important;
    }

    /* ── Headings inside pages ───────────────────────────────── */
    section[data-testid="stMain"] h1 { font-size: 1.5rem !important; margin: .15rem 0 !important; font-weight: 800; letter-spacing: -.02em; }
    section[data-testid="stMain"] h2 { font-size: 1.28rem !important; margin: .15rem 0 !important; font-weight: 700; }
    section[data-testid="stMain"] h3 { font-size: 1.08rem !important; margin: .1rem 0 !important; font-weight: 700; color: var(--fs-text); }
    section[data-testid="stMain"] h4 {
        font-size: .82rem !important; margin: .1rem 0 .35rem !important;
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
        font-size: .86rem !important;
        color: var(--fs-body) !important;
        padding: .55rem .8rem !important;
    }

    /* ── Alerts ──────────────────────────────────────────────── */
    .stAlert {
        padding: .6rem .8rem !important;
        font-size: .85rem !important;
        border-radius: 11px !important;
        border: 1px solid var(--fs-border) !important;
    }

    /* ── Info bar / custom components ────────────────────────── */
    .info-bar {
        background: linear-gradient(135deg, #F8F5FF, #F1ECFF);
        border: 1px solid #E5DBFF;
        border-radius: 12px;
        padding: .55rem .9rem;
        font-size: .84rem;
        color: #5B21B6;
        display: flex; flex-wrap: wrap; gap: .35rem .9rem;
    }
    .info-bar b { color: var(--fs-text); }

    .ds-card-stats {
        display: flex; gap: .5rem .9rem; flex-wrap: wrap;
        font-size: .82rem; color: var(--fs-muted); line-height: 1.5;
    }
    .ds-card-stats span { white-space: nowrap; }
    .ds-card-stats b { color: var(--fs-text); }

    .ds-check-row { display: flex; flex-wrap: wrap; gap: .4rem; margin: .15rem 0 .5rem; }
    .ds-check {
        display: inline-flex; align-items: center;
        padding: .26rem .62rem; border-radius: 999px;
        font-size: .76rem; font-weight: 600; white-space: nowrap;
    }
    .ds-ok { background: #ECFDF5; color: #047857; }
    .ds-fail { background: #FEF2F2; color: #B91C1C; }
    .ds-note { background: #FFFBEB; color: #92400E; }
    .ds-badge {
        display: inline-block; padding: .12rem .45rem; border-radius: 999px;
        background: #ECFDF5; color: #047857; font-size: .68rem; font-weight: 700;
        vertical-align: middle;
    }

    .pipeline-steps { display: flex; flex-wrap: wrap; align-items: center; gap: .3rem; }
    .pipeline-step {
        display: inline-flex; align-items: center; gap: .3rem;
        padding: .3rem .6rem; border-radius: 999px;
        font-size: .78rem; font-weight: 600;
    }
    .pipeline-arrow { color: var(--fs-faint); font-size: .7rem; }
    .ps-done { background: #ECFDF5; color: #065F46; }
    .ps-active { background: #FEF3C7; color: #92400E; }
    .ps-pending { background: #F4F5FA; color: #9CA3AF; }

    .stProgress > div > div {
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
        font-size: .85rem !important;
        font-weight: 600 !important;
        min-height: 2.3rem !important;
    }

    [data-testid="stAppDeployButton"] { display: none !important; }

    .app-footer {
        text-align: center;
        color: var(--fs-faint);
        font-size: .72rem;
        padding: 2rem 0 .5rem;
        letter-spacing: .02em;
    }

    /* Zone canvas keeps a usable viewport. */
    div[data-testid="stCustomComponentV1"] {
        min-height: 440px !important;
        width: 670px !important;
        max-width: 100% !important;
        margin: 0 !important;
    }
    iframe[data-testid="stCustomComponentV1"] {
        width: 670px !important;
        max-width: 100% !important;
        height: 440px !important;
        min-height: 440px !important;
        margin: 0 !important;
        display: block !important;
    }

    section[data-testid="stMain"] .stCaption, section[data-testid="stMain"] small { font-size: .78rem !important; color: var(--fs-muted) !important; }

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
    .workflow-step { font-size: .75rem; color: var(--fs-muted); font-weight: 600; }
    .workflow-hint { font-size: .72rem; color: var(--fs-faint); }

    .step-indicator { display: flex; gap: .35rem; padding: .2rem 0 .6rem; }
    .step-item { flex: 1; text-align: center; }
    .step-icon { font-size: .85rem; display: block; color: var(--fs-faint); }
    .step-label { font-size: .74rem; font-weight: 600; color: var(--fs-muted); }
    .step-bar { height: 3px; border-radius: 999px; margin-top: .2rem; background: #E9E7F1; }
    .step-active .step-label { color: var(--fs-primary); }
    .step-active .step-bar { background: var(--fs-primary); }
    .step-done .step-bar { background: #10B981; }
    .step-done .step-label { color: var(--fs-body); }
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
