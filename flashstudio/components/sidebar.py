"""FlashStudio sidebar — friendly workspace navigation and status."""

import base64
import html
import os

import streamlit as st

from flashstudio.components.project_manager import (
    get_active_project, save_project_state,
)
from flashstudio.constants import DATASET_NAME_TRUNCATE, GPU_NAME_TRUNCATE
from flashstudio.utils.device import get_gpu_info

NAV_GROUPS = [
    ("Workspace", [("Dashboard", "🏠")]),
    ("Build", [("Data", "🗂"), ("Model", "🧩"), ("Training", "🎯")]),
    ("Ship", [("Export", "📦"), ("Inference", "🔎")]),
]

STEP_ORDER = ["Dashboard", "Data", "Model", "Training", "Export", "Inference"]

# Backwards-compatible flat navigation list used by older tests/integrations.
NAV = [
    ("🏠", "Dashboard"),
    ("🗂", "Data"),
    ("🧩", "Model"),
    ("🎯", "Training"),
    ("📦", "Export"),
    ("🔎", "Inference"),
]


def _get_logo_b64():
    logo_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "assets", "logo.png",
    )
    if not os.path.isfile(logo_path):
        logo_path = os.path.join(os.getcwd(), "assets", "logo.png")
    if os.path.isfile(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def _model_label() -> str:
    from flashstudio.utils import get_state

    framework = str(get_state("model_framework") or "FlashDet")
    if framework == "Ultralytics":
        series = str(get_state("yolo_series") or "YOLOv8")
        size = str(get_state("yolo_size") or "n").upper()
        return f"{series}-{size}"
    if framework == "YOLOX":
        return "YOLOX"
    arch = str(get_state("model_arch") or "FlashDet-Pico")
    return arch.replace("FlashDet-", "")


def _render_brand():
    logo_b64 = _get_logo_b64()
    if logo_b64:
        mark = (
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'style="width:42px;height:42px;border-radius:12px;object-fit:cover;">'
        )
    else:
        mark = '<div class="fs-brand-mark">FS</div>'
    st.markdown(
        f'<div class="fs-brand">{mark}'
        f'<div><div class="fs-brand-name">FlashStudio</div>'
        f'<div class="fs-brand-tag">Detection workspace</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_project(project):
    if not project:
        st.markdown(
            '<div class="fs-project-card">'
            '<div class="fs-project-kicker">No project</div>'
            '<div class="fs-project-name">Create a project</div>'
            '<div class="fs-project-meta">Datasets, runs, and exports stay together.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        name = html.escape(str(project.get("name", "Untitled")))
        desc = html.escape(str(project.get("description", ""))[:54])
        st.markdown(
            f'<div class="fs-project-card">'
            f'<div class="fs-project-kicker">Project</div>'
            f'<div class="fs-project-name">{name}</div>'
            f'<div class="fs-project-meta">{desc or "Ready to build."}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if st.button("Manage projects", key="sb_project", use_container_width=True):
        save_project_state()
        st.session_state["show_project_manager"] = True
        st.rerun()


def _navigate_to(index: int):
    st.session_state["current_step"] = index
    st.session_state["show_project_manager"] = False
    st.session_state.pop("_pending_nav", None)
    st.rerun()


def _render_nav_guard():
    """Ask what to do when the user leaves while a download is running."""
    pending = st.session_state.get("_pending_nav")
    if pending is None:
        return
    st.warning("A dataset download is still running.")
    if st.button("Keep downloading", key="nav_dl_continue",
                 use_container_width=True):
        _navigate_to(pending)
    if st.button("Cancel download and switch", key="nav_dl_cancel",
                 use_container_width=True):
        from flashstudio.pages.data.helpers import _cancel_dataset_download

        _cancel_dataset_download()
        _navigate_to(pending)
    if st.button("Stay here", key="nav_dl_stay", use_container_width=True):
        st.session_state.pop("_pending_nav", None)
        st.rerun()


def _render_nav(current: int):
    _render_nav_guard()
    for group, items in NAV_GROUPS:
        st.markdown(f'<div class="fs-nav-label">{group}</div>', unsafe_allow_html=True)
        for label, icon in items:
            index = STEP_ORDER.index(label)
            if st.button(
                f"{icon}  {label}",
                key=f"sb_{index}",
                type="primary" if index == current else "secondary",
                use_container_width=True,
            ):
                download = st.session_state.get("_dataset_download")
                if download and index != current and index != 1:
                    # Don't silently abandon an active download: offer to
                    # keep it running or cancel it before switching pages.
                    st.session_state["_pending_nav"] = index
                    st.rerun()
                _navigate_to(index)


def _render_progress(current: int):
    total = len(STEP_ORDER)
    pct = int(round(((current + 1) / total) * 100))
    st.markdown(
        f'<div class="fs-side-progress">'
        f'<div class="fs-side-bar"><span style="width:{pct}%"></span></div>'
        f'<div class="fs-side-step">Step {current + 1} of {total}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_status():
    gpu = get_gpu_info()
    device = (
        html.escape(str(gpu["name"])[:GPU_NAME_TRUNCATE])
        if gpu.get("available") else "CPU"
    )
    dataset = html.escape(
        str(st.session_state.get("dataset_name") or "Not loaded")[:DATASET_NAME_TRUNCATE]
    )
    model = html.escape(_model_label())
    raw_status = str(st.session_state.get("training_status") or "Idle")
    status = html.escape(raw_status[:14] + ("…" if len(raw_status) > 14 else ""))
    status_color = {
        "Complete": "#059669",
        "Running": "#D97706",
        "Paused": "#D97706",
        "Failed": "#DC2626",
    }.get(str(st.session_state.get("training_status") or ""), "#9CA3AF")

    st.markdown(
        f'<div class="fs-status-card">'
        f'<div class="fs-status-row"><span>Device</span><span>{device}</span></div>'
        f'<div class="fs-status-row"><span>Dataset</span><span>{dataset}</span></div>'
        f'<div class="fs-status-row"><span>Model</span><span>{model}</span></div>'
        f'<div class="fs-status-row"><span>Training</span>'
        f'<span><span class="fs-dot" style="background:{status_color};"></span>{status}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    download = st.session_state.get("_dataset_download")
    if download:
        pct = download.get("percent")
        pct_text = f"{float(pct) * 100:.0f}%" if isinstance(pct, (int, float)) else "working"
        st.caption(
            f"Downloading **{html.escape(str(download.get('dataset_id') or download.get('name') or 'dataset'))}** · {pct_text}"
        )
        if st.button("Cancel download", key="sb_cancel_download", use_container_width=True):
            try:
                from flashstudio.pages.data.helpers import _cancel_dataset_download

                _cancel_dataset_download()
            except Exception as exc:
                st.caption(f"Cancel failed: {exc}")
            st.rerun()


def _render_storage(project):
    """Let the user choose where datasets, runs, and weights are stored."""
    from flashstudio.utils import storage

    defaults = {
        "datasets": storage.default_datasets_root(),
        "runs": storage.default_runs_root(),
        "weights": storage.DEFAULT_WEIGHTS_DIR,
    }
    getters = {
        "datasets": storage.get_datasets_root,
        "runs": storage.get_runs_root,
        "weights": storage.get_weights_root,
    }
    resolvers = {
        "datasets": storage.resolve_datasets_root,
        "runs": storage.resolve_runs_root,
        "weights": storage.resolve_weights_root,
    }
    setters = {
        "datasets": storage.set_datasets_root,
        "runs": storage.set_runs_root,
        "weights": storage.set_weights_root,
    }
    keys = {
        "datasets": "_storage_datasets_input",
        "runs": "_storage_runs_input",
        "weights": "_storage_weights_input",
    }
    labels = {
        "datasets": "Datasets",
        "runs": "Training runs",
        "weights": "Pretrained weights",
    }
    help_text = {
        "datasets": "Downloaded and extracted datasets are saved here.",
        "runs": "Checkpoints, logs, and curves for each training run.",
        "weights": "Shared weight cache. Used by every project.",
    }

    with st.expander("📁 Storage locations"):
        st.caption(
            "Pick any folder, including another drive. Dataset and run folders "
            "belong to the active project; weights are shared."
        )
        for name in ("datasets", "runs", "weights"):
            if keys[name] not in st.session_state:
                st.session_state[keys[name]] = getters[name]()
            st.text_input(
                labels[name],
                key=keys[name],
                help=f"{help_text[name]} Leave empty to use the default.",
            )

        if st.button(
            "Apply", key="sb_storage_apply", use_container_width=True,
        ):
            errors = []
            resolved = {}
            for name in ("datasets", "runs", "weights"):
                path = resolvers[name](
                    str(st.session_state.get(keys[name]) or "")
                )
                ok, message = storage.ensure_dir(path)
                if not ok:
                    errors.append(f"{labels[name]}: {message}")
                resolved[name] = path
            if errors:
                for message in errors:
                    st.error(message)
            else:
                for name in ("datasets", "runs", "weights"):
                    setters[name](resolved[name])
                    st.session_state[keys[name]] = resolved[name]
                if project:
                    save_project_state()
                st.rerun()

        if st.button(
            "Use defaults", key="sb_storage_reset", use_container_width=True,
            help="Restore the project folders and the shared weight cache.",
        ):
            for name in ("datasets", "runs", "weights"):
                setters[name](defaults[name])
                st.session_state[keys[name]] = defaults[name]
            if project:
                save_project_state()
            st.rerun()

        current = {name: getters[name]() for name in keys}
        st.caption(
            f"Datasets: `{current['datasets']}`  \n"
            f"Runs: `{current['runs']}`  \n"
            f"Weights: `{current['weights']}`"
        )


def render_sidebar():
    with st.sidebar:
        _render_brand()
        project = get_active_project()
        _render_project(project)

        current = int(st.session_state.get("current_step", 0) or 0)
        _render_nav(current)
        _render_progress(current)

        _render_storage(project)

        st.divider()
        _render_status()
