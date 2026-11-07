"""FlashStudio — Main Streamlit Application."""

import streamlit as st

st.set_page_config(
    page_title="FlashStudio",
    page_icon="FS",
    layout="wide",
    initial_sidebar_state="expanded",
)

from flashstudio.components.styles import inject_custom_css  # noqa: E402
from flashstudio.components.sidebar import render_sidebar  # noqa: E402
from flashstudio.components.wizard import render_navigation  # noqa: E402
from flashstudio.components.project_manager import (  # noqa: E402
    get_active_project, render_project_manager_page,
    save_project_state, list_projects, create_project,
    load_project_state, _clear_session_for_new_project,
)
from flashstudio.pages.dashboard import render_dashboard  # noqa: E402
from flashstudio.pages.data import render_data_page  # noqa: E402
from flashstudio.pages.model import render_model_page  # noqa: E402
from flashstudio.pages.training import render_training_page  # noqa: E402
from flashstudio.pages.export import render_export_page  # noqa: E402
from flashstudio.pages.inference import render_inference_page  # noqa: E402

STEPS = [
    {"id": "dashboard", "label": "Dashboard", "icon": "1"},
    {"id": "data", "label": "Data", "icon": "2"},
    {"id": "model", "label": "Model", "icon": "3"},
    {"id": "training", "label": "Training", "icon": "4"},
    {"id": "export", "label": "Export", "icon": "5"},
    {"id": "inference", "label": "Inference", "icon": "6"},
]

PAGE_RENDERERS = {
    "dashboard": render_dashboard,
    "data": render_data_page,
    "model": render_model_page,
    "training": render_training_page,
    "export": render_export_page,
    "inference": render_inference_page,
}


def _ensure_config_in_session(project_id: str):
    """Re-seed any missing model config keys from saved project state.

    Streamlit may clean up widget-owned session state keys when the widget is
    not rendered (e.g. navigating away from the Model page). This ensures
    critical config values like epochs, batch_size, lr etc. always remain
    available in session state for other pages to read.
    """
    import os
    import json
    from flashstudio.components.project_manager import get_project_dir
    from flashstudio.utils.config import CONFIG_KEYS

    extra_keys = (
        "dataset_classes", "dataset_id", "dataset_output_path",
        "train_ann_path", "val_ann_path", "detected_format",
        "class_file", "resume_path", "exported_files",
    )
    config_keys = tuple(dict.fromkeys(CONFIG_KEYS + extra_keys))

    missing = [k for k in config_keys if k not in st.session_state]
    if not missing:
        return

    proj_dir = get_project_dir(project_id)
    state_file = os.path.join(proj_dir, "session_state.json")
    saved = {}
    if os.path.isfile(state_file):
        try:
            with open(state_file, "r") as f:
                saved = json.load(f)
        except (json.JSONDecodeError, OSError):
            saved = {}

    # The config mirror holds the freshest values (it is updated on every
    # Model page render). Prefer it over the file so a widget key that
    # Streamlit dropped during navigation is not re-seeded with a stale value.
    mirror = st.session_state.get("_config_mirror") or {}

    for key in missing:
        if key in mirror:
            st.session_state[key] = mirror[key]
        elif key in saved:
            st.session_state[key] = saved[key]

    if not saved:
        return

    # Exported artifacts live on disk; drop registry entries whose files were
    # deleted or belong to another project/install.
    exported = st.session_state.get("exported_files")
    if isinstance(exported, list):
        st.session_state["exported_files"] = [
            entry for entry in exported
            if isinstance(entry, dict)
            and entry.get("path")
            and os.path.isfile(str(entry.get("path")))
        ]

    # Drop stale paths that point at a different FlashStudio checkout and no
    # longer exist.  The project index is shared across installs, so old
    # absolute paths otherwise leak into the current workspace.
    from flashstudio.constants import PROJECT_ROOT

    for key, value in list(st.session_state.items()):
        if not isinstance(value, str) or not value:
            continue
        if not (key.endswith("_path") or key.endswith("_dir")):
            continue
        if os.path.exists(value):
            continue
        normalized = value.replace("\\", "/")
        if "/FlashStudio/" in normalized and not normalized.startswith(
            PROJECT_ROOT.replace("\\", "/")
        ):
            st.session_state.pop(key, None)

    # Migrate sessions/projects that still point at the legacy global
    # workspace.  Explicit custom save paths are left untouched.
    from flashstudio.constants import DEFAULT_SAVE_DIR
    from flashstudio.components.project_manager import get_project_runs_dir

    saved_save_dir = saved.get("save_dir")
    legacy_workspace = (
        saved_save_dir
        and os.path.basename(os.path.normpath(saved_save_dir)) == "flashstudio_workspaces"
        and os.path.abspath(saved_save_dir) != os.path.abspath(DEFAULT_SAVE_DIR)
    )
    target = (
        saved_save_dir
        if saved_save_dir and saved_save_dir != DEFAULT_SAVE_DIR and not legacy_workspace
        else get_project_runs_dir(project_id)
    )
    current = st.session_state.get("save_dir")
    if not current or current == DEFAULT_SAVE_DIR or legacy_workspace:
        st.session_state["save_dir"] = target
        mirror = st.session_state.get("_config_mirror")
        if isinstance(mirror, dict) and mirror.get("save_dir") in (DEFAULT_SAVE_DIR, saved_save_dir):
            mirror.pop("save_dir", None)


def _render_footer():
    try:
        from flashstudio import __version__ as _fs_version
        st.markdown(
            f'<div class="app-footer">FlashStudio v{_fs_version} · Apache-2.0</div>',
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def _init_config_mirror():
    """Build initial _config_mirror from loaded session state."""
    from flashstudio.utils.config import MIRROR_KEYS

    mirror = {}
    for k in MIRROR_KEYS:
        if k in st.session_state:
            mirror[k] = st.session_state[k]
    if mirror:
        st.session_state["_config_mirror"] = mirror


def main():
    import traceback as _tb

    inject_custom_css()

    if "current_step" not in st.session_state:
        st.session_state["current_step"] = 0

    # Project system — check if user has an active project
    active_project = get_active_project()
    projects = list_projects()

    if active_project is None:
        if not projects:
            # First time user — show project creation
            _render_first_time_setup()
            return
        # Projects exist but none is active (e.g. index was edited or a
        # project was removed) — let the user pick one instead of rendering
        # the workspace without project context.
        render_project_manager_page()
        _render_footer()
        return

    # Load project state once on initial page load
    if active_project and not st.session_state.get("_project_state_loaded"):
        load_project_state(active_project["id"])
        st.session_state["_project_state_loaded"] = True
        # Rebuild the mirror from this project's values instead of inheriting
        # a previous project's stale config.
        st.session_state.pop("_config_mirror", None)
        _init_config_mirror()

    # Re-seed any config keys that Streamlit's widget lifecycle may have removed
    if active_project:
        _ensure_config_in_session(active_project["id"])

    # Keep the sidebar/dashboard status current even when the user is not on
    # the Training page when a background run finishes or fails.
    try:
        from flashstudio.pages.training.page import _check_training_process

        _check_training_process()
    except Exception:
        pass

    if st.session_state.get("show_project_manager"):
        render_sidebar()
        if st.button("← Back to Workspace", key="back_from_projects"):
            st.session_state["show_project_manager"] = False
            st.rerun()
        render_project_manager_page()
        _render_footer()
        return

    current = STEPS[st.session_state["current_step"]]

    # Track page transitions so a page can reset its own sub-navigation when
    # the user arrives from elsewhere (e.g. Training always opens on Launch).
    previous_page = st.session_state.get("_active_page")
    st.session_state["_active_page"] = current["id"]
    st.session_state["_page_changed"] = previous_page != current["id"]

    try:
        PAGE_RENDERERS[current["id"]]()
    except Exception as _e:
        st.error(f"Error: {_e}")
        st.code(_tb.format_exc())

    render_navigation(STEPS, st.session_state["current_step"])

    # Render the sidebar after the page so its status card reflects values
    # changed on this run (framework, dataset, training state) instead of
    # showing last run's values for one extra interaction.
    render_sidebar()

    # Auto-save project state on each page render
    if active_project:
        save_project_state()

    _render_footer()


def _render_first_time_setup():
    """First-time project creation."""
    st.markdown(
        '<div style="max-width:760px;margin:2.5rem auto 1.2rem;text-align:center;">'
        '<div class="fs-brand-mark" style="margin:0 auto .8rem;width:54px;height:54px;'
        'font-size:1.1rem;">FS</div>'
        '<div class="page-kicker" style="justify-content:center;">Welcome</div>'
        '<h1 class="page-title" style="margin:.2rem 0 .2rem;">Create your first project</h1>'
        '<div class="page-subtitle" style="margin:0 auto;">'
        'A project keeps datasets, model configuration, training runs, exports, and '
        'inference results together.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col_form, col_info = st.columns([3, 2])
        with col_form:
            st.markdown("#### Project details")
            name = st.text_input("Project name", placeholder="e.g. Traffic Detection",
                                 key="first_project_name")
            desc = st.text_input("Description (optional)",
                                 placeholder="Detect vehicles in traffic footage",
                                 key="first_project_desc")
            if st.button("Create and start", type="primary",
                         disabled=not name, use_container_width=True):
                project = create_project(name, desc)
                _clear_session_for_new_project()
                load_project_state(project["id"])
                st.session_state["_project_state_loaded"] = True
                st.session_state.pop("_config_mirror", None)
                _init_config_mirror()
                st.rerun()
        with col_info:
            st.markdown("#### What you get")
            st.caption("Every project is isolated and portable.")
            for item in [
                "Dataset management and annotation repair",
                "Framework-specific model configuration",
                "Training runs with live metrics",
                "Fusion, ONNX, INT8, and benchmarks",
                "Image, video, and live inference",
            ]:
                st.markdown(
                    f'<div style="font-size:.83rem;color:var(--fs-body);padding:.18rem 0;">'
                    f'<span style="color:#059669;font-weight:700;">✓</span> {item}</div>',
                    unsafe_allow_html=True,
                )

    _render_footer()


if __name__ == "__main__":
    main()
