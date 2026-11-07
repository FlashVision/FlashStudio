"""Monitor tab — browse and inspect training runs."""

import os
import streamlit as st

from flashstudio.constants import TRAINING_LOG_CSV
from flashstudio.pages.training._common import (
    _get_save_dir, _training_display_status,
)
from flashstudio.pages.training.monitor.run_meta import _get_run_meta
from flashstudio.pages.training.monitor.dashboard import (
    _load_run_data, _render_metrics_from_history, _render_run_dashboard,
)
from flashstudio.pages.training.progress import _render_training_progress


def _is_training_run(folder_path):
    """Check if a folder looks like a FlashDet training run."""
    try:
        entries = os.listdir(folder_path)
    except OSError:
        return False
    has_log = any(f.startswith("train_") and f.endswith(".log") for f in entries)
    has_csv = TRAINING_LOG_CSV in entries
    has_ckpt = any(f.endswith(".pth") for f in entries)
    return has_log or has_csv or has_ckpt


def _list_run_folders(workspace):
    """Collect training run folders (max 2 levels deep), newest first."""
    if not workspace or not os.path.isdir(workspace):
        return []

    all_folders = []
    try:
        entries = sorted(os.listdir(workspace))
    except OSError:
        return []
    for d in entries:
        full = os.path.join(workspace, d)
        if os.path.isdir(full):
            if _is_training_run(full):
                all_folders.append(d)
            else:
                try:
                    for sd in sorted(os.listdir(full)):
                        sf = os.path.join(full, sd)
                        if os.path.isdir(sf) and _is_training_run(sf):
                            all_folders.append(os.path.join(d, sd))
                except OSError:
                    pass

    all_folders.sort(
        key=lambda d: os.path.getmtime(os.path.join(workspace, d)),
        reverse=True,
    )
    return all_folders


def _selected_run_dir(workspace=None, folders=None):
    """Resolve the selected run from session state without rendering widgets."""
    workspace = (
        workspace
        or st.session_state.get("monitor_workspace")
        or _get_save_dir()
    )
    if folders is None:
        folders = _list_run_folders(workspace)
    if not folders:
        return None, workspace, folders
    idx = st.session_state.get("selected_run_idx", 0)
    if not isinstance(idx, int) or idx < 0 or idx >= len(folders):
        idx = 0
    return os.path.join(workspace, folders[idx]), workspace, folders


def _render_run_meta_cards(run_dir):
    """Render the six run summary metrics and return the run meta."""
    meta = _get_run_meta(run_dir)
    # The file-based meta can say "In Progress" for a run that was stopped.
    # Prefer the live process status when this is the active run.
    active_run = st.session_state.get("active_run_path") or ""
    if active_run and os.path.abspath(str(active_run)) == os.path.abspath(str(run_dir)):
        meta["status"] = _training_display_status()
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Status", meta["status"])
    with m2:
        st.metric("Model", meta["model"] or "—")
    with m3:
        st.metric("Epochs", meta["epochs"])
    with m4:
        st.metric("mAP@0.5", f"{meta['mAP']:.3f}" if meta["mAP"] else "—")
    with m5:
        st.metric("Size", meta["size"] or "—")
    with m6:
        st.metric("Date", meta["date"] or "—")
    return meta


def _render_selected_run_summary():
    """Read-only status summary of the selected run (used on the Launch tab)."""
    run_dir, workspace, _folders = _selected_run_dir()
    if not run_dir:
        st.info(
            "No runs found in this workspace yet. "
            "Create a run above to start training."
        )
        return

    st.caption(f"Selected run · **{os.path.basename(run_dir)}**")
    _render_run_meta_cards(run_dir)
    _log_file, history = _load_run_data(run_dir)
    _render_training_progress(run_dir, show_eta=False)
    _render_metrics_from_history(history, run_dir)


def _sync_workspace(save_dir):
    st.session_state["monitor_workspace"] = save_dir


def _delete_run(workspace, run_name):
    import shutil

    from flashstudio.utils import flash

    shutil.rmtree(os.path.join(workspace, run_name), ignore_errors=True)
    st.session_state.pop("show_delete_run", None)
    flash(f"Deleted run `{run_name}`", "success")


def _render_monitor_tab(embedded: bool = False, show_summary: bool = True):
    """Monitor a training run — choose workspace, browse runs.

    ``embedded`` renders the same dashboard inside the Launch view: the empty
    state then points at the Start button instead of a "Go to Launch" button.
    ``show_summary`` toggles the run-level metric cards; the Launch tab already
    shows them, so the Display tab skips them to save vertical space.
    """
    # Auto-refresh component is registered by the Training page so it keeps
    # running even when the Display tab body is not rendered.
    is_training = st.session_state.get("training_active", False) and not st.session_state.get("training_paused", False)

    # Monitor workspace — defaults to Save Dir from Model > Advanced
    save_dir = _get_save_dir()
    if "monitor_workspace" not in st.session_state:
        st.session_state["monitor_workspace"] = save_dir

    wc1, wc2, wc3, wc4 = st.columns([5, 1, 1, 1])
    with wc1:
        workspace = st.text_input("Workspace Path", key="monitor_workspace",
                                  label_visibility="collapsed")
    with wc2:
        st.button(
            "Sync", key="mon_sync_btn", use_container_width=True,
            help="Sync with Save Dir from Model > Advanced",
            on_click=_sync_workspace, args=(save_dir,),
        )
    with wc3:
        # The click itself reruns the script; calling st.rerun here would
        # abort inside the tab body and duplicate the tab component.
        st.button("Refresh", key="mon_refresh_btn", use_container_width=True)
    with wc4:
        if "monitor_auto_refresh" not in st.session_state:
            st.session_state["monitor_auto_refresh"] = is_training
        st.toggle("Auto", key="monitor_auto_refresh")

    if not os.path.isdir(workspace):
        st.warning("Workspace not found")
        _render_run_dashboard(None)
        return

    all_folders = _list_run_folders(workspace)

    if not all_folders:
        if embedded:
            st.info("No run data yet — start training to see live metrics here.")
        else:
            st.info("No runs found. Start training first.")
        _render_run_dashboard(None)
        return

    labels = []
    for r in all_folders:
        meta = _get_run_meta(os.path.join(workspace, r))
        parts = [r]
        if meta["mAP"]:
            parts.append(f"mAP={meta['mAP']:.3f}")
        if meta["size"]:
            parts.append(meta["size"])
        labels.append(" · ".join(parts))

    sc1, sc2 = st.columns([7, 1])
    with sc1:
        selected_idx = st.selectbox("Select Run", range(len(all_folders)),
                                    format_func=lambda i: labels[i],
                                    key="selected_run_idx", label_visibility="collapsed")
    selected_run = all_folders[selected_idx]
    run_dir = os.path.join(workspace, selected_run)
    with sc2:
        if st.button("Delete", key="delete_run_btn", help="Delete this run"):
            st.session_state["show_delete_run"] = selected_run

    if st.session_state.get("show_delete_run"):
        dc1, dc2 = st.columns([1, 1])
        with dc1:
            st.button(
                f"Confirm delete {st.session_state['show_delete_run']}",
                type="primary", key="confirm_delete_run",
                on_click=_delete_run,
                args=(workspace, st.session_state["show_delete_run"]),
            )
        with dc2:
            if st.button("Cancel", key="cancel_delete_run"):
                st.session_state.pop("show_delete_run", None)

    if show_summary:
        _render_run_meta_cards(run_dir)
    _render_run_dashboard(run_dir, show_summary=show_summary)
