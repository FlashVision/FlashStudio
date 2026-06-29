"""Monitor tab — browse and inspect training runs."""

import os
import streamlit as st

from flashstudio.constants import AUTOREFRESH_INTERVAL_MS, TRAINING_LOG_CSV
from flashstudio.pages.training._common import _get_save_dir
from flashstudio.pages.training.monitor.run_meta import _get_run_meta
from flashstudio.pages.training.monitor.dashboard import _render_run_dashboard


def _render_monitor_tab():
    """Tab: monitor an existing training run — choose workspace, browse runs."""
    # Auto-refresh toggle
    auto_refresh = st.session_state.get("monitor_auto_refresh", False)
    is_training = st.session_state.get("training_active", False) and not st.session_state.get("training_paused", False)
    if is_training or auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=AUTOREFRESH_INTERVAL_MS, limit=None, key="monitor_autorefresh")
        except ImportError:
            pass

    # Monitor workspace — defaults to Save Dir from Model > Advanced
    save_dir = _get_save_dir()
    if "monitor_workspace" not in st.session_state:
        st.session_state["monitor_workspace"] = save_dir

    wc1, wc2, wc3, wc4 = st.columns([5, 1, 1, 1])
    with wc1:
        workspace = st.text_input("Workspace Path", key="monitor_workspace",
                                  label_visibility="collapsed")
    with wc2:
        if st.button("Sync", key="mon_sync_btn", use_container_width=True,
                      help="Sync with Save Dir from Model > Advanced"):
            st.session_state["monitor_workspace"] = save_dir
            st.rerun()
    with wc3:
        if st.button("Refresh", key="mon_refresh_btn", use_container_width=True):
            st.rerun()
    with wc4:
        st.toggle("Auto", value=is_training, key="monitor_auto_refresh")

    if not os.path.isdir(workspace):
        st.warning("Workspace not found")
        _render_run_dashboard(None)
        return

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

    # Collect training run folders (max 2 levels deep)
    all_folders = []
    for d in sorted(os.listdir(workspace)):
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

    if not all_folders:
        st.info("No runs found. Start training first.")
        _render_run_dashboard(None)
        return

    # Sort by modification time (most recent first)
    all_folders.sort(
        key=lambda d: os.path.getmtime(os.path.join(workspace, d)),
        reverse=True,
    )

    labels = []
    for r in all_folders:
        meta = _get_run_meta(os.path.join(workspace, r))
        parts = [r]
        if meta["mAP"]:
            parts.append(f"mAP={meta['mAP']:.3f}")
        if meta["size"]:
            parts.append(meta["size"])
        labels.append(" | ".join(parts))

    sc1, sc2 = st.columns([7, 1])
    with sc1:
        selected_idx = st.selectbox("Select Run", range(len(all_folders)),
                                    format_func=lambda i: labels[i],
                                    key="selected_run_idx", label_visibility="collapsed")
    selected_run = all_folders[selected_idx]
    run_dir = os.path.join(workspace, selected_run)
    with sc2:
        if st.button("Del", key="delete_run_btn", help="Delete this run"):
            st.session_state["show_delete_run"] = selected_run

    if st.session_state.get("show_delete_run"):
        dc1, dc2 = st.columns([1, 1])
        with dc1:
            if st.button(f"Confirm delete {st.session_state['show_delete_run']}", type="primary",
                         key="confirm_delete_run"):
                import shutil
                from flashstudio.utils import flash
                deleted_name = st.session_state["show_delete_run"]
                shutil.rmtree(os.path.join(workspace, deleted_name), ignore_errors=True)
                st.session_state.pop("show_delete_run", None)
                flash(f"Deleted run `{deleted_name}`", "success")
                st.rerun()
        with dc2:
            if st.button("Cancel", key="cancel_delete_run"):
                st.session_state.pop("show_delete_run", None)
                st.rerun()

    meta = _get_run_meta(run_dir)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Status", meta["status"][:10])
    with m2:
        st.metric("Model", (meta["model"] or "—")[:12])
    with m3:
        st.metric("Epochs", meta["epochs"])
    with m4:
        st.metric("mAP@50", f"{meta['mAP']:.3f}" if meta["mAP"] else "—")
    with m5:
        st.metric("Size", meta["size"] or "—")
    with m6:
        st.metric("Date", meta["date"][:8])

    _render_run_dashboard(run_dir)
