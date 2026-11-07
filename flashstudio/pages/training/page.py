"""FlashStudio — Training Dashboard Page (reads real FlashDet workspace output)."""

import os
import streamlit as st

from flashstudio.constants import COMPLETE_MARKERS
from flashstudio.pages.training._common import _training_display_status
from flashstudio.pages.training.processes import (
    clear_pid_file, find_run_training_pids, pid_is_alive, pid_is_paused,
)
from flashstudio.pages.training.launch.tab import _render_start_tab
from flashstudio.pages.training.monitor.tab import (
    _list_run_folders, _render_monitor_tab, _render_selected_run_summary,
)
from flashstudio.pages.training.compare.tab import _render_compare_tab


TRAINING_TABS = ["Launch", "Display", "Compare"]


def _training_tabs(labels):
    """Render the training workflow tabs.

    Mirrors the Data page: newer Streamlit releases can keep the active tab in
    session state, which lets other sections jump straight to a tab. Older
    releases fall back to plain tabs.
    """
    import inspect

    next_tab = st.session_state.pop("_training_tab_next", None)
    params = inspect.signature(st.tabs).parameters
    if "default" not in params or "on_change" not in params:
        return st.tabs(labels)

    if next_tab in labels:
        st.session_state["_training_tab"] = next_tab
    selected = st.session_state.get("_training_tab")
    if selected not in labels:
        selected = labels[0]
    st.session_state["_training_tab"] = selected
    return st.tabs(labels, key="_training_tab", default=selected, on_change="rerun")


def _pid_is_alive(pid: int) -> bool:
    """Return True only for a live process (zombies count as dead on Linux)."""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            state = f.read().split()[2]
        if state == "Z":
            return False
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _restore_training_state():
    """Re-attach to a training process the session lost track of.

    The Popen handle (and ``training_pid``) can disappear on app restart or
    project switch while the trainer keeps running. Recover the run's live PID
    so the status banner and Stop control stay available.
    """
    if st.session_state.get("training_active"):
        return

    run_dir = st.session_state.get("active_run_path") or ""
    if not run_dir or not os.path.isdir(run_dir):
        workspace = (
            st.session_state.get("monitor_workspace")
            or st.session_state.get("save_dir")
            or ""
        )
        folders = _list_run_folders(workspace)
        if not folders:
            return
        idx = st.session_state.get("selected_run_idx", 0)
        if not isinstance(idx, int) or idx < 0 or idx >= len(folders):
            idx = 0
        run_dir = os.path.join(workspace, folders[idx])

    pids = find_run_training_pids(run_dir)
    if not pids:
        return

    from flashstudio.pages.training.monitor.parsers import _find_log_file

    pid = pids[0]
    st.session_state["training_active"] = True
    st.session_state["training_paused"] = pid_is_paused(pid)
    st.session_state["training_pid"] = pid
    st.session_state["training_status"] = "Running"
    st.session_state["active_run_path"] = run_dir
    log_file = _find_log_file(run_dir)
    if log_file:
        st.session_state["training_log_file"] = log_file


def _check_training_process():
    """Check if the training subprocess is still alive; extract errors if it crashed."""
    _restore_training_state()
    if not st.session_state.get("training_active"):
        return

    proc = st.session_state.get("_training_process")
    exit_code = None
    stale = False
    if proc is not None:
        exit_code = proc.poll()
        if exit_code is None:
            return
    else:
        # Session was reloaded and the Popen object is gone; fall back to the
        # PID check (best effort -- zombies cannot be reaped from here).
        pid = st.session_state.get("training_pid")
        if pid and _pid_is_alive(pid):
            return
        run_dir = st.session_state.get("active_run_path") or ""
        if not pid and run_dir and find_run_training_pids(run_dir):
            return
        # No process handle and no live PID: this is a stale "active" flag
        # left over from an earlier session, not a crash.
        stale = True

    log_path = st.session_state.get("training_log_file", "")
    error_lines = []
    last_lines = []
    if log_path and os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            last_lines = all_lines[-30:] if len(all_lines) > 30 else all_lines

            in_traceback = False
            for line in all_lines:
                stripped = line.strip()
                if "Traceback" in stripped:
                    in_traceback = True
                    error_lines = [stripped]
                elif in_traceback:
                    error_lines.append(stripped)
                elif stripped.startswith("[") and "[ERROR]" in stripped:
                    error_lines.append(stripped)
                elif stripped.startswith("ERROR:") or stripped.startswith("RuntimeError:"):
                    error_lines.append(stripped)
        except OSError:
            pass

    st.session_state["training_active"] = False
    st.session_state["training_pid"] = None
    st.session_state["_training_process"] = None
    fh = st.session_state.pop("_log_file_handle", None)
    if fh and not fh.closed:
        try:
            fh.close()
        except Exception:
            pass

    run_path = st.session_state.get("active_run_path", "")
    completed = any(
        os.path.isfile(os.path.join(run_path, m)) for m in COMPLETE_MARKERS
    ) if run_path else False
    if not completed and last_lines:
        completed = any("Training Complete!" in line for line in last_lines)

    if stale:
        st.session_state["training_status"] = "Complete" if completed else "Stopped"
        return

    if exit_code == 0 and completed:
        st.session_state["training_status"] = "Complete"
    elif exit_code == 0:
        st.session_state["training_status"] = "Stopped"
        tail = "".join(last_lines[-5:]).strip() if last_lines else "No log output"
        st.session_state["training_error"] = (
            "Training process exited without producing final weights.\n\n"
            f"Last log output:\n{tail}"
        )
    elif error_lines:
        st.session_state["training_status"] = "Failed"
        st.session_state["training_error"] = "\n".join(error_lines[-20:])
    else:
        tail = "".join(last_lines[-8:]).strip() if last_lines else "No log output"
        st.session_state["training_status"] = "Failed"
        st.session_state["training_error"] = (
            f"Training process exited with code {exit_code}.\n\nLast log output:\n{tail}"
        )


def _ensure_autorefresh():
    """Keep live progress updating even when Display's body is not rendered."""
    from flashstudio.constants import AUTOREFRESH_INTERVAL_MS

    is_training = bool(st.session_state.get("training_active")) and not bool(
        st.session_state.get("training_paused")
    )
    if is_training or st.session_state.get("monitor_auto_refresh", False):
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(
                interval=AUTOREFRESH_INTERVAL_MS,
                limit=None,
                key="monitor_autorefresh",
            )
        except ImportError:
            pass


def _render_status_banner():
    """Show one accurate, live status card above the tabs, with run controls."""
    from flashstudio.pages.training.launch.controls import (
        _pause_training, _resume_active_training, _stop_training,
    )

    status = _training_display_status()
    active = bool(st.session_state.get("training_active"))
    paused = bool(st.session_state.get("training_paused"))
    pid = st.session_state.get("training_pid")

    if active:
        message_col, control_col, stop_col = st.columns([6, 1, 1.3])
        with message_col:
            if paused:
                st.warning(f"Training paused · PID {pid}")
            else:
                st.success(f"Training in progress · PID {pid}")
        with control_col:
            if paused:
                if st.button(
                    "Resume", key="banner_resume_training",
                    use_container_width=True,
                ):
                    _resume_active_training()
            else:
                if st.button(
                    "Pause", key="banner_pause_training",
                    use_container_width=True,
                ):
                    _pause_training()
        with stop_col:
            if st.button(
                "Stop training", key="banner_stop_training",
                type="primary", use_container_width=True,
            ):
                _stop_training()
    elif status == "Complete":
        st.success("Last run complete")
    elif status == "Starting":
        st.info("Starting training…")
    elif status == "Failed":
        st.error("Last run failed")
    elif status == "Stopped":
        st.warning("Last run stopped")


def _render_launch_tab():
    """Launch tab — launch controls, with the selected run's summary in place
    of the "create a run" blocker when no new run has been created yet."""
    _render_start_tab(empty_hint=_render_selected_run_summary)


def _render_display_tab():
    """Display tab — curves, visualizations, ground truth, log, and checkpoints.

    The run-level status/metric cards stay on the Launch tab; skipping them
    here keeps the Display tab focused on the five content views.
    """
    _render_monitor_tab(embedded=True, show_summary=False)


def _render_compare_tab_section():
    """Compare tab — compare metrics across runs."""
    _render_compare_tab()


def render_training_page():
    """Render training dashboard — ultra-compact."""
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Training")

    show_flashes()

    if "training_active" not in st.session_state:
        st.session_state["training_active"] = False
    if "training_status" not in st.session_state:
        st.session_state["training_status"] = "Not started"
    if "training_paused" not in st.session_state:
        st.session_state["training_paused"] = False

    # Handle a queued start before rendering anything. The Start button uses
    # on_click so this runs in a fresh script pass, and _start_training's rerun
    # happens outside any tab body.
    pending_start = st.session_state.pop("_start_training_pending", None)
    if pending_start:
        st.session_state["training_active"] = True
        st.session_state["training_status"] = "Starting"
        from flashstudio.pages.training.launch.runner import _start_training
        _start_training()

    # Controls inside the tabs queue their action; run it here, before any tab
    # component is rendered, so the rerun cannot leave a duplicate tab body.
    training_action = st.session_state.pop("_training_action", None)
    if training_action:
        kind, payload = training_action
        from flashstudio.pages.training.launch.controls import (
            _stop_training, _stop_run, _pause_training,
            _resume_active_training,
        )
        if kind == "stop":
            _stop_training()
        elif kind == "pause":
            _pause_training()
        elif kind == "resume":
            _resume_active_training()
        elif kind == "stop_run":
            _stop_run(payload)

    _ensure_autorefresh()
    _check_training_process()

    training_err = st.session_state.pop("training_error", None)
    if training_err:
        lines = training_err.strip().split("\n")
        error_summary = ""
        for line in reversed(lines):
            stripped = line.strip()
            if "Error:" in stripped or "Exception:" in stripped:
                error_summary = stripped
                break
            if stripped.startswith("[") and "[ERROR]" in stripped:
                error_summary = stripped
                break
        if error_summary:
            st.error(f"**Training failed:** {error_summary}")
        else:
            st.error("**Training failed** — see the log below.")
        with st.expander("Error log", expanded=True):
            st.code(training_err, language="text")

    _render_status_banner()

    # Same layout as the Data and Model pages: one tab bar at the top instead
    # of stacked clusters. Only the selected tab body is rendered: the Display
    # tab parses logs and encodes GT images, so rendering it on every Launch
    # rerun made the page feel stale during Start.
    tab_launch, tab_display, tab_compare = _training_tabs(TRAINING_TABS)
    selected_tab = st.session_state.get("_training_tab", TRAINING_TABS[0])

    with tab_launch:
        if selected_tab == "Launch":
            _render_launch_tab()

    with tab_display:
        if selected_tab == "Display":
            _render_display_tab()

    with tab_compare:
        if selected_tab == "Compare":
            _render_compare_tab_section()
