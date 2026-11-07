"""FlashStudio — Training Dashboard Page (reads real FlashDet workspace output)."""

import os
import streamlit as st

from flashstudio.constants import COMPLETE_MARKERS
from flashstudio.pages.training.launch.tab import _render_start_tab
from flashstudio.pages.training.monitor.tab import _render_monitor_tab
from flashstudio.pages.training.compare.tab import _render_compare_tab


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


def _check_training_process():
    """Check if the training subprocess is still alive; extract errors if it crashed."""
    if not st.session_state.get("training_active"):
        return

    proc = st.session_state.get("_training_process")
    exit_code = None
    if proc is not None:
        exit_code = proc.poll()
        if exit_code is None:
            return
    else:
        # Session was reloaded and the Popen object is gone; fall back to the
        # PID check (best effort -- zombies cannot be reaped from here).
        pid = st.session_state.get("training_pid")
        if not pid or _pid_is_alive(pid):
            return

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


def _training_tabs(labels):
    """Stateful training tabs so a running/completed run stays visible."""
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
    return st.tabs(
        labels, key="_training_tab", default=selected, on_change="rerun"
    )


def render_training_page():
    """Render training dashboard — ultra-compact."""
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Training", "Create a run, launch training, and monitor progress")

    show_flashes()

    if "training_active" not in st.session_state:
        st.session_state["training_active"] = False
    if "training_status" not in st.session_state:
        st.session_state["training_status"] = "Not started"
    if "training_paused" not in st.session_state:
        st.session_state["training_paused"] = False

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

    is_active = st.session_state.get("training_active", False)
    pid = st.session_state.get("training_pid")
    status = st.session_state.get("training_status", "")
    if is_active:
        if st.session_state.get("training_paused"):
            st.warning(f"Paused · PID {pid}")
        else:
            st.success(f"Training in progress · PID {pid}")
    elif status == "Failed":
        st.error("Training failed")
    elif status == "Complete":
        st.success("Training complete")

    goto_monitor = st.session_state.pop("_goto_monitor", False)
    is_active = st.session_state.get("training_active", False)
    # Keep results visible after a run finishes; otherwise the tab order flips
    # back to Launch and hides the completed run's metrics.
    monitor_first = bool(
        goto_monitor or is_active
        or status in ("Complete", "Running", "Paused")
    )
    if goto_monitor:
        st.session_state["_training_tab_next"] = "Monitor"

    if monitor_first:
        labels = ["Monitor", "Launch", "Compare"]
    else:
        labels = ["Launch", "Monitor", "Compare"]
    tabs = _training_tabs(labels)
    if monitor_first:
        tab_monitor, tab_start, tab_compare = tabs
    else:
        tab_start, tab_monitor, tab_compare = tabs

    with tab_start:
        _render_start_tab()
    with tab_monitor:
        _render_monitor_tab()
    with tab_compare:
        with st.container(border=True):
            st.markdown("#### Compare runs")
            _render_compare_tab()
