"""FlashStudio — Training Dashboard Page (reads real FlashDet workspace output)."""

import os
import streamlit as st

from flashstudio.constants import COMPLETE_MARKERS
from flashstudio.pages.training.launch.tab import _render_start_tab
from flashstudio.pages.training.monitor.tab import _render_monitor_tab


def _check_training_process():
    """Check if the training subprocess is still alive; extract errors if it crashed."""
    pid = st.session_state.get("training_pid")
    if not pid or not st.session_state.get("training_active"):
        return

    import signal
    try:
        os.kill(pid, 0)
        return
    except (ProcessLookupError, PermissionError):
        pass

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
                elif "Error:" in stripped or "Exception:" in stripped:
                    error_lines.append(stripped)
        except OSError:
            pass

    st.session_state["training_active"] = False
    st.session_state["training_pid"] = None
    fh = st.session_state.pop("_log_file_handle", None)
    if fh and not fh.closed:
        try:
            fh.close()
        except Exception:
            pass

    if error_lines:
        st.session_state["training_status"] = "Failed"
        st.session_state["training_error"] = "\n".join(error_lines[-15:])
    else:
        complete_markers = COMPLETE_MARKERS
        run_path = st.session_state.get("active_run_path", "")
        completed = any(
            os.path.isfile(os.path.join(run_path, m)) for m in complete_markers
        ) if run_path else False
        if not completed and last_lines:
            completed = any("Training Complete!" in l for l in last_lines)
        if completed:
            st.session_state["training_status"] = "Complete"
        else:
            tail = "".join(last_lines[-5:]).strip() if last_lines else "No log output"
            st.session_state["training_status"] = "Stopped"
            st.session_state["training_error"] = f"Training process exited.\n\nLast log output:\n{tail}"


def render_training_page():
    """Render training dashboard — ultra-compact."""
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Training")

    show_flashes()

    if "training_active" not in st.session_state:
        st.session_state["training_active"] = False
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
            st.error(f"**Training Error:** {error_summary}")
        else:
            st.error("**Training failed** — see log details below")
        with st.expander("Full error log", expanded=True):
            st.code(training_err, language="text")

    is_active = st.session_state.get("training_active", False)
    pid = st.session_state.get("training_pid")
    status = st.session_state.get("training_status", "")
    if is_active:
        if st.session_state.get("training_paused"):
            st.warning(f"Paused (PID {pid})")
        else:
            st.success(f"Running (PID {pid})")
    elif status == "Failed":
        st.error(f"Training failed")
    elif status == "Complete":
        st.success("Training complete")

    tab_start, tab_monitor = st.tabs(["Launch", "Monitor"])

    with tab_start:
        _render_start_tab()
    with tab_monitor:
        _render_monitor_tab()
