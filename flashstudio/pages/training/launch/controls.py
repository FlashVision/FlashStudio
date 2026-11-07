"""Training process controls — stop, pause, resume."""

import os
import signal
import streamlit as st


def _process_pid() -> int | None:
    proc = st.session_state.get("_training_process")
    if proc is not None:
        return proc.pid
    return st.session_state.get("training_pid")


def _signal_process_group(pid: int, sig: int):
    """Send a signal to the training process group, falling back to the PID."""
    try:
        os.killpg(os.getpgid(pid), sig)
    except (OSError, ProcessLookupError):
        os.kill(pid, sig)


def _stop_training():
    """Stop the training subprocess by killing its process tree."""
    from flashstudio.utils import flash

    proc = st.session_state.get("_training_process")
    pid = _process_pid()
    if proc is not None and proc.poll() is None:
        try:
            _signal_process_group(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except Exception:
                _signal_process_group(proc.pid, signal.SIGKILL)
            flash(f"Training stopped (PID {proc.pid})", "success")
        except ProcessLookupError:
            flash("Training process already finished", "info")
        except Exception as e:
            flash(f"Error stopping training: {e}", "error")
    elif pid:
        try:
            _signal_process_group(pid, signal.SIGTERM)
            flash(f"Training stopped (PID {pid})", "success")
        except ProcessLookupError:
            flash("Training process already finished", "info")
        except Exception as e:
            flash(f"Error stopping training: {e}", "error")
    else:
        flash("No training process to stop", "warning")

    st.session_state["_training_process"] = None
    st.session_state["training_active"] = False
    st.session_state["training_status"] = "Stopped"
    st.session_state["training_paused"] = False
    st.session_state["training_pid"] = None
    fh = st.session_state.pop("_log_file_handle", None)
    if fh and not fh.closed:
        try:
            fh.close()
        except Exception:
            pass
    st.rerun()


def _pause_training():
    """Pause the training subprocess by sending SIGSTOP."""
    from flashstudio.utils import flash

    proc = st.session_state.get("_training_process")
    pid = _process_pid()
    if proc is not None and proc.poll() is not None:
        st.session_state["training_active"] = False
        st.session_state["training_paused"] = False
        st.session_state["_training_process"] = None
        flash("Training process already finished", "info")
        st.rerun()
        return

    if pid:
        try:
            _signal_process_group(pid, signal.SIGSTOP)
            st.session_state["training_paused"] = True
            st.session_state["training_status"] = "Paused"
            flash(f"Training paused (PID {pid}). GPU memory still held.", "success")
        except ProcessLookupError:
            flash("Training process already finished", "info")
            st.session_state["training_active"] = False
            st.session_state["training_paused"] = False
        except Exception as e:
            flash(f"Could not pause training: {e}", "error")
    else:
        flash("No training process found to pause", "warning")
    st.rerun()


def _resume_active_training():
    """Resume a paused training subprocess by sending SIGCONT."""
    from flashstudio.utils import flash

    proc = st.session_state.get("_training_process")
    pid = _process_pid()
    if proc is not None and proc.poll() is not None:
        st.session_state["training_active"] = False
        st.session_state["training_paused"] = False
        st.session_state["_training_process"] = None
        flash("Training process already finished", "info")
        st.rerun()
        return

    if pid:
        try:
            _signal_process_group(pid, signal.SIGCONT)
            st.session_state["training_paused"] = False
            st.session_state["training_status"] = "Running"
            flash(f"Training resumed (PID {pid})", "success")
        except ProcessLookupError:
            flash("Training process already finished", "info")
            st.session_state["training_active"] = False
            st.session_state["training_paused"] = False
        except Exception as e:
            flash(f"Could not resume training: {e}", "error")
    else:
        flash("No training process found to resume", "warning")
    st.rerun()
