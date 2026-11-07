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
    from flashstudio.pages.training.processes import (
        clear_pid_file, find_run_training_pids,
    )

    proc = st.session_state.get("_training_process")
    pid = _process_pid()
    run_dir = st.session_state.get("active_run_path") or ""
    stopped = []
    errors = []

    def _terminate(target):
        try:
            _signal_process_group(target, signal.SIGTERM)
            stopped.append(target)
        except ProcessLookupError:
            pass
        except Exception as exc:
            errors.append(str(exc))

    if proc is not None and proc.poll() is None:
        _terminate(proc.pid)
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                _signal_process_group(proc.pid, signal.SIGKILL)
            except Exception:
                pass
    elif pid:
        _terminate(pid)

    # Stop any other launcher roots still attached to this run (for example a
    # job started before the app restarted, or a duplicate launch).
    for extra in find_run_training_pids(run_dir):
        if extra not in stopped and extra != pid:
            _terminate(extra)

    for message in errors:
        flash(f"Error stopping training: {message}", "error")
    if stopped:
        flash(f"Training stopped (PID {', '.join(map(str, stopped))})", "success")
    elif not errors:
        flash("No training process to stop", "warning")

    clear_pid_file(run_dir)
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


def _stop_run(run_dir: str):
    """Stop every trainer process attached to an arbitrary run directory."""
    from flashstudio.utils import flash
    from flashstudio.pages.training.processes import (
        clear_pid_file, find_run_training_pids,
    )

    pids = find_run_training_pids(run_dir)
    stopped = []
    for pid in pids:
        try:
            _signal_process_group(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            pass
        except Exception as exc:
            flash(f"Could not stop PID {pid}: {exc}", "error")
    clear_pid_file(run_dir)

    active = str(st.session_state.get("active_run_path") or "")
    if active and os.path.abspath(active) == os.path.abspath(str(run_dir)):
        st.session_state["_training_process"] = None
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Stopped"
        st.session_state["training_paused"] = False
        st.session_state["training_pid"] = None

    if stopped:
        flash(
            f"Stopped `{os.path.basename(run_dir)}` "
            f"(PID {', '.join(map(str, stopped))})",
            "success",
        )
    else:
        flash(
            f"No running process found for `{os.path.basename(run_dir)}`",
            "info",
        )
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
