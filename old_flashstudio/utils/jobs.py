"""Background job tracker — tracks training and inference processes across page navigation."""

import os
import signal
import time
import streamlit as st


def _get_jobs() -> list:
    """Get the list of active jobs from session state."""
    if "active_jobs" not in st.session_state:
        st.session_state["active_jobs"] = []
    return st.session_state["active_jobs"]


def add_job(pid: int, job_type: str, name: str, details: dict | None = None):
    """Register a new background job."""
    jobs = _get_jobs()
    jobs.append({
        "pid": pid,
        "type": job_type,
        "name": name,
        "started": time.time(),
        "details": details or {},
    })
    st.session_state["active_jobs"] = jobs


def remove_job(pid: int):
    """Remove a job by PID."""
    jobs = _get_jobs()
    st.session_state["active_jobs"] = [j for j in jobs if j["pid"] != pid]


def get_jobs_by_type(job_type: str) -> list:
    """Get all jobs of a given type (e.g., 'training', 'inference')."""
    return [j for j in _get_jobs() if j["type"] == job_type]


def is_pid_alive(pid: int) -> bool:
    """Check if a process is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def cleanup_dead_jobs():
    """Remove jobs whose PIDs are no longer running."""
    jobs = _get_jobs()
    alive = [j for j in jobs if is_pid_alive(j["pid"])]
    if len(alive) != len(jobs):
        st.session_state["active_jobs"] = alive


def stop_job(pid: int) -> bool:
    """Stop a job by sending SIGTERM, then SIGKILL if needed."""
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.5)
        if is_pid_alive(pid):
            os.kill(pid, signal.SIGKILL)
        remove_job(pid)
        return True
    except (ProcessLookupError, PermissionError):
        remove_job(pid)
        return False


def stop_all_by_type(job_type: str):
    """Stop all jobs of a given type."""
    for job in get_jobs_by_type(job_type):
        stop_job(job["pid"])


def get_active_count() -> dict:
    """Return count of active jobs by type."""
    cleanup_dead_jobs()
    jobs = _get_jobs()
    counts = {}
    for j in jobs:
        counts[j["type"]] = counts.get(j["type"], 0) + 1
    return counts


def format_job_duration(started: float) -> str:
    """Format elapsed time since job started."""
    elapsed = time.time() - started
    if elapsed < 60:
        return f"{elapsed:.0f}s"
    elif elapsed < 3600:
        return f"{elapsed / 60:.0f}m"
    else:
        return f"{elapsed / 3600:.1f}h"
