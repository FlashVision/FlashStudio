"""Shared helpers for the training page."""

import os
import glob as glob_module
import streamlit as st
from flashstudio.constants import TRAINING_LOG_GLOB


def _get_save_dir():
    """Training output folder: Model → Advanced → Save Dir.

    Falls back to the project's ``runs`` folder (or the user's configured
    storage location) instead of the legacy global workspace.
    """
    from flashstudio.constants import DEFAULT_SAVE_DIR
    from flashstudio.utils import get_state
    from flashstudio.utils.storage import default_runs_root

    val = get_state("save_dir")
    if not val or val == DEFAULT_SAVE_DIR:
        return default_runs_root()
    return val


def _find_log_file(run_dir: str):
    """Find the best training log file — prefer the largest (stdout-redirected has more data)."""
    logs = glob_module.glob(os.path.join(run_dir, TRAINING_LOG_GLOB))
    if not logs:
        return None
    return max(logs, key=lambda p: os.path.getsize(p))


def _training_display_status() -> str:
    """Current status, preferring the live process over a stale stored status.

    Session state keeps the last run's status (e.g. "Complete") until a new
    process updates it.  When training is actually alive we must report that
    instead of the previous run's result.
    """
    if st.session_state.get("training_active"):
        return "Paused" if st.session_state.get("training_paused") else "Running"
    return str(st.session_state.get("training_status") or "Not started")
