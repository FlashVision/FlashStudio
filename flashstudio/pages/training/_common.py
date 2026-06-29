"""Shared helpers for the training page."""

import os
import glob as glob_module
import streamlit as st
from flashstudio.constants import TRAINING_LOG_CSV, TRAINING_LOG_GLOB


def _get_save_dir():
    """Single source of truth: Model > Advanced > Save Dir. Used across the whole project."""
    from flashstudio.utils import DEFAULTS
    val = st.session_state.get("save_dir", "")
    if not val:
        val = DEFAULTS["save_dir"]
        st.session_state["save_dir"] = val
    return val


def _find_log_file(run_dir: str):
    """Find the best training log file — prefer the largest (stdout-redirected has more data)."""
    logs = glob_module.glob(os.path.join(run_dir, TRAINING_LOG_GLOB))
    if not logs:
        return None
    return max(logs, key=lambda p: os.path.getsize(p))
