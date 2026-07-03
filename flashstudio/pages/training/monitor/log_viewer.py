"""Training log viewer."""

import os
import streamlit as st


def _render_full_log(log_path):
    """Log viewer with refresh and error detection."""
    if not log_path or not os.path.isfile(log_path):
        st.info("No log found.")
        return

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    lines = content.strip().split("\n") if content.strip() else []

    # Detect the LAST error block (Traceback + exception)
    error_lines = []
    last_tb_start = -1
    for i, line in enumerate(lines):
        if "Traceback" in line:
            last_tb_start = i
    if last_tb_start >= 0:
        error_lines = lines[last_tb_start:]
    has_error = bool(error_lines)

    lc1, lc2, lc3 = st.columns([4, 1, 1])
    with lc1:
        st.caption(f"`{os.path.basename(log_path)}` — {len(lines)} lines")
    with lc2:
        show_all = st.checkbox("Full", value=False, key="show_full_log")
    with lc3:
        if st.button("Refresh", key="refresh_log_btn", use_container_width=True):
            st.rerun()

    if has_error and error_lines:
        st.error("Training failed with error:")
        st.code("\n".join(error_lines[-20:]), language="python")

    if lines:
        st.code("\n".join(lines if show_all else lines[-30:]), language="bash")
    else:
        st.warning("Log file is empty. Training may still be starting...")
