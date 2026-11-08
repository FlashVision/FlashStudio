"""Regression test: starting training must not regenerate the run name."""

from streamlit.testing.v1 import AppTest


def test_created_run_name_survives_missing_widget_state(tmp_path):
    body = f"""
import streamlit as st
st.session_state.update({{
    "_config_mirror": {{"save_dir": {str(tmp_path)!r}}},
    "save_dir": {str(tmp_path)!r},
    "run_created": True,
    "_run_name_at_create": "keep_me",
    "training_active": False,
    # Simulates Streamlit dropping the text_input widget key for one run.
}})
from flashstudio.pages.training.launch.tab import _render_start_tab
_render_start_tab()
"""
    probe = tmp_path / "probe_run_name.py"
    probe.write_text(body)
    at = AppTest.from_file(str(probe), default_timeout=120)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    assert at.session_state["run_name"] == "keep_me"
    assert at.session_state["run_created"] is True
