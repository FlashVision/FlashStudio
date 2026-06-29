"""Tests for training monitor log viewer."""

from flashstudio.pages.training.monitor.log_viewer import _render_full_log


def test_render_full_log_importable():
    assert _render_full_log is not None


def test_render_full_log_callable():
    assert callable(_render_full_log)
