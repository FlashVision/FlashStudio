"""Tests for training monitor tab."""

from flashstudio.pages.training.monitor.tab import _render_monitor_tab


def test_render_monitor_tab_importable():
    assert _render_monitor_tab is not None


def test_render_monitor_tab_callable():
    assert callable(_render_monitor_tab)
