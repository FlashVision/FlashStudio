"""Tests for training launch tab."""

from flashstudio.pages.training.launch.tab import _render_start_tab


def test_render_start_tab_importable():
    assert _render_start_tab is not None


def test_render_start_tab_callable():
    assert callable(_render_start_tab)
