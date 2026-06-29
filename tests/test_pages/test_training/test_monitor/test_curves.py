"""Tests for training monitor curves."""

from flashstudio.pages.training.monitor.curves import _render_curves


def test_render_curves_importable():
    assert _render_curves is not None


def test_render_curves_callable():
    assert callable(_render_curves)
