"""Tests for training monitor visualizations."""

from flashstudio.pages.training.monitor.visualizations import (
    _render_visualizations,
    _render_image_grid,
)


def test_render_visualizations_importable():
    assert _render_visualizations is not None


def test_render_image_grid_importable():
    assert _render_image_grid is not None


def test_render_visualizations_callable():
    assert callable(_render_visualizations)


def test_render_image_grid_callable():
    assert callable(_render_image_grid)
