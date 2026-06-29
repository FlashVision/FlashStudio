"""Tests for training monitor dashboard."""

from flashstudio.pages.training.monitor.dashboard import (
    _render_run_dashboard,
    _render_metrics_from_history,
)


def test_render_run_dashboard_importable():
    assert _render_run_dashboard is not None


def test_render_metrics_from_history_importable():
    assert _render_metrics_from_history is not None


def test_render_run_dashboard_callable():
    assert callable(_render_run_dashboard)


def test_render_metrics_from_history_callable():
    assert callable(_render_metrics_from_history)
