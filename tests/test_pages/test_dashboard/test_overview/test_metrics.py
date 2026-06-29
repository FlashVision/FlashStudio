"""Tests for metrics module."""




def test_render_metrics_importable():
    from flashstudio.pages.dashboard.overview.metrics import render_metrics

    assert render_metrics is not None


def test_render_metrics_callable():
    from flashstudio.pages.dashboard.overview.metrics import render_metrics

    assert callable(render_metrics)
