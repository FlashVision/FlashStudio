"""Tests for dashboard page module."""




def test_render_dashboard_importable():
    from flashstudio.pages.dashboard.page import render_dashboard

    assert render_dashboard is not None


def test_render_dashboard_callable():
    from flashstudio.pages.dashboard.page import render_dashboard

    assert callable(render_dashboard)
