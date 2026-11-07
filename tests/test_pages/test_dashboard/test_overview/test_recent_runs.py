"""Tests for recent_runs module."""




def test_render_recent_runs_importable():
    from flashstudio.pages.dashboard.overview.recent_runs import render_recent_runs

    assert render_recent_runs is not None


def test_render_recent_runs_callable():
    from flashstudio.pages.dashboard.overview.recent_runs import render_recent_runs

    assert callable(render_recent_runs)
