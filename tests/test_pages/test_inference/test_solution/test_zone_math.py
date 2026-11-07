"""Zone coordinate conversion tests — no double-scaling, no degenerate rects."""

import pytest


@pytest.fixture(autouse=True)
def _quiet_streamlit(monkeypatch):
    import streamlit as st
    monkeypatch.setattr(st, "success", lambda *a, **k: None)
    monkeypatch.setattr(st, "info", lambda *a, **k: None)


def test_rect_uses_opposite_corners(mock_session_state):
    from flashstudio.pages.inference.solution.tab import _store_zone_coords

    _store_zone_coords(
        [[10, 20], [110, 20], [110, 80], [10, 80]], "rect",
    )
    poly = mock_session_state["zone_polygons"][0]
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert (min(xs), max(xs)) == (10, 110)
    assert (min(ys), max(ys)) == (20, 80)


def test_line_stores_exact_points(mock_session_state):
    from flashstudio.pages.inference.solution.tab import _store_zone_coords

    _store_zone_coords([[100, 100], [300, 250]], "line")
    assert mock_session_state["zone_line_points"] == [(100, 100), (300, 250)]


def test_rect_initial_drawing_is_not_degenerate():
    from flashstudio.components.zone_drawer import _build_initial_drawing

    rect = _build_initial_drawing(
        [[10, 20], [110, 20], [110, 80], [10, 80]], "rect", True, 200, 200
    )["objects"][0]
    assert rect["width"] == 100
    assert rect["height"] == 60
