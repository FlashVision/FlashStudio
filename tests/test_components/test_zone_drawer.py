"""Tests for flashstudio.components.zone_drawer."""

from unittest.mock import MagicMock

import sys
sys.modules.setdefault("streamlit", MagicMock())
sys.modules.setdefault("streamlit.elements", MagicMock())
sys.modules.setdefault("streamlit.elements.image", MagicMock())
sys.modules.setdefault("streamlit_drawable_canvas", MagicMock())

from flashstudio.components.zone_drawer import _build_initial_drawing, _HAS_CANVAS


class TestHasCanvas:
    def test_has_canvas_is_bool(self):
        assert isinstance(_HAS_CANVAS, bool)


class TestBuildInitialDrawing:
    def test_polygon_mode_returns_dict_with_objects(self):
        points = [[10, 10], [100, 10], [100, 100], [10, 100]]
        result = _build_initial_drawing(points, "polygon", closed=True, dw=700, dh=400)
        assert isinstance(result, dict)
        assert "objects" in result

    def test_line_mode(self):
        points = [[0, 0], [100, 100]]
        result = _build_initial_drawing(points, "line", closed=False, dw=700, dh=400)
        assert isinstance(result, dict)
        assert "objects" in result
        assert len(result["objects"]) == 1
        assert result["objects"][0]["type"] == "line"

    def test_empty_points_returns_empty_objects(self):
        result = _build_initial_drawing([], "polygon", closed=False, dw=700, dh=400)
        assert isinstance(result, dict)
        assert result["objects"] == []

    def test_rect_mode(self):
        points = [[10, 10], [200, 150]]
        result = _build_initial_drawing(points, "rect", closed=True, dw=700, dh=400)
        assert isinstance(result, dict)
        assert len(result["objects"]) == 1
        assert result["objects"][0]["type"] == "rect"
