"""Tests for flashstudio.components.wizard — step indicator and navigation logic."""

import pytest
from unittest.mock import patch, MagicMock


SAMPLE_STEPS = [
    {"id": "data", "label": "Data", "icon": "1"},
    {"id": "model", "label": "Model", "icon": "2"},
    {"id": "training", "label": "Training", "icon": "3"},
    {"id": "export", "label": "Export", "icon": "4"},
]


class TestRenderStepIndicator:
    def test_all_pending(self):
        with patch("streamlit.markdown") as mock_md:
            from flashstudio.components.wizard import render_step_indicator
            render_step_indicator(SAMPLE_STEPS, 0)
            html = mock_md.call_args[0][0]
            assert "step-active" in html
            assert "step-pending" in html
            assert "Data" in html

    def test_middle_step_active(self):
        with patch("streamlit.markdown") as mock_md:
            from flashstudio.components.wizard import render_step_indicator
            render_step_indicator(SAMPLE_STEPS, 2)
            html = mock_md.call_args[0][0]
            assert "step-done" in html
            assert "step-active" in html

    def test_last_step_active(self):
        with patch("streamlit.markdown") as mock_md:
            from flashstudio.components.wizard import render_step_indicator
            render_step_indicator(SAMPLE_STEPS, 3)
            html = mock_md.call_args[0][0]
            assert "step-active" in html
            assert "Export" in html


class TestRenderNavigation:
    def test_single_step_no_render(self):
        with patch("streamlit.markdown") as mock_md:
            from flashstudio.components.wizard import render_navigation
            render_navigation([SAMPLE_STEPS[0]], 0)
            mock_md.assert_not_called()

    def test_importable(self):
        from flashstudio.components.wizard import render_navigation
        assert callable(render_navigation)
