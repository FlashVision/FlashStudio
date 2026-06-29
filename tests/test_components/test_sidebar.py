"""Tests for flashstudio.components.sidebar."""

import pytest
from unittest.mock import patch, MagicMock

import sys
sys.modules.setdefault("streamlit", MagicMock())
sys.modules.setdefault("streamlit.elements", MagicMock())
sys.modules.setdefault("streamlit.elements.image", MagicMock())

from flashstudio.components.sidebar import NAV, _get_logo_b64, render_sidebar


class TestNAV:
    def test_nav_has_6_entries(self):
        assert len(NAV) == 6

    def test_nav_entries_are_tuples(self):
        for entry in NAV:
            assert isinstance(entry, tuple)
            assert len(entry) == 2

    def test_nav_labels(self):
        labels = [label for _, label in NAV]
        assert labels == ["Dashboard", "Data", "Model", "Training", "Export", "Inference"]


class TestGetLogoB64:
    def test_returns_none_or_string(self):
        result = _get_logo_b64()
        assert result is None or isinstance(result, str)


class TestRenderSidebar:
    def test_is_callable(self):
        assert callable(render_sidebar)
