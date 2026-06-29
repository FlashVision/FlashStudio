"""Tests for project_banner module."""

from unittest.mock import patch, MagicMock

import pytest


def test_render_project_banner_importable():
    from flashstudio.pages.dashboard.overview.project_banner import render_project_banner

    assert render_project_banner is not None


def test_render_project_banner_callable():
    from flashstudio.pages.dashboard.overview.project_banner import render_project_banner

    assert callable(render_project_banner)


@patch("streamlit.markdown")
def test_render_project_banner_calls_st_markdown(mock_markdown):
    from flashstudio.pages.dashboard.overview.project_banner import render_project_banner

    project = {
        "name": "Test",
        "description": "desc",
        "created": "2024-01-01",
        "updated": "2024-06-01",
    }
    render_project_banner(project)
    mock_markdown.assert_called()
