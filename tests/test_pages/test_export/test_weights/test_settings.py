"""Tests for export weights settings."""

from flashstudio.pages.export.weights.settings import (
    _get_export_save_dir,
    render_export_settings,
)
from flashstudio.constants import DEFAULT_SAVE_DIR


def test_get_export_save_dir_returns_session_value(mock_session_state):
    mock_session_state["save_dir"] = "/custom/path"
    result = _get_export_save_dir()
    assert result == "/custom/path"


def test_get_export_save_dir_returns_default_when_empty(mock_session_state):
    mock_session_state["save_dir"] = ""
    result = _get_export_save_dir()
    assert result == DEFAULT_SAVE_DIR


def test_get_export_save_dir_returns_default_when_missing(mock_session_state):
    result = _get_export_save_dir()
    assert result == DEFAULT_SAVE_DIR


def test_render_export_settings_importable():
    assert render_export_settings is not None


def test_render_export_settings_callable():
    assert callable(render_export_settings)
