"""Tests for flashstudio.launcher — launch helpers and config."""

import os
from unittest.mock import patch, MagicMock

from flashstudio.launcher import _get_app_path, _STREAMLIT_CONFIG, _ensure_streamlit_config


class TestGetAppPath:
    def test_returns_absolute(self):
        path = _get_app_path()
        assert os.path.isabs(path)

    def test_points_to_app_py(self):
        path = _get_app_path()
        assert path.endswith("app.py")

    def test_file_exists(self):
        path = _get_app_path()
        assert os.path.isfile(path)


class TestStreamlitConfig:
    def test_config_has_theme(self):
        assert "[theme]" in _STREAMLIT_CONFIG

    def test_config_has_server(self):
        assert "[server]" in _STREAMLIT_CONFIG

    def test_config_headless(self):
        assert "headless = true" in _STREAMLIT_CONFIG

    def test_config_port(self):
        assert "port = 8501" in _STREAMLIT_CONFIG


class TestEnsureStreamlitConfig:
    def test_skips_if_exists(self, tmp_dir):
        config_dir = os.path.join(tmp_dir, ".streamlit")
        config_file = os.path.join(config_dir, "config.toml")
        os.makedirs(config_dir)
        with open(config_file, "w") as f:
            f.write("existing")

        with patch("flashstudio.launcher.Path") as MockPath:
            mock_home = MagicMock()
            MockPath.home.return_value = mock_home
            mock_config_dir = MagicMock()
            mock_home.__truediv__ = MagicMock(return_value=mock_config_dir)
            mock_config_file = MagicMock()
            mock_config_dir.__truediv__ = MagicMock(return_value=mock_config_file)
            mock_config_file.exists.return_value = True
            _ensure_streamlit_config()
            mock_config_file.write_text.assert_not_called()


class TestLaunchFunction:
    def test_import(self):
        from flashstudio.launcher import launch
        assert callable(launch)
