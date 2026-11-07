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


class TestCloudflaredTunnel:
    def test_tunnel_process_is_kept_alive(self):
        """The tunnel must stay running, otherwise trycloudflare returns 1033."""
        from flashstudio import launcher

        class FakeProc:
            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = -15

            def wait(self, timeout=None):
                return self.returncode

            def kill(self):
                self.returncode = -9

        fake_proc = FakeProc()

        def fake_popen(cmd, stdout=None, stderr=None):
            stdout.write(b"Your quick Tunnel: https://unit-test-tunnel.trycloudflare.com\n")
            stdout.flush()
            return fake_proc

        launcher._LIVE_TUNNELS.clear()
        try:
            with patch.object(launcher, "_cloudflared_binary", return_value="/usr/bin/cloudflared"), \
                 patch.object(launcher.subprocess, "Popen", side_effect=fake_popen):
                url = launcher._start_cloudflared_tunnel(8501)

            assert url == "https://unit-test-tunnel.trycloudflare.com"
            assert fake_proc.poll() is None
            assert fake_proc in launcher._LIVE_TUNNELS
        finally:
            launcher._LIVE_TUNNELS.clear()

    def test_missing_binary_returns_none(self):
        from flashstudio import launcher

        with patch.object(launcher, "_cloudflared_binary", return_value=None):
            assert launcher._start_cloudflared_tunnel(8501) is None
