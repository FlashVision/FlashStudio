"""Tests for flashstudio.cli — CLI argument parsing."""

import pytest
from unittest.mock import patch, MagicMock


class TestCliMain:
    def test_import(self):
        from flashstudio.cli import main
        assert callable(main)

    def test_default_args(self):
        import argparse
        from flashstudio.cli import main

        with patch("argparse.ArgumentParser.parse_args") as mock_parse, \
             patch("flashstudio.launcher.launch") as mock_launch:
            mock_parse.return_value = argparse.Namespace(
                port=8501, no_share=False, ngrok_token=None,
            )
            main()
            mock_launch.assert_called_once_with(
                port=8501, share=True, ngrok_token=None,
            )

    def test_custom_port(self):
        import argparse
        from flashstudio.cli import main

        with patch("argparse.ArgumentParser.parse_args") as mock_parse, \
             patch("flashstudio.launcher.launch") as mock_launch:
            mock_parse.return_value = argparse.Namespace(
                port=9000, no_share=True, ngrok_token="test_token",
            )
            main()
            mock_launch.assert_called_once_with(
                port=9000, share=False, ngrok_token="test_token",
            )
