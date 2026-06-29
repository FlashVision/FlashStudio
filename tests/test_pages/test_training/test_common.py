"""Tests for flashstudio.pages.training._common — save dir and log finder."""

import os


class TestGetSaveDir:
    def test_from_session(self, mock_session_state):
        mock_session_state["save_dir"] = "/custom/save"
        from flashstudio.pages.training._common import _get_save_dir
        assert _get_save_dir() == "/custom/save"

    def test_falls_back_to_default(self, mock_session_state):
        from flashstudio.pages.training._common import _get_save_dir
        from flashstudio.utils import DEFAULTS
        result = _get_save_dir()
        assert result == DEFAULTS["save_dir"]


class TestFindLogFile:
    def test_finds_largest_log(self, tmp_dir):
        from flashstudio.pages.training._common import _find_log_file

        log1 = os.path.join(tmp_dir, "train_20240101.log")
        log2 = os.path.join(tmp_dir, "train_20240102.log")
        with open(log1, "w") as f:
            f.write("short")
        with open(log2, "w") as f:
            f.write("this is a much longer log file with more content")

        result = _find_log_file(tmp_dir)
        assert result == log2

    def test_no_logs(self, tmp_dir):
        from flashstudio.pages.training._common import _find_log_file
        assert _find_log_file(tmp_dir) is None

    def test_single_log(self, tmp_dir):
        from flashstudio.pages.training._common import _find_log_file

        log = os.path.join(tmp_dir, "train_20240101.log")
        with open(log, "w") as f:
            f.write("content")

        assert _find_log_file(tmp_dir) == log
