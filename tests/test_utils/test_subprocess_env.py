"""Tests for the UTF-8 child-process environment helper."""

import os

from flashstudio.utils.subprocess_env import utf8_subprocess_env


class TestUtf8SubprocessEnv:
    def test_forces_utf8_io(self, monkeypatch):
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        env = utf8_subprocess_env()
        assert env["PYTHONIOENCODING"] == "utf-8"
        assert env["PYTHONUTF8"] == "1"
        assert env["PYTHONUNBUFFERED"] == "1"

    def test_prepends_package_root_to_pythonpath(self, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "existing")
        env = utf8_subprocess_env("/pkg/root")
        parts = env["PYTHONPATH"].split(os.pathsep)
        assert parts == ["/pkg/root", "existing"]

    def test_no_package_root_keeps_pythonpath(self, monkeypatch):
        monkeypatch.setenv("PYTHONPATH", "existing")
        env = utf8_subprocess_env()
        assert env["PYTHONPATH"] == "existing"

    def test_does_not_mutate_parent_environment(self, monkeypatch):
        monkeypatch.delenv("PYTHONIOENCODING", raising=False)
        utf8_subprocess_env()
        assert "PYTHONIOENCODING" not in os.environ
