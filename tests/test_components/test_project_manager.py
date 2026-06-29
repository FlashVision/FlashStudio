"""Tests for flashstudio.components.project_manager — pure-logic helpers."""

import pytest
from flashstudio.components.project_manager import _slugify, _dir_size_str


class TestSlugify:
    def test_basic(self):
        assert _slugify("My Project") == "my_project"

    def test_special_chars(self):
        result = _slugify("Test! @#$ Project")
        assert "!" not in result
        assert "@" not in result

    def test_truncation(self):
        long_name = "a" * 100
        result = _slugify(long_name)
        assert len(result) <= 30

    def test_preserves_dashes(self):
        result = _slugify("my-project")
        assert result in ("my_project", "my-project")


class TestDirSizeStr:
    def test_empty_dir(self, tmp_dir):
        result = _dir_size_str(tmp_dir)
        assert "KB" in result

    def test_returns_string(self, tmp_dir):
        assert isinstance(_dir_size_str(tmp_dir), str)
