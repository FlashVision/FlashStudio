"""Tests for flashstudio.utils.filesystem."""

import os
from flashstudio.utils.filesystem import (
    dir_size_bytes, dir_size_str, ensure_dir,
    safe_rmtree, list_subdirs, count_files,
)


class TestDirSizeBytes:
    def test_empty_dir(self, tmp_dir):
        assert dir_size_bytes(tmp_dir) == 0

    def test_dir_with_files(self, tmp_dir):
        with open(os.path.join(tmp_dir, "a.txt"), "w") as f:
            f.write("hello")
        assert dir_size_bytes(tmp_dir) > 0

    def test_nonexistent_dir(self):
        assert dir_size_bytes("/nonexistent/path/xyz") == 0


class TestDirSizeStr:
    def test_empty_dir(self, tmp_dir):
        result = dir_size_str(tmp_dir)
        assert "KB" in result or "B" in result

    def test_returns_string(self, tmp_dir):
        assert isinstance(dir_size_str(tmp_dir), str)


class TestEnsureDir:
    def test_creates_dir(self, tmp_dir):
        new_dir = os.path.join(tmp_dir, "subdir", "nested")
        result = ensure_dir(new_dir)
        assert os.path.isdir(new_dir)
        assert result == new_dir

    def test_existing_dir_ok(self, tmp_dir):
        result = ensure_dir(tmp_dir)
        assert result == tmp_dir


class TestSafeRmtree:
    def test_removes_dir(self, tmp_dir):
        target = os.path.join(tmp_dir, "to_remove")
        os.makedirs(target)
        with open(os.path.join(target, "file.txt"), "w") as f:
            f.write("data")
        assert safe_rmtree(target) is True
        assert not os.path.exists(target)

    def test_nonexistent_returns_false(self):
        assert safe_rmtree("/nonexistent/path/xyz") is False


class TestListSubdirs:
    def test_lists_subdirs(self, tmp_dir):
        os.makedirs(os.path.join(tmp_dir, "a"))
        os.makedirs(os.path.join(tmp_dir, "b"))
        with open(os.path.join(tmp_dir, "file.txt"), "w") as f:
            f.write("data")
        dirs = list_subdirs(tmp_dir, sort_by_mtime=False)
        assert sorted(dirs) == ["a", "b"]

    def test_nonexistent_returns_empty(self):
        assert list_subdirs("/nonexistent/path") == []

    def test_empty_dir(self, tmp_dir):
        assert list_subdirs(tmp_dir) == []


class TestCountFiles:
    def test_counts_all_files(self, tmp_dir):
        for name in ["a.txt", "b.py", "c.jpg"]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write("x")
        assert count_files(tmp_dir) == 3

    def test_counts_filtered(self, tmp_dir):
        for name in ["a.txt", "b.py", "c.jpg", "d.jpg"]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write("x")
        assert count_files(tmp_dir, extensions=(".jpg",)) == 2

    def test_nonexistent_returns_zero(self):
        assert count_files("/nonexistent/path") == 0
