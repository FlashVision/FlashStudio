"""Tests for data helpers — _find_root edge cases, _find_images nested, _find_labels nested."""

import os
import json
import pytest


class TestFindRootEdgeCases:
    def test_empty_dir_returns_base(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_root
        assert _find_root(str(tmp_path)) == str(tmp_path)

    def test_single_file_returns_base(self, tmp_path):
        (tmp_path / "readme.txt").write_text("x")
        from flashstudio.pages.data.helpers import _find_root
        assert _find_root(str(tmp_path)) == str(tmp_path)

    def test_nested_single_subdir(self, tmp_path):
        inner = tmp_path / "dataset" / "images"
        inner.mkdir(parents=True)
        from flashstudio.pages.data.helpers import _find_root
        result = _find_root(str(tmp_path))
        assert result == str(tmp_path / "dataset")


class TestFindImagesDeepNesting:
    def test_finds_images_in_deeply_nested_dir(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_images
        deep = tmp_path / "a" / "b" / "c" / "images"
        deep.mkdir(parents=True)
        for name in ["1.jpg", "2.jpg", "3.png"]:
            (deep / name).write_bytes(b"\xff")
        result = _find_images(str(tmp_path))
        assert result is not None
        assert "images" in result

    def test_prefers_images_dir_name(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_images
        other = tmp_path / "other"
        other.mkdir()
        (other / "1.jpg").write_bytes(b"\xff")

        imgs = tmp_path / "images"
        imgs.mkdir()
        for name in ["a.jpg", "b.jpg"]:
            (imgs / name).write_bytes(b"\xff")

        result = _find_images(str(tmp_path))
        assert result is not None
        assert "images" in result

    def test_empty_images_dir_skipped(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_images
        (tmp_path / "images").mkdir()
        assert _find_images(str(tmp_path)) is None


class TestFindAnnotationEdgeCases:
    def test_prefers_coco_json_over_random(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_ann
        (tmp_path / "_annotations.coco.json").write_text("{}")
        (tmp_path / "something.json").write_text("{}")
        result = _find_ann(str(tmp_path))
        assert "annotation" in result.lower() or "coco" in result.lower()

    def test_finds_nested_annotation(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_ann
        nested = tmp_path / "data" / "annotations"
        nested.mkdir(parents=True)
        (nested / "instances_train.coco.json").write_text("{}")
        result = _find_ann(str(tmp_path))
        assert result is not None
        assert result.endswith(".json")


class TestFindLabelsEdgeCases:
    def test_nested_labels_dir(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_labels
        labels = tmp_path / "data" / "labels"
        labels.mkdir(parents=True)
        (labels / "001.txt").write_text("0 0.5 0.5 0.3 0.3")
        result = _find_labels(str(tmp_path))
        assert result is not None
        assert "labels" in result

    def test_labels_dir_case_insensitive(self, tmp_path):
        from flashstudio.pages.data.helpers import _find_labels
        labels = tmp_path / "Labels"
        labels.mkdir()
        (labels / "001.txt").write_text("0 0.5 0.5 0.3 0.3")
        result = _find_labels(str(tmp_path))
        assert result is not None
