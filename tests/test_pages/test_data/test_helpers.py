"""Tests for flashstudio.pages.data.helpers — file discovery utilities."""

import os
import pytest


class TestFindImages:
    def test_finds_images_dir(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_images

        img_dir = os.path.join(tmp_dir, "images")
        os.makedirs(img_dir)
        for name in ["a.jpg", "b.png", "c.bmp"]:
            with open(os.path.join(img_dir, name), "w") as f:
                f.write("x")

        result = _find_images(tmp_dir)
        assert result is not None
        assert "images" in result

    def test_no_images(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_images

        with open(os.path.join(tmp_dir, "readme.txt"), "w") as f:
            f.write("x")

        assert _find_images(tmp_dir) is None

    def test_prefers_most_images(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_images

        dir_a = os.path.join(tmp_dir, "a")
        dir_b = os.path.join(tmp_dir, "b")
        os.makedirs(dir_a)
        os.makedirs(dir_b)

        with open(os.path.join(dir_a, "img.jpg"), "w") as f:
            f.write("x")
        for name in ["1.jpg", "2.jpg", "3.jpg"]:
            with open(os.path.join(dir_b, name), "w") as f:
                f.write("x")

        result = _find_images(tmp_dir)
        assert result is not None
        assert os.path.basename(result) == "b"


class TestFindAnn:
    def test_finds_coco_json(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_ann

        ann_path = os.path.join(tmp_dir, "_annotations.coco.json")
        with open(ann_path, "w") as f:
            f.write("{}")

        result = _find_ann(tmp_dir)
        assert result == ann_path

    def test_finds_annotation_json(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_ann

        ann_path = os.path.join(tmp_dir, "annotation_file.json")
        with open(ann_path, "w") as f:
            f.write("{}")

        result = _find_ann(tmp_dir)
        assert result == ann_path

    def test_ignores_random_json(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_ann

        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            f.write("{}")

        assert _find_ann(tmp_dir) is None

    def test_no_json(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_ann
        assert _find_ann(tmp_dir) is None


class TestFindLabels:
    def test_finds_labels_dir(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_labels

        labels_dir = os.path.join(tmp_dir, "labels")
        os.makedirs(labels_dir)
        with open(os.path.join(labels_dir, "001.txt"), "w") as f:
            f.write("0 0.5 0.5 0.3 0.3")

        result = _find_labels(tmp_dir)
        assert result is not None
        assert "labels" in result

    def test_no_labels(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_labels
        assert _find_labels(tmp_dir) is None

    def test_labels_dir_without_txt(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_labels

        labels_dir = os.path.join(tmp_dir, "labels")
        os.makedirs(labels_dir)
        with open(os.path.join(labels_dir, "readme.md"), "w") as f:
            f.write("x")

        assert _find_labels(tmp_dir) is None


class TestFindRoot:
    def test_single_subdir(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_root

        sub = os.path.join(tmp_dir, "dataset")
        os.makedirs(sub)
        result = _find_root(tmp_dir)
        assert result == sub

    def test_multiple_entries_returns_base(self, tmp_dir):
        from flashstudio.pages.data.helpers import _find_root

        os.makedirs(os.path.join(tmp_dir, "a"))
        os.makedirs(os.path.join(tmp_dir, "b"))
        result = _find_root(tmp_dir)
        assert result == tmp_dir
