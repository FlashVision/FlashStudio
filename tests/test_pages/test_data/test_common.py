"""Tests for flashstudio.pages.data._common — class extraction, dataset helpers."""

import os
import json


class TestExtractClassesFromCocoJson:
    def test_valid_coco(self, tmp_dir):
        from flashstudio.pages.data._common import _extract_classes_from_coco_json

        ann = {
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
                {"id": 0, "name": "person"},
            ],
            "images": [],
            "annotations": [],
        }
        path = os.path.join(tmp_dir, "_annotations.coco.json")
        with open(path, "w") as f:
            json.dump(ann, f)

        classes = _extract_classes_from_coco_json(path)
        assert classes == ["person", "cat", "dog"]

    def test_empty_categories(self, tmp_dir):
        from flashstudio.pages.data._common import _extract_classes_from_coco_json

        ann = {"categories": [], "images": [], "annotations": []}
        path = os.path.join(tmp_dir, "ann.json")
        with open(path, "w") as f:
            json.dump(ann, f)

        assert _extract_classes_from_coco_json(path) == []

    def test_missing_file(self):
        from flashstudio.pages.data._common import _extract_classes_from_coco_json
        assert _extract_classes_from_coco_json("/nonexistent/ann.json") == []

    def test_invalid_json(self, tmp_dir):
        from flashstudio.pages.data._common import _extract_classes_from_coco_json

        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as f:
            f.write("not json")

        assert _extract_classes_from_coco_json(path) == []

    def test_no_categories_key(self, tmp_dir):
        from flashstudio.pages.data._common import _extract_classes_from_coco_json

        path = os.path.join(tmp_dir, "ann.json")
        with open(path, "w") as f:
            json.dump({"images": []}, f)

        assert _extract_classes_from_coco_json(path) == []


class TestDatasetAlreadyDownloaded:
    def test_complete_dataset(self, tmp_dir):
        from flashstudio.pages.data._common import _dataset_already_downloaded

        train = os.path.join(tmp_dir, "train")
        valid = os.path.join(tmp_dir, "valid")
        os.makedirs(train)
        os.makedirs(valid)
        with open(os.path.join(train, "img.jpg"), "w") as f:
            f.write("x")
        with open(os.path.join(valid, "img.jpg"), "w") as f:
            f.write("x")

        result = _dataset_already_downloaded("test", tmp_dir)
        assert result == tmp_dir

    def test_missing_valid(self, tmp_dir):
        from flashstudio.pages.data._common import _dataset_already_downloaded

        train = os.path.join(tmp_dir, "train")
        os.makedirs(train)
        with open(os.path.join(train, "img.jpg"), "w") as f:
            f.write("x")

        assert _dataset_already_downloaded("test", tmp_dir) is None

    def test_empty_train(self, tmp_dir):
        from flashstudio.pages.data._common import _dataset_already_downloaded

        os.makedirs(os.path.join(tmp_dir, "train"))
        os.makedirs(os.path.join(tmp_dir, "valid"))
        assert _dataset_already_downloaded("test", tmp_dir) is None

    def test_nonexistent_dir(self):
        from flashstudio.pages.data._common import _dataset_already_downloaded
        assert _dataset_already_downloaded("test", "/nonexistent/dir") is None

    def test_val_fallback(self, tmp_dir):
        """When 'valid' doesn't exist but 'val' does."""
        from flashstudio.pages.data._common import _dataset_already_downloaded

        train = os.path.join(tmp_dir, "train")
        val = os.path.join(tmp_dir, "val")
        os.makedirs(train)
        os.makedirs(val)
        with open(os.path.join(train, "img.jpg"), "w") as f:
            f.write("x")
        with open(os.path.join(val, "img.jpg"), "w") as f:
            f.write("x")

        result = _dataset_already_downloaded("test", tmp_dir)
        assert result == tmp_dir


class TestExternalDatasets:
    def test_external_datasets_nonempty(self):
        from flashstudio.pages.data._common import EXTERNAL_DATASETS
        assert len(EXTERNAL_DATASETS) > 0

    def test_external_datasets_structure(self):
        from flashstudio.pages.data._common import EXTERNAL_DATASETS
        for ds in EXTERNAL_DATASETS:
            assert "name" in ds
            assert "imgs" in ds
            assert "cls" in ds
            assert "url" in ds

    def test_quick_start_datasets(self):
        from flashstudio.pages.data._common import QUICK_START
        assert len(QUICK_START) > 0
        for ds in QUICK_START:
            assert "id" in ds
            assert "name" in ds
