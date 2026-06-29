"""Tests for data._common logic — auto-detect classes, network check, dataset check."""

import json
from unittest.mock import patch, MagicMock


class TestAutoDetectClasses:
    def test_detects_from_coco_json(self, mock_session_state, tmp_path):
        from flashstudio.pages.data._common import _auto_detect_classes
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        ann = {
            "categories": [
                {"id": 0, "name": "person"},
                {"id": 1, "name": "car"},
            ],
            "images": [], "annotations": [],
        }
        (train_dir / "_annotations.coco.json").write_text(json.dumps(ann))
        mock_session_state["train_img_path"] = str(train_dir)
        classes = _auto_detect_classes()
        assert classes == ["person", "car"]

    def test_falls_back_to_first_json(self, mock_session_state, tmp_path):
        from flashstudio.pages.data._common import _auto_detect_classes
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        ann = {"categories": [{"id": 0, "name": "dog"}], "images": [], "annotations": []}
        (train_dir / "custom_ann.json").write_text(json.dumps(ann))
        mock_session_state["train_img_path"] = str(train_dir)
        classes = _auto_detect_classes()
        assert "dog" in classes

    def test_returns_empty_when_no_path(self, mock_session_state):
        from flashstudio.pages.data._common import _auto_detect_classes
        assert _auto_detect_classes() == []

    def test_returns_empty_for_invalid_dir(self, mock_session_state):
        from flashstudio.pages.data._common import _auto_detect_classes
        mock_session_state["train_img_path"] = "/nonexistent/dir"
        assert _auto_detect_classes() == []

    def test_returns_empty_when_no_json(self, mock_session_state, tmp_path):
        from flashstudio.pages.data._common import _auto_detect_classes
        train_dir = tmp_path / "train"
        train_dir.mkdir()
        (train_dir / "img.jpg").write_bytes(b"\xff")
        mock_session_state["train_img_path"] = str(train_dir)
        assert _auto_detect_classes() == []


class TestCheckNetwork:
    def test_returns_true_when_reachable(self):
        from flashstudio.pages.data._common import _check_network
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = MagicMock()
            assert _check_network() is True

    def test_returns_false_when_unreachable(self):
        from flashstudio.pages.data._common import _check_network
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no network")):
            assert _check_network() is False

    def test_returns_false_on_timeout(self):
        from flashstudio.pages.data._common import _check_network
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
            assert _check_network() is False


class TestDatasetAlreadyDownloadedEdgeCases:
    def test_with_none_output_dir(self, tmp_path):
        from flashstudio.pages.data._common import _dataset_already_downloaded
        result = _dataset_already_downloaded("nonexistent_ds_xyz")
        assert result is None

    def test_empty_valid_dir(self, tmp_path):
        from flashstudio.pages.data._common import _dataset_already_downloaded
        train = tmp_path / "train"
        val = tmp_path / "valid"
        train.mkdir()
        val.mkdir()
        (train / "img.jpg").write_bytes(b"\xff")
        assert _dataset_already_downloaded("test", str(tmp_path)) is None


class TestExtractClassesSorting:
    def test_categories_sorted_by_id(self, tmp_path):
        from flashstudio.pages.data._common import _extract_classes_from_coco_json
        ann = {
            "categories": [
                {"id": 3, "name": "bird"},
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ],
        }
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(ann))
        classes = _extract_classes_from_coco_json(str(path))
        assert classes == ["cat", "dog", "bird"]

    def test_single_category(self, tmp_path):
        from flashstudio.pages.data._common import _extract_classes_from_coco_json
        ann = {"categories": [{"id": 0, "name": "object"}]}
        path = tmp_path / "ann.json"
        path.write_text(json.dumps(ann))
        assert _extract_classes_from_coco_json(str(path)) == ["object"]
