"""Tests for dataset-path repair of stale saved selections."""

import json
import os

from flashstudio.utils.dataset_state import repair_dataset_paths


def _write_split(root, split, image_name="image.jpg", class_name="cat"):
    folder = root / split
    folder.mkdir(parents=True, exist_ok=True)
    (folder / image_name).write_bytes(b"x")
    (folder / "_annotations.coco.json").write_text(
        json.dumps({
            "images": [{
                "id": 1, "file_name": image_name, "width": 10, "height": 10,
            }],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]},
            ],
            "categories": [
                {"id": 1, "name": class_name, "supercategory": "object"},
            ],
        })
    )
    return folder


def _write_nested_split(root, split, image_name="image.jpg"):
    images = root / split / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / image_name).write_bytes(b"x")
    (images / "_annotations.coco.json").write_text(
        json.dumps({
            "images": [{
                "id": 1, "file_name": image_name, "width": 10, "height": 10,
            }],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]},
            ],
            "categories": [
                {"id": 1, "name": "raccoon", "supercategory": "object"},
            ],
        })
    )
    return images


class TestRepairDatasetPaths:
    def test_healthy_pair_is_untouched(self, tmp_path, mock_session_state):
        root = tmp_path / "dataset"
        train = _write_split(root, "train")
        val = _write_split(root, "valid")
        mock_session_state["train_img_path"] = str(train)
        mock_session_state["val_img_path"] = str(val)
        mock_session_state["dataset_output_path"] = str(root)

        report = repair_dataset_paths()

        assert report["repaired"] is False
        assert mock_session_state["train_img_path"] == str(train)

    def test_repairs_internal_train_path_from_dataset_root(
        self, tmp_path, mock_session_state
    ):
        root = tmp_path / "raccoon"
        train = _write_nested_split(root, "train")
        val = _write_nested_split(root, "valid")
        stale = tmp_path / "_yolo_abc123" / "valid" / "images"
        stale.mkdir(parents=True)
        (stale / "x.jpg").write_bytes(b"x")

        mock_session_state["train_img_path"] = str(stale)
        mock_session_state["val_img_path"] = str(val)
        mock_session_state["dataset_output_path"] = str(root)
        mock_session_state["class_names"] = "wrong"

        report = repair_dataset_paths()

        assert report["repaired"] is True
        assert report["root"] == str(root)
        assert mock_session_state["train_img_path"] == str(train)
        assert mock_session_state["val_img_path"] == str(val)
        assert mock_session_state["class_names"] == "raccoon"
        assert mock_session_state["num_classes"] == 1
        assert mock_session_state["dataset_name"] == "raccoon"
        assert os.path.isfile(mock_session_state["train_ann_path"])

    def test_repairs_validation_folder_used_as_train(self, tmp_path, mock_session_state):
        root = tmp_path / "dataset"
        train = _write_split(root, "train")
        val = _write_split(root, "valid")
        mock_session_state["train_img_path"] = str(val)
        mock_session_state["val_img_path"] = str(val)
        mock_session_state["dataset_output_path"] = str(root)

        report = repair_dataset_paths()

        assert report["repaired"] is True
        assert mock_session_state["train_img_path"] == str(train)
        assert mock_session_state["val_img_path"] == str(val)

    def test_repairs_from_validation_root_when_output_path_missing(
        self, tmp_path, mock_session_state
    ):
        root = tmp_path / "dataset"
        train = _write_split(root, "train")
        val = _write_split(root, "valid")
        mock_session_state["train_img_path"] = str(tmp_path / "_yolo_x" / "valid")
        mock_session_state["val_img_path"] = str(val)

        report = repair_dataset_paths()

        assert report["repaired"] is True
        assert mock_session_state["train_img_path"] == str(train)

    def test_does_not_repair_without_a_full_pair(self, tmp_path, mock_session_state):
        only_train = tmp_path / "only"
        _write_split(only_train, "train")
        mock_session_state["train_img_path"] = str(tmp_path / "_yolo_x" / "valid")
        mock_session_state["dataset_output_path"] = str(only_train)

        report = repair_dataset_paths()

        assert report["repaired"] is False
        assert mock_session_state["train_img_path"] == str(
            tmp_path / "_yolo_x" / "valid"
        )
