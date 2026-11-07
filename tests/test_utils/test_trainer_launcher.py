"""Ultralytics backend helper tests."""

import json

from PIL import Image

from flashstudio.utils.trainer_launcher import (
    _coco_to_yolo,
    _ultralytics_model_name,
)


def test_model_name_maps_yolo_families():
    assert _ultralytics_model_name({"arch_family": "YOLOv8", "model_size": "n"}) == "yolov8n"
    assert _ultralytics_model_name({"arch_family": "YOLOv11", "model_size": "s"}) == "yolo11s"
    assert _ultralytics_model_name({"arch_family": "YOLOv9", "model_size": "x"}) == "yolov9e"
    assert _ultralytics_model_name({"arch_family": "YOLOv10", "model_size": "n"}) == "yolov10n"


def test_model_name_rejects_non_yolo_families():
    import pytest

    with pytest.raises(RuntimeError):
        _ultralytics_model_name({"arch_family": "FlashDet (recommended)"})
    with pytest.raises(RuntimeError):
        _ultralytics_model_name({"arch_family": "YOLOX"})


def _write_split(root, split, start_id):
    split_dir = root / split
    split_dir.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(split_dir / "a.jpg")
    data = {
        "images": [
            {"id": start_id, "file_name": "a.jpg", "width": 100, "height": 80},
        ],
        "categories": [
            {"id": 0, "name": "cat"},
            {"id": 1, "name": "dog"},
        ],
        "annotations": [
            {
                "id": start_id, "image_id": start_id, "category_id": 1,
                "bbox": [10, 10, 20, 20], "area": 400,
            },
        ],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(data))


def test_coco_to_yolo_writes_zero_based_labels(tmp_path):
    src = tmp_path / "src"
    _write_split(src, "train", 1)
    _write_split(src, "valid", 2)
    out = tmp_path / "yolo"

    yaml_path = _coco_to_yolo(
        str(src / "train"), str(src / "valid"), str(out),
        class_names=["cat", "dog"],
    )

    assert (out / "train" / "images" / "a.jpg").exists()
    label = (out / "train" / "labels" / "a.txt").read_text().split()
    assert label[0] == "1"
    assert abs(float(label[1]) - 0.2) < 1e-3
    assert (out / "valid" / "labels" / "a.txt").exists()
    yaml_text = open(yaml_path, encoding="utf-8").read()
    assert "nc: 2" in yaml_text
    assert '"dog"' in yaml_text
