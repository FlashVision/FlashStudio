"""Validation-percentage plumbing and official COCO detection."""

import json
import os

from PIL import Image


def _yolo_dir(root, count):
    images = os.path.join(root, "images")
    labels = os.path.join(root, "labels")
    os.makedirs(images)
    os.makedirs(labels)
    for i in range(count):
        Image.new("RGB", (64, 64), "white").save(
            os.path.join(images, f"i{i}.jpg")
        )
        with open(os.path.join(labels, f"i{i}.txt"), "w") as fh:
            fh.write("0 0.5 0.5 0.2 0.2\n")
    return images, labels


def test_yolo_split_respects_val_percentage(tmp_path):
    from flashstudio.pages.data.helpers import _convert_yolo_layout

    source = tmp_path / "yolo"
    _yolo_dir(str(source), 20)
    out = str(tmp_path / "yolo_coco")
    stats = _convert_yolo_layout(
        str(source), out, class_names=["obj"], val_pct=10,
    )
    assert stats["train"]["images"] == 18
    assert stats["valid"]["images"] == 2


def test_kitti_split_respects_val_percentage(tmp_path):
    from flashstudio.pages.data.helpers import _convert_kitti_layout

    root = tmp_path / "kitti"
    img_dir = root / "training" / "image_2"
    lbl_dir = root / "training" / "label_2"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (100, 50), "white").save(img_dir / f"{i}.png")
        (lbl_dir / f"{i}.txt").write_text(
            "Car 0.0 0 0.0 10 10 50 40\n"
        )
    stats = _convert_kitti_layout(
        str(root), str(tmp_path / "kitti_coco"), val_pct=40,
    )
    assert stats["train"] == 3
    assert stats["valid"] == 2


def test_convert_tab_detects_official_coco(tmp_path):
    from flashstudio.pages.data.convert.tab import _detect

    root = tmp_path / "coco"
    (root / "train2017").mkdir(parents=True)
    (root / "val2017").mkdir()
    (root / "annotations").mkdir()
    Image.new("RGB", (50, 40), "white").save(root / "train2017" / "a.jpg")
    Image.new("RGB", (50, 40), "white").save(root / "val2017" / "b.jpg")
    for split, name in (("train", "a.jpg"), ("val", "b.jpg")):
        (root / "annotations" / f"instances_{split}2017.json").write_text(
            json.dumps({
                "images": [{"id": 1, "file_name": name,
                            "width": 50, "height": 40}],
                "annotations": [{"id": 1, "image_id": 1,
                                 "category_id": 1, "bbox": [1, 1, 10, 10]}],
                "categories": [{"id": 1, "name": "cat"}],
            })
        )
    assert _detect(str(root), "Auto-detect") == "coco"
