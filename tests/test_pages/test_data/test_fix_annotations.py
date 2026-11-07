"""Annotation repair tests."""

import json
import os

from PIL import Image


def _write_coco(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (100, 80)).save(img_dir / name)
    data = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "b.jpg", "width": 100, "height": 80},
        ],
        "categories": [
            {"id": 0, "name": "cat"},
            {"id": 1, "name": "dog"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400},
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [90, 70, 30, 30], "area": 900},
            {"id": 3, "image_id": 1, "category_id": 1, "bbox": [10, 10, 0, 0], "area": 0},
            {"id": 4, "image_id": 1, "category_id": 99, "bbox": [10, 10, 10, 10], "area": 100},
            {"id": 5, "image_id": 99, "category_id": 0, "bbox": [10, 10, 10, 10], "area": 100},
            {"id": 6, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "area": 400},
        ],
    }
    ann = tmp_path / "annotations.coco.json"
    ann.write_text(json.dumps(data))
    return ann, img_dir


def test_scan_coco_finds_issues(tmp_path):
    from flashstudio.pages.data.fix.annotations import scan_coco
    ann, img_dir = _write_coco(tmp_path)
    report = scan_coco(str(ann), str(img_dir))
    assert report["images"] == 2
    assert report["annotations"] == 6
    assert len(report["zero_area"]) == 1
    assert len(report["unknown_categories"]) == 1
    assert len(report["missing_image_refs"]) == 1
    assert len(report["out_of_bounds"]) == 1
    assert len(report["duplicate_boxes"]) == 1


def test_fix_coco_repairs_and_writes_backup(tmp_path):
    from flashstudio.pages.data.fix.annotations import fix_coco
    ann, img_dir = _write_coco(tmp_path)
    result = fix_coco(str(ann), str(img_dir), backup=True)
    assert os.path.isfile(str(ann) + ".bak")
    assert result["annotations"] == 2
    fixed = json.loads(ann.read_text())
    assert [c["id"] for c in fixed["categories"]] == [1, 2]
    bbox = fixed["annotations"][0]["bbox"]
    assert bbox[2] <= 100 and bbox[3] <= 80
    assert all(1 <= a["category_id"] <= 2 for a in fixed["annotations"])


def test_scan_and_fix_yolo(tmp_path):
    from flashstudio.pages.data.fix.annotations import scan_yolo, fix_yolo
    labels = tmp_path / "labels"
    labels.mkdir()
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.2 0.2\n0 1.2 0.5 0.2 0.2\nbad line\n")
    report = scan_yolo(str(labels), num_classes=2)
    assert report["invalid_lines"]
    assert report["out_of_range_classes"]
    assert report["out_of_bounds"]
    fix_yolo(str(labels), num_classes=2, backup=True)
    lines = (labels / "a.txt").read_text().strip().splitlines()
    assert len(lines) == 1
    assert (labels / "a.txt.bak").is_file()
