"""Tests for extra annotation-format conversion."""

import json

from PIL import Image

from flashstudio.utils.universal_convert import (
    convert_extra_to_coco, detect_extra_format,
)


def test_labelme_conversion(tmp_path):
    source = tmp_path / "labelme"
    source.mkdir()
    for i in range(3):
        Image.new("RGB", (100, 80), "white").save(source / f"img{i}.jpg")
        (source / f"img{i}.json").write_text(json.dumps({
            "imagePath": f"img{i}.jpg",
            "imageWidth": 100,
            "imageHeight": 80,
            "shapes": [
                {"label": "cat", "shape_type": "rectangle",
                 "points": [[10, 10], [50, 40]]},
                {"label": "dog", "shape_type": "polygon",
                 "points": [[60, 20], [90, 20], [90, 60], [60, 60]]},
            ],
        }))

    assert detect_extra_format(str(source)) == "labelme"
    out = tmp_path / "labelme_coco"
    stats = convert_extra_to_coco(str(source), str(out), val_ratio=0.33, seed=1)
    assert stats["train"] == 2
    assert stats["valid"] == 1
    assert stats["classes"] == ["cat", "dog"]

    payload = json.loads(
        (out / "train" / "images" / "_annotations.coco.json").read_text()
    )
    assert len(payload["images"]) == 2
    assert len(payload["annotations"]) == 4


def test_csv_conversion(tmp_path):
    source = tmp_path / "csv"
    (source / "images").mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (200, 100), "white").save(
            source / "images" / f"a{i}.jpg"
        )
    (source / "boxes.csv").write_text(
        "filename,class,xmin,ymin,xmax,ymax\n"
        "a0.jpg,person,10,10,60,80\n"
        "a1.jpg,person,20,20,70,90\n"
        "a2.jpg,person,5,5,55,75\n"
        "a3.jpg,person,0,0,50,70\n"
    )

    assert detect_extra_format(str(source)) == "csv"
    out = tmp_path / "csv_coco"
    stats = convert_extra_to_coco(str(source), str(out), val_ratio=0.5, seed=2)
    assert stats["train"] == 2
    assert stats["valid"] == 2
    assert stats["classes"] == ["person"]
