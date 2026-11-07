"""Tests for dataset path resolution and annotation normalization."""

import json
import os

from flashstudio.utils.dataset_paths import (
    ensure_coco_annotation,
    find_validation_dir,
    load_coco_classes,
    resolve_annotation_file,
    split_dir,
    summarize_coco_splits,
)


def _write_coco(path: str, names=("cat",)) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "images": [{"id": 1, "file_name": "image.jpg", "width": 10, "height": 10}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}
        ],
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(names)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_split_dir_handles_images_folder(tmp_path):
    images = tmp_path / "dataset" / "train" / "images"
    images.mkdir(parents=True)
    assert split_dir(str(images)) == str(tmp_path / "dataset" / "train")


def test_resolves_annotation_in_parent_split_folder(tmp_path):
    root = tmp_path / "dataset"
    images = root / "train" / "images"
    images.mkdir(parents=True)
    (images / "image.jpg").write_bytes(b"x")
    annotation = root / "train" / "_annotations.coco.json"
    _write_coco(str(annotation))

    assert resolve_annotation_file(str(images), split="train") == str(annotation)


def test_resolves_official_coco_annotations_without_mixing_splits(tmp_path):
    root = tmp_path / "coco"
    train_images = root / "train2017"
    val_images = root / "val2017"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    (train_images / "image.jpg").write_bytes(b"x")
    (val_images / "image.jpg").write_bytes(b"x")
    train_ann = root / "annotations" / "instances_train2017.json"
    val_ann = root / "annotations" / "instances_val2017.json"
    _write_coco(str(train_ann), names=("cat", "dog"))
    _write_coco(str(val_ann), names=("cat", "dog"))

    assert resolve_annotation_file(str(train_images), split="train") == str(train_ann)
    assert resolve_annotation_file(str(val_images), split="val") == str(val_ann)


def test_missing_split_annotation_does_not_reuse_train_file(tmp_path):
    root = tmp_path / "coco"
    train_images = root / "train2017"
    val_images = root / "val2017"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    _write_coco(str(root / "annotations" / "instances_train2017.json"))

    assert resolve_annotation_file(str(val_images), split="val") == ""


def test_ensure_coco_annotation_creates_expected_file(tmp_path):
    root = tmp_path / "dataset"
    images = root / "train" / "images"
    images.mkdir(parents=True)
    (images / "image.jpg").write_bytes(b"x")
    source = root / "train" / "_annotations.coco.json"
    _write_coco(str(source))

    result = ensure_coco_annotation(str(images), split="train")

    expected = images / "_annotations.coco.json"
    assert result == str(expected)
    assert expected.exists()
    assert json.loads(expected.read_text())["categories"][0]["name"] == "cat"


def test_find_validation_dir_for_nested_images_layout(tmp_path):
    root = tmp_path / "dataset"
    train_images = root / "train" / "images"
    valid_images = root / "valid" / "images"
    train_images.mkdir(parents=True)
    valid_images.mkdir(parents=True)
    (train_images / "a.jpg").write_bytes(b"x")
    (valid_images / "b.jpg").write_bytes(b"x")

    assert find_validation_dir(str(train_images)) == str(valid_images)


def test_find_validation_dir_for_flat_layout(tmp_path):
    root = tmp_path / "dataset"
    train = root / "train"
    valid = root / "valid"
    train.mkdir(parents=True)
    valid.mkdir(parents=True)
    (train / "a.jpg").write_bytes(b"x")
    (valid / "b.jpg").write_bytes(b"x")

    assert find_validation_dir(str(train)) == str(valid)


def test_find_validation_dir_for_coco_year_suffixes(tmp_path):
    root = tmp_path / "coco"
    train_images = root / "train2017"
    val_images = root / "val2017"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    (train_images / "a.jpg").write_bytes(b"x")
    (val_images / "b.jpg").write_bytes(b"x")

    assert find_validation_dir(str(train_images), root=str(root)) == str(val_images)


def test_find_validation_dir_searches_parent_of_images_root(tmp_path):
    root = tmp_path / "dataset"
    train_images = root / "images" / "train2017"
    val_images = root / "val2017"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    (train_images / "a.jpg").write_bytes(b"x")
    (val_images / "b.jpg").write_bytes(b"x")

    assert find_validation_dir(str(train_images)) == str(val_images)


def test_find_validation_dir_prefers_valid_over_test(tmp_path):
    root = tmp_path / "dataset"
    train = root / "train"
    valid = root / "valid"
    test = root / "test"
    for folder in (train, valid, test):
        folder.mkdir(parents=True)
        (folder / "a.jpg").write_bytes(b"x")

    assert find_validation_dir(str(train)) == str(valid)


def test_load_coco_classes_sorts_by_id(tmp_path):
    annotation = tmp_path / "_annotations.coco.json"
    _write_coco(str(annotation), names=("zebra", "ant"))

    assert load_coco_classes(str(annotation)) == ["zebra", "ant"]


def test_summarize_coco_splits_for_nested_layout(tmp_path):
    root = tmp_path / "dataset"
    train_images = root / "train" / "images"
    val_images = root / "valid" / "images"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    (train_images / "image.jpg").write_bytes(b"x")
    (val_images / "image.jpg").write_bytes(b"x")
    _write_coco(str(root / "train" / "_annotations.coco.json"), names=("cat", "dog"))
    _write_coco(str(root / "valid" / "_annotations.coco.json"), names=("cat", "dog"))

    summary = summarize_coco_splits(str(train_images), str(val_images))

    assert summary["train_images"] == 1
    assert summary["val_images"] == 1
    assert summary["annotations"] == 2
    assert summary["class_names"] == ["cat", "dog"]
