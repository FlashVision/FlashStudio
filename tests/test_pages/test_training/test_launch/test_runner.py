"""Tests for training launch runner."""

import os

from flashstudio.pages.training.launch.runner import (
    _generate_run_name, _start_training,
)


def _write_split(root, split, name="cat"):
    images = root / split / "images"
    images.mkdir(parents=True)
    (images / "image.jpg").write_bytes(b"x")
    (root / split / "_annotations.coco.json").write_text(
        '{"images":[{"id":1,"file_name":"image.jpg","width":10,"height":10}],'
        '"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[1,1,2,2]}],'
        '"categories":[{"id":1,"name":"' + name + '","supercategory":"object"}]}'
    )
    return images


def test_start_training_normalizes_official_coco_layout(
    tmp_path, mock_session_state, monkeypatch
):
    root = tmp_path / "coco"
    train_images = root / "train2017"
    val_images = root / "val2017"
    train_images.mkdir(parents=True)
    val_images.mkdir(parents=True)
    (train_images / "a.jpg").write_bytes(b"x")
    (val_images / "b.jpg").write_bytes(b"x")
    ann_dir = root / "annotations"
    ann_dir.mkdir()
    annotation = (
        '{"images":[{"id":1,"file_name":"image.jpg","width":10,"height":10}],'
        '"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[1,1,2,2]}],'
        '"categories":[{"id":1,"name":"cat","supercategory":"object"}]}'
    )
    (ann_dir / "instances_train2017.json").write_text(annotation)
    (ann_dir / "instances_val2017.json").write_text(annotation)
    mock_session_state["train_img_path"] = str(train_images)
    mock_session_state["val_img_path"] = str(val_images)
    mock_session_state["class_names"] = "cat"
    mock_session_state["num_classes"] = 1
    mock_session_state["dataset_name"] = "COCO"

    called = []
    monkeypatch.setattr(
        "flashstudio.pages.training.launch.runner._run_flashdet_training",
        lambda: called.append(True),
    )

    _start_training()

    assert called == [True]
    assert os.path.isfile(str(root / "train" / "_annotations.coco.json"))
    assert os.path.isfile(str(root / "valid" / "_annotations.coco.json"))


def test_start_training_replaces_stale_train_symlink(
    tmp_path, mock_session_state, monkeypatch
):
    old_root = tmp_path / "old"
    old_train = _write_split(old_root, "train")

    root = tmp_path / "dataset"
    train_images = _write_split(root, "train")
    val_images = _write_split(root, "valid")
    data_root = root / "train"
    os.symlink(str(old_train), str(data_root / "train"))

    mock_session_state["train_img_path"] = str(train_images)
    mock_session_state["val_img_path"] = str(val_images)
    mock_session_state["class_names"] = "cat"
    mock_session_state["num_classes"] = 1
    mock_session_state["dataset_name"] = "Test"

    called = []
    monkeypatch.setattr(
        "flashstudio.pages.training.launch.runner._run_flashdet_training",
        lambda: called.append(True),
    )

    _start_training()

    assert called == [True]
    assert os.path.realpath(str(data_root / "train")) == os.path.realpath(
        str(train_images)
    )


def test_start_training_converts_nonstandard_yolo_layout(
    tmp_path, mock_session_state, monkeypatch
):
    from PIL import Image

    root = tmp_path / "yolo"
    for split, count in (("train2017", 4), ("val2017", 2)):
        images = root / "images" / split
        labels = root / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (12, 12), "white").save(images / f"{i}.jpg")
            (labels / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")

    mock_session_state["train_img_path"] = str(root / "images" / "train2017")
    mock_session_state["val_img_path"] = str(root / "images" / "val2017")
    mock_session_state["dataset_name"] = "YOLO"

    called = []
    monkeypatch.setattr(
        "flashstudio.pages.training.launch.runner._run_flashdet_training",
        lambda: called.append(True),
    )

    _start_training()

    assert called == [True]
    converted = tmp_path / "yolo_coco"
    assert os.path.isfile(str(converted / "train" / "_annotations.coco.json"))
    assert os.path.isfile(str(converted / "valid" / "_annotations.coco.json"))
    assert mock_session_state["train_img_path"] == str(converted / "train")
    assert mock_session_state["val_img_path"] == str(converted / "valid")
    assert mock_session_state["class_names"] == "class_0"
    assert mock_session_state["num_classes"] == 1


def test_start_training_normalizes_nested_images_layout(
    tmp_path, mock_session_state, monkeypatch
):
    root = tmp_path / "dataset"
    train_images = _write_split(root, "train")
    val_images = _write_split(root, "valid")
    mock_session_state["train_img_path"] = str(train_images)
    mock_session_state["val_img_path"] = str(val_images)
    mock_session_state["class_names"] = "cat"
    mock_session_state["num_classes"] = 1
    mock_session_state["dataset_name"] = "Test"

    called = []
    monkeypatch.setattr(
        "flashstudio.pages.training.launch.runner._run_flashdet_training",
        lambda: called.append(True),
    )

    _start_training()

    assert called == [True]
    # FlashDet computes data_root from dirname(train_images) and verifies
    # data_root/train and data_root/valid.  The runner must create those
    # symlinks and put annotations inside the actual image directories.
    data_root = root / "train"
    assert os.path.isfile(str(data_root / "train" / "_annotations.coco.json"))
    assert os.path.isfile(str(data_root / "valid" / "_annotations.coco.json"))


def test_start_training_rejects_validation_folder_as_train(
    tmp_path, mock_session_state, monkeypatch
):
    root = tmp_path / "dataset"
    _write_split(root, "valid")
    mock_session_state["train_img_path"] = str(root / "valid" / "images")
    mock_session_state["val_img_path"] = str(root / "valid" / "images")
    mock_session_state["dataset_name"] = "Test"

    called = []
    monkeypatch.setattr(
        "flashstudio.pages.training.launch.runner._run_flashdet_training",
        lambda: called.append(True),
    )

    _start_training()

    assert called == []
    assert mock_session_state["training_status"] == "Invalid dataset layout"
    assert "validation/test split" in mock_session_state["training_error"]


def test_start_training_repairs_stale_internal_train_path(
    tmp_path, mock_session_state, monkeypatch
):
    root = tmp_path / "raccoon"

    def _flat_split(split):
        folder = root / split
        folder.mkdir(parents=True)
        (folder / "image.jpg").write_bytes(b"x")
        (folder / "_annotations.coco.json").write_text(
            '{"images":[{"id":1,"file_name":"image.jpg","width":10,"height":10}],'
            '"annotations":[{"id":1,"image_id":1,"category_id":1,'
            '"bbox":[1,1,2,2]}],'
            '"categories":[{"id":1,"name":"raccoon","supercategory":"object"}]}'
        )
        return folder

    train = _flat_split("train")
    val = _flat_split("valid")
    stale = tmp_path / "_yolo_abc" / "valid" / "images"
    stale.mkdir(parents=True)

    mock_session_state["train_img_path"] = str(stale)
    mock_session_state["val_img_path"] = str(val)
    mock_session_state["dataset_output_path"] = str(root)
    mock_session_state["dataset_name"] = "Raccoon"
    mock_session_state["class_names"] = "raccoon"
    mock_session_state["num_classes"] = 1

    called = []
    monkeypatch.setattr(
        "flashstudio.pages.training.launch.runner._run_flashdet_training",
        lambda: called.append(True),
    )

    _start_training()

    assert called == [True]
    assert mock_session_state["train_img_path"] == str(train)
    assert mock_session_state["val_img_path"] == str(val)


def test_generate_run_name_returns_non_empty(mock_session_state):
    name = _generate_run_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_generate_run_name_contains_model_size_code(mock_session_state):
    mock_session_state["model_arch"] = "FlashDet-Nano"
    name = _generate_run_name()
    assert "nano" in name


def test_generate_run_name_contains_dataset_code(mock_session_state):
    mock_session_state["dataset_name"] = "MyDataset (demo)"
    name = _generate_run_name()
    assert "mydata" in name.lower()


def test_generate_run_name_contains_timestamp(mock_session_state):
    name = _generate_run_name()
    parts = name.split("_")
    assert len(parts) >= 3
    # Timestamp part should have digits (MMDD_HHMM pattern)
    timestamp_part = "_".join(parts[-2:])
    assert any(c.isdigit() for c in timestamp_part)
