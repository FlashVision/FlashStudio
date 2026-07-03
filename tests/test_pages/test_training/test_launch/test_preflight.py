"""Tests for training launch preflight checks."""

import os
import json

from flashstudio.pages.training.launch.preflight import _run_preflight_checks


def test_run_preflight_checks_callable():
    assert callable(_run_preflight_checks)


def test_run_preflight_checks_returns_list_of_tuples(mock_session_state):
    checks = _run_preflight_checks()
    assert isinstance(checks, list)
    for item in checks:
        assert isinstance(item, tuple)
        assert len(item) == 3
        label, ok, msg = item
        assert isinstance(label, str)
        assert isinstance(ok, bool)
        assert isinstance(msg, str)


def test_no_data_dataset_fails(mock_session_state):
    mock_session_state["train_img_path"] = ""
    checks = _run_preflight_checks()
    dataset_checks = [c for c in checks if c[0] == "Dataset"]
    assert len(dataset_checks) == 1
    assert dataset_checks[0][1] is False


def test_with_data_classes_model_all_pass(mock_session_state, tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
    ann_path = os.path.join(tmp_dir, "_annotations.coco.json")
    with open(ann_path, "w") as f:
        json.dump({"images": [], "annotations": [], "categories": [{"id": 1, "name": "cat"}]}, f)

    mock_session_state["train_img_path"] = tmp_dir
    mock_session_state["num_classes"] = 3
    mock_session_state["class_names"] = "cat\ndog\nbird"
    mock_session_state["model_arch"] = "FlashDet-N"

    checks = _run_preflight_checks()
    for label, ok, msg in checks:
        if label in ("Dataset", "Annotations", "Classes", "Model"):
            assert ok, f"{label} should pass but got: {msg}"


def test_gpu_and_disk_checks_always_present(mock_session_state):
    checks = _run_preflight_checks()
    labels = [c[0] for c in checks]
    assert "GPU" in labels
    assert "Disk" in labels
