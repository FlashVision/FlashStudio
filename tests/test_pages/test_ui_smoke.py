"""Structural smoke tests for every page and framework mode.

These use Streamlit's AppTest to render real page functions and assert that the
expected tab skeleton exists and no runtime exception is raised.
"""

import json
import os

import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_probe(tmp_path, name, body, state=None):
    state = state or {}
    lines = [
        "import os",
        "import sys",
        "",
        f"sys.path.insert(0, {ROOT!r})",
        "import streamlit as st",
        "",
        f'st.session_state["_config_mirror"] = {{"save_dir": {str(tmp_path)!r}}}',
        f'st.session_state["save_dir"] = {str(tmp_path)!r}',
        f"st.session_state.update({state!r})",
        body,
    ]
    path = tmp_path / f"probe_{name}.py"
    path.write_text("\n".join(lines) + "\n")
    at = AppTest.from_file(str(path), default_timeout=120)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def _write_checkpoint(root, framework):
    path = root / ("model.pt" if framework == "Ultralytics" else "model.pth")
    if framework == "Ultralytics":
        path.write_bytes(b"not-a-real-checkpoint")
    else:
        architecture = "yolox" if framework == "YOLOX" else "flashdet"
        torch.save(
            {"model_state_dict": {}, "config": {"architecture": architecture}},
            path,
        )
    return path


@pytest.mark.parametrize("family", ["FlashDet", "Ultralytics", "YOLOX"])
def test_inference_skeleton(tmp_path, family):
    _write_checkpoint(tmp_path, family)
    at = _run_probe(
        tmp_path,
        f"inference_{family.lower()}",
        "from flashstudio.pages.inference.page import render_inference_page\n"
        "render_inference_page()",
        {"infer_family": family, "weight_source": "Training output"},
    )
    labels = [tab.label for tab in at.tabs]
    assert labels == ["Model", "Data", "Solution", "Run", "Live"]
    assert any("Framework" == radio.label for radio in at.radio)


@pytest.mark.parametrize("framework", ["FlashDet", "Ultralytics", "YOLOX"])
def test_export_skeleton(tmp_path, framework):
    _write_checkpoint(tmp_path, framework)
    at = _run_probe(
        tmp_path,
        f"export_{framework.lower()}",
        "from flashstudio.pages.export.page import render_export_page\n"
        "render_export_page()",
        {"export_framework": framework},
    )
    labels = [tab.label for tab in at.tabs]
    assert labels == ["Convert", "Optimize", "Fusion", "Export", "INT8", "Benchmark"]
    assert any("Framework" == radio.label for radio in at.radio)


def test_data_skeleton(tmp_path):
    at = _run_probe(
        tmp_path, "data",
        "from flashstudio.pages.data.page import render_data_page\nrender_data_page()",
    )
    assert [tab.label for tab in at.tabs] == [
        "Upload", "Download", "Preview", "Verify", "Edit", "Fix", "Quality"
    ]


def test_data_skeleton_with_dataset(tmp_path):
    for split in ("train", "valid"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        Image.new("RGB", (64, 48), "white").save(split_dir / "a.jpg")
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
            "categories": [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [4, 4, 20, 20], "area": 400,
            }],
        }
        (split_dir / "_annotations.coco.json").write_text(json.dumps(data))

    at = _run_probe(
        tmp_path, "data_dataset",
        "from flashstudio.pages.data.page import render_data_page\nrender_data_page()",
        {
            "dataset_name": "mini",
            "train_img_path": str(tmp_path / "train"),
            "val_img_path": str(tmp_path / "valid"),
            "num_classes": 2,
            "class_names": "cat\ndog",
        },
    )
    assert [tab.label for tab in at.tabs] == [
        "Upload", "Download", "Preview", "Verify", "Edit", "Fix", "Quality"
    ]


def test_download_finalize_after_data_page(tmp_path):
    source = tmp_path / "downloaded"
    for split in ("images", "labels"):
        (source / split / "train2017").mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (64, 48), "white").save(source / "images" / "train2017" / f"{i}.jpg")
        (source / "labels" / "train2017" / f"{i}.txt").write_text(
            f"{i % 2} 0.5 0.5 0.2 0.2\n"
        )

    body = (
        "from flashstudio.pages.data.page import render_data_page\n"
        "from flashstudio.pages.data.helpers import _finalize_downloaded_dataset\n"
        "render_data_page()\n"
        f"_finalize_downloaded_dataset({str(source)!r}, 'COCO128')\n"
        "st.write('NC', st.session_state.get('num_classes'))\n"
        "st.write('TRAIN', st.session_state.get('train_img_path'))\n"
        "st.write('PENDING_FMT', st.session_state.get('_pending_ann_format'))\n"
    )
    at = _run_probe(tmp_path, "download_finalize", body)
    marks = [m.value for m in at.markdown]
    assert any(m.startswith("NC") and "80" in m for m in marks)
    assert any(m.startswith("TRAIN") and "train" in m for m in marks)
    assert any("PENDING_FMT" in m and "COCO JSON" in m for m in marks)


def test_model_skeleton(tmp_path):
    at = _run_probe(
        tmp_path, "model",
        "from flashstudio.pages.model.page import render_model_page\nrender_model_page()",
        {"model_framework": "FlashDet", "num_classes": 10},
    )
    assert [tab.label for tab in at.tabs] == [
        "Architecture", "Hyperparameters", "Augmentation", "Advanced"
    ]
    assert any("Framework" == radio.label for radio in at.radio)


def test_training_skeleton(tmp_path):
    at = _run_probe(
        tmp_path, "training",
        "from flashstudio.pages.training.page import render_training_page\n"
        "render_training_page()",
        {"training_active": False},
    )
    labels = [tab.label for tab in at.tabs]
    assert labels[:2] == ["Launch", "Monitor"]
    assert "Compare" in labels


def test_dashboard_skeleton(tmp_path):
    at = _run_probe(
        tmp_path, "dashboard",
        "from flashstudio.pages.dashboard import render_dashboard\nrender_dashboard()",
    )
    text = " ".join(m.value for m in at.markdown)
    assert "What to do next" in text
    assert "Pipeline status" in text


def test_missing_dependency_shows_install_option(tmp_path):
    from flashstudio.components.dependencies import dependency_info

    if dependency_info("TensorRT")["installed"]:
        pytest.skip("TensorRT is installed in this environment")

    _write_checkpoint(tmp_path, "Ultralytics")
    at = _run_probe(
        tmp_path,
        "export_missing_dep",
        "from flashstudio.pages.export.page import render_export_page\n"
        "render_export_page()",
        {"export_framework": "Ultralytics"},
    )
    assert any("Install TensorRT" in button.label for button in at.button)
    assert any("TensorRT is not installed" in warning.value for warning in at.warning)
