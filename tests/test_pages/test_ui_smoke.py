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
    assert labels == ["Model", "Data", "Solution", "Run"]
    assert any(radio.label == "Run mode" for radio in at.radio)
    assert any("Framework" == radio.label for radio in at.radio)


def test_inference_live_alias_opens_run_tab(tmp_path):
    _write_checkpoint(tmp_path, "FlashDet")
    at = _run_probe(
        tmp_path,
        "infer_live_alias",
        "import streamlit as st\n"
        "st.session_state['_infer_tab_next'] = 'Live'\n"
        "from flashstudio.pages.inference.page import render_inference_page\n"
        "render_inference_page()\n"
        "st.write('TAB', st.session_state.get('_infer_tab'))\n"
        "st.write('MODE', st.session_state.get('_infer_run_mode'))\n",
        {"infer_family": "FlashDet", "weight_source": "Training output"},
    )
    assert any(
        str(m.value).startswith("TAB") and "Run" in str(m.value)
        for m in at.markdown
    )
    assert any(
        str(m.value).startswith("MODE") and "Live" in str(m.value)
        for m in at.markdown
    )


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
    assert "Checkpoint:" in " ".join(str(m.value) for m in at.markdown)
    assert any("Framework" == radio.label for radio in at.radio)


def test_data_skeleton(tmp_path):
    at = _run_probe(
        tmp_path, "data",
        "from flashstudio.pages.data.page import render_data_page\nrender_data_page()",
    )
    assert [tab.label for tab in at.tabs] == [
        "Load data", "Convert", "Preview", "Verify", "Fix", "Edit", "Quality"
    ]


def test_data_load_tab_combines_upload_and_download(tmp_path):
    at = _run_probe(
        tmp_path, "data_load",
        "from flashstudio.pages.data.page import render_data_page\nrender_data_page()",
    )
    source = [
        r for r in at.radio
        if r.label == "How do you want to load your data?"
    ]
    assert source and list(source[0].options) == ["Upload", "Download"]
    assert source[0].value == "Upload"


def test_data_download_alias_opens_load_tab(tmp_path):
    at = _run_probe(
        tmp_path, "data_download_alias",
        "import streamlit as st\n"
        "st.session_state['_data_tab_next'] = 'Download'\n"
        "from flashstudio.pages.data.page import render_data_page\n"
        "render_data_page()\n"
        "st.write('TAB', st.session_state.get('_data_tab'))\n"
        "st.write('MODE', st.session_state.get('_data_source_mode'))\n",
    )
    assert any(
        str(m.value).startswith("TAB") and "Load data" in str(m.value)
        for m in at.markdown
    )
    assert any(
        str(m.value).startswith("MODE") and "Download" in str(m.value)
        for m in at.markdown
    )


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
        "Load data", "Convert", "Preview", "Verify", "Fix", "Edit",
        "Objects", "Labels", "Quality",
    ]
    assert any(e.label == "Replace dataset" for e in at.expander)


def test_data_workflow_advances_to_preview(tmp_path):
    for split in ("train", "valid"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        Image.new("RGB", (64, 48), "white").save(split_dir / "a.jpg")
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [4, 4, 20, 20], "area": 400,
            }],
        }
        (split_dir / "_annotations.coco.json").write_text(json.dumps(data))

    at = _run_probe(
        tmp_path, "data_advance",
        "import streamlit as st\n"
        "st.session_state['_data_tab_next'] = 'Preview'\n"
        "from flashstudio.pages.data.page import render_data_page\n"
        "render_data_page()\n"
        "st.write('TAB', st.session_state.get('_data_tab'))\n",
        {
            "dataset_name": "mini",
            "train_img_path": str(tmp_path / "train"),
            "val_img_path": str(tmp_path / "valid"),
            "num_classes": 1,
            "class_names": "cat",
        },
    )
    assert any(
        str(m.value).startswith("TAB") and "Preview" in str(m.value)
        for m in at.markdown
    )


def test_inference_workflow_advances_to_run(tmp_path):
    at = _run_probe(
        tmp_path, "infer_advance",
        "import streamlit as st\n"
        "st.session_state['_infer_tab_next'] = 'Run'\n"
        "from flashstudio.pages.inference.page import render_inference_page\n"
        "render_inference_page()\n"
        "st.write('TAB', st.session_state.get('_infer_tab'))\n",
    )
    assert any(
        str(m.value).startswith("TAB") and "Run" in str(m.value)
        for m in at.markdown
    )


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
        "Architecture", "Training options",
        "Hyperparameters", "Augmentation", "Advanced",
    ]
    assert any("Framework" == radio.label for radio in at.radio)


def _training_markdown(at):
    return "\n".join(str(m.value) for m in at.markdown)


def test_training_skeleton(tmp_path):
    at = _run_probe(
        tmp_path, "training",
        "from flashstudio.pages.training.page import render_training_page\n"
        "render_training_page()",
        {"training_active": False},
    )
    labels = [tab.label for tab in at.tabs]
    # One tab bar; only the selected tab body renders its nested tabs.
    assert labels == ["Launch", "Display", "Compare"]
    assert len(at.button_group) == 0


def test_training_cluster_order_ignores_last_status(tmp_path):
    """A previous run's status must not reorder the tabs."""
    at = _run_probe(
        tmp_path,
        "training_complete_clusters",
        "from flashstudio.pages.training.page import render_training_page\n"
        "render_training_page()\n",
        {"training_active": False, "training_status": "Complete"},
    )
    labels = [tab.label for tab in at.tabs]
    assert labels == ["Launch", "Display", "Compare"]


def test_training_live_dashboard_is_inside_monitoring_cluster(tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "training_log.csv").write_text("epoch,lr,train_loss\n1,0.1,5\n")
    at = _run_probe(
        tmp_path,
        "training_live_dashboard",
        "import os\n"
        "import streamlit as st\n"
        "st.session_state.update({\n"
        "    'training_active': True,\n"
        "    'training_pid': os.getpid(),\n"
        f"    'active_run_path': {str(run_dir)!r},\n"
        "    '_training_tab': 'Display',\n"
        "})\n"
        "from flashstudio.pages.training.page import render_training_page\n"
        "render_training_page()\n",
    )
    labels = [tab.label for tab in at.tabs]
    order = {label: i for i, label in enumerate(labels)}
    assert order["Launch"] < order["Display"] < order["Curves"]
    assert (
        order["Curves"] < order["Visualizations"] < order["Ground Truth"]
        < order["Log"] < order["Files"] < order["Compare"]
    )
    # A running job can be cancelled from the status banner on any tab.
    assert any(str(b.label) == "Stop training" for b in at.button)


def test_training_monitoring_empty_state_before_a_run(tmp_path):
    at = _run_probe(
        tmp_path,
        "training_no_run",
        "from flashstudio.pages.training.page import render_training_page\n"
        "render_training_page()\n",
        {"training_active": False, "_training_tab": "Display"},
    )
    infos = [str(item.value) for item in at.get("info")]
    assert any("No run data yet" in msg for msg in infos)
    # The monitoring tabs are always shown under Display, with empty tabs.
    labels = [tab.label for tab in at.tabs]
    order = {label: i for i, label in enumerate(labels)}
    assert order["Launch"] < order["Display"] < order["Curves"]
    assert order["Files"] < order["Compare"]


def test_training_running_status_wins_over_stale_complete(tmp_path):
    """The live process status must override the previous run's status."""
    at = _run_probe(
        tmp_path,
        "training_live_status",
        "from flashstudio.pages.training.page import render_training_page\n"
        "render_training_page()\n",
        {
            "training_active": True,
            "training_status": "Complete",
            # A live PID keeps _check_training_process from reaping the run.
            "training_pid": os.getpid(),
        },
    )
    messages = [str(item.value) for item in at.success]
    assert any("Training in progress" in msg for msg in messages)
    assert not any("Training complete" in msg for msg in messages)


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


def test_data_tools_offer_load_action(tmp_path):
    at = _run_probe(
        tmp_path, "data_empty_tools",
        "from flashstudio.pages.data.page import render_data_page\nrender_data_page()",
    )
    assert any(b.label == "Go to Load data" for b in at.button)


def test_data_health_card_has_no_duplicate_actions(tmp_path):
    for split in ("train", "valid"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        Image.new("RGB", (64, 48), "white").save(split_dir / "a.jpg")
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [4, 4, 20, 20], "area": 400,
            }],
        }
        (split_dir / "_annotations.coco.json").write_text(json.dumps(data))

    at = _run_probe(
        tmp_path, "data_health_edit",
        "from flashstudio.pages.data.page import render_data_page\nrender_data_page()",
        {
            "dataset_name": "mini",
            "train_img_path": str(tmp_path / "train"),
            "val_img_path": str(tmp_path / "valid"),
            "num_classes": 1,
            "class_names": "cat",
        },
    )
    # The health card is informational; the tool tabs below are the only
    # navigation, so the same five actions are not shown twice.
    keys = [b.key for b in at.button]
    for key in (
        "health_preview", "health_verify", "health_fix",
        "health_edit", "health_quality",
    ):
        assert key not in keys
    assert [tab.label for tab in at.tabs] == [
        "Load data", "Convert", "Preview", "Verify", "Fix", "Edit",
        "Objects", "Labels", "Quality",
    ]


def test_model_framework_syncs_export_and_inference(tmp_path):
    at = _run_probe(
        tmp_path, "framework_sync",
        "import streamlit as st\n"
        "st.session_state['model_framework'] = 'Ultralytics'\n"
        "st.session_state['_model_framework_last'] = 'FlashDet'\n"
        "st.session_state['export_framework'] = 'FlashDet'\n"
        "st.session_state['infer_family'] = 'FlashDet'\n"
        "from flashstudio.pages.model.page import _sync_runtime_frameworks\n"
        "_sync_runtime_frameworks()\n"
        "st.write('EXPORT', st.session_state.get('export_framework'))\n"
        "st.write('INFER', st.session_state.get('infer_family'))\n",
    )
    marks = [str(m.value) for m in at.markdown]
    assert any(m.startswith("EXPORT") and "Ultralytics" in m for m in marks)
    assert any(m.startswith("INFER") and "Ultralytics" in m for m in marks)


def test_model_framework_keeps_explicit_page_choice(tmp_path):
    at = _run_probe(
        tmp_path, "framework_sync_override",
        "import streamlit as st\n"
        "st.session_state['model_framework'] = 'Ultralytics'\n"
        "st.session_state['_model_framework_last'] = 'FlashDet'\n"
        "st.session_state['export_framework'] = 'YOLOX'\n"
        "st.session_state['infer_family'] = 'FlashDet'\n"
        "from flashstudio.pages.model.page import _sync_runtime_frameworks\n"
        "_sync_runtime_frameworks()\n"
        "st.write('EXPORT', st.session_state.get('export_framework'))\n"
        "st.write('INFER', st.session_state.get('infer_family'))\n",
    )
    marks = [str(m.value) for m in at.markdown]
    assert any(m.startswith("EXPORT") and "YOLOX" in m for m in marks)
    assert any(m.startswith("INFER") and "Ultralytics" in m for m in marks)


def test_dashboard_running_training_is_not_complete(tmp_path):
    at = _run_probe(
        tmp_path, "dash_running",
        "from flashstudio.pages.dashboard.overview.quick_actions import "
        "render_quick_actions\n"
        "from flashstudio.pages.dashboard.overview.pipeline_status import "
        "render_pipeline_status\n"
        "render_quick_actions()\n"
        "render_pipeline_status()\n",
        {"training_status": "Running"},
    )
    captions = [str(c.value) for c in at.caption]
    assert any("1 of 5 steps complete" in c for c in captions)
    assert any("1 of 4 stages ready" in c for c in captions)


def test_inference_run_empty_state_offers_data_action(tmp_path):
    _write_checkpoint(tmp_path, "FlashDet")
    at = _run_probe(
        tmp_path, "infer_run_empty",
        "import streamlit as st\n"
        "st.session_state['_infer_tab'] = 'Run'\n"
        "from flashstudio.pages.inference.page import render_inference_page\n"
        "render_inference_page()\n",
        {"infer_family": "FlashDet", "weight_source": "Training output"},
    )
    assert any(b.label == "Add input in Data" for b in at.button)


def test_training_compare_cluster_empty_state(tmp_path):
    at = _run_probe(
        tmp_path, "train_compare_empty",
        "from flashstudio.pages.training.page import render_training_page\n"
        "render_training_page()\n",
        {"training_active": False, "_training_tab": "Compare"},
    )
    labels = [tab.label for tab in at.tabs]
    assert "Compare" in labels
    infos = [str(item.value) for item in at.get("info")]
    assert any("No runs found in the workspace" in msg for msg in infos)


def test_preview_supports_validation_only_dataset(tmp_path):
    valid = tmp_path / "valid"
    valid.mkdir()
    Image.new("RGB", (64, 48), "white").save(valid / "a.jpg")
    data = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": [{
            "id": 1, "image_id": 1, "category_id": 1,
            "bbox": [4, 4, 20, 20], "area": 400,
        }],
    }
    (valid / "_annotations.coco.json").write_text(json.dumps(data))

    at = _run_probe(
        tmp_path, "data_valid_only",
        "from flashstudio.pages.data.page import render_data_page\nrender_data_page()",
        {
            "dataset_name": "mini",
            "val_img_path": str(valid),
            "num_classes": 1,
            "class_names": "cat",
        },
    )
    split = [r for r in at.radio if r.label == "Split"]
    assert split and list(split[0].options) == ["Validation"]


def test_missing_config_keys_prefer_fresh_mirror(tmp_path):
    at = _run_probe(
        tmp_path, "mirror_seed",
        "import streamlit as st\n"
        "st.session_state['_config_mirror'] = {\n"
        "    'model_framework': 'FlashDet', 'epochs': 42,\n"
        "}\n"
        "from flashstudio.app import _ensure_config_in_session\n"
        "_ensure_config_in_session('missing_project')\n"
        "st.write('FW', st.session_state.get('model_framework'))\n"
        "st.write('EPOCHS', st.session_state.get('epochs'))\n",
    )
    marks = [str(m.value) for m in at.markdown]
    assert any(m.startswith("FW") and "FlashDet" in m for m in marks)
    assert any(m.startswith("EPOCHS") and "42" in m for m in marks)


def test_data_load_does_not_clear_paths_from_stale_widgets(tmp_path):
    for split in ("train", "valid"):
        d = tmp_path / split
        d.mkdir()
        Image.new("RGB", (64, 48), "white").save(d / "a.jpg")
        data = {
            "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
            "categories": [{"id": 1, "name": "cat"}],
            "annotations": [{
                "id": 1, "image_id": 1, "category_id": 1,
                "bbox": [4, 4, 20, 20], "area": 400,
            }],
        }
        (d / "_annotations.coco.json").write_text(json.dumps(data))

    train_path = str(tmp_path / "train")
    val_path = str(tmp_path / "valid")
    at = _run_probe(
        tmp_path, "data_stale_paths",
        "from flashstudio.pages.data.page import render_data_page\n"
        "render_data_page()\n"
        "import streamlit as st\n"
        "st.write('TRAIN', st.session_state.get('train_img_path'))\n"
        "st.write('VAL', st.session_state.get('val_img_path'))\n",
        {
            "dataset_name": "mini",
            "train_img_path": train_path,
            "val_img_path": val_path,
            # Widget values left empty while markers already match the paths.
            "_train_img_input": "",
            "_train_img_input_synced": train_path,
            "_val_img_input": "",
            "_val_img_input_synced": val_path,
            "num_classes": 1,
            "class_names": "cat",
        },
    )
    marks = [str(m.value) for m in at.markdown]
    assert any(m.startswith("TRAIN") and train_path in m for m in marks)
    assert any(m.startswith("VAL") and val_path in m for m in marks)


def test_preview_uses_fixed_two_image_pages(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    images, anns = [], []
    for i in range(3):
        name = f"a{i}.jpg"
        Image.new("RGB", (80 + i * 40, 60), "white").save(train / name)
        images.append({"id": i + 1, "file_name": name,
                       "width": 80 + i * 40, "height": 60})
        anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1,
                     "bbox": [4, 4, 20, 20], "area": 400})
    data = {
        "images": images,
        "categories": [{"id": 1, "name": "cat"}],
        "annotations": anns,
    }
    (train / "_annotations.coco.json").write_text(json.dumps(data))

    at = _run_probe(
        tmp_path, "data_preview_pages",
        "import streamlit as st\n"
        "st.session_state['_data_tab'] = 'Preview'\n"
        "from flashstudio.pages.data.page import render_data_page\n"
        "render_data_page()\n",
        {
            "dataset_name": "mini",
            "train_img_path": str(train),
            "num_classes": 1,
            "class_names": "cat",
        },
    )
    assert not any(s.label == "Count" for s in at.slider)
    text = " ".join(str(m.value) for m in at.markdown)
    text += " " + " ".join(str(c.value) for c in at.caption)
    assert "1–2 of 3" in text
    assert any(
        ni.label == "Image #" for ni in at.number_input
    ), "jump box lets you reach an image directly"


def test_download_root_honours_custom_folder(tmp_path):
    custom = tmp_path / "my datasets"
    at = _run_probe(
        tmp_path, "download_root_custom",
        "import streamlit as st\n"
        "from flashstudio.pages.data._common import _resolve_download_root\n"
        "st.write('ROOT', _resolve_download_root())\n"
        "st.session_state.pop('download_output_dir', None)\n"
        "st.write('DEFAULT', _resolve_download_root())\n",
        {"download_output_dir": str(custom)},
    )
    marks = [str(m.value) for m in at.markdown]
    assert any(m.startswith("ROOT") and str(custom) in m for m in marks)
    assert any(m.startswith("DEFAULT") and m.endswith("datasets") for m in marks)


def test_download_tab_has_location_control(tmp_path):
    at = _run_probe(
        tmp_path, "download_location",
        "from flashstudio.pages.data.download.tab import _render_download\n"
        "_render_download()\n",
    )
    assert any(t.label == "Save datasets to" for t in at.text_input)
    captions = " ".join(str(c.value) for c in at.caption)
    assert "Datasets save to" in captions
    assert any(b.label == "Reset" for b in at.button)
