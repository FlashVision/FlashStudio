"""Training manifest construction tests."""

from flashstudio.pages.training.launch.runner import _build_training_params


def test_build_training_params_reads_model_state(mock_session_state):
    mock_session_state.update({
        "arch_family": "FlashDet (recommended)",
        "model_arch": "FlashDet-Pico",
        "img_size": 320,
        "class_names": "person\ncar",
        "num_classes": 2,
        "epochs": 7,
        "batch_size": 4,
        "optimizer": "SGD",
        "lora_targets": ["backbone", "neck"],
    })
    params = _build_training_params("/tmp/run")
    assert params["model_size"] == "p"
    assert params["architecture"] == "flashdet"
    assert params["num_classes"] == 2
    assert params["class_names"] == ["person", "car"]
    assert params["epochs"] == 7
    assert params["batch_size"] == 4
    assert params["optimizer"] == "SGD"
    assert params["save_dir"] == "/tmp/run"
    assert params["lora_targets"] == ["backbone", "neck"]


def test_build_training_params_maps_yolo(mock_session_state):
    mock_session_state.update({
        "arch_family": "YOLOv8",
        "model_arch": "FlashDet-Nano",
        "yolo_width_mult": 0.5,
        "yolo_depth_mult": 0.33,
    })
    params = _build_training_params("/tmp/run")
    assert params["architecture"] == "yolov8"
    assert params["yolo_width_mult"] == 0.5
    assert params["yolo_depth_mult"] == 0.33


def test_checkpoint_metadata_patch_adds_yolo_multipliers():
    from types import SimpleNamespace
    from flashstudio.utils.trainer_launcher import _patch_checkpoint_metadata

    captured = {}

    def save_inference_weights(model, path, config=None, half=False):
        captured["config"] = config

    module = SimpleNamespace(save_inference_weights=save_inference_weights)
    _patch_checkpoint_metadata(module, {
        "yolo_width_mult": 0.5, "yolo_depth_mult": 0.25,
        "yolo_use_pgi": False, "yolo_use_psa": True, "yolo_use_c2psa": True,
    })
    module.save_inference_weights(None, "x.pth", config={"architecture": "yolov8"})
    assert captured["config"]["width_mult"] == 0.5
    assert captured["config"]["depth_mult"] == 0.25
    assert captured["config"]["use_pgi"] is False

