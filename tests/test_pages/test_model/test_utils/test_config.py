"""Tests for flashstudio.pages.model.utils.config."""

from flashstudio.constants import (
    ARCH_FAMILIES,
    DEFAULT_MODEL_ARCH,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_LR,
    TRAIN_IMG_SIZE,
)


def test_build_cfg_returns_dict(mock_session_state):
    from flashstudio.pages.model.utils.config import _build_cfg

    result = _build_cfg()
    assert isinstance(result, dict)


def test_build_cfg_has_required_keys(mock_session_state):
    from flashstudio.pages.model.utils.config import _build_cfg

    result = _build_cfg()
    for key in ("model", "training", "augmentation", "advanced"):
        assert key in result, f"Missing key: {key}"


def test_build_cfg_defaults_from_constants(mock_session_state):
    from flashstudio.pages.model.utils.config import _build_cfg

    result = _build_cfg()
    assert result["model"]["family"] == ARCH_FAMILIES[0]
    assert result["model"]["variant"] == DEFAULT_MODEL_ARCH
    assert result["training"]["epochs"] == TRAIN_EPOCHS
    assert result["training"]["batch_size"] == TRAIN_BATCH_SIZE
    assert result["training"]["lr"] == TRAIN_LR
    assert result["training"]["img_size"] == TRAIN_IMG_SIZE


def test_build_cfg_respects_session_state(mock_session_state):
    from flashstudio.pages.model.utils.config import _build_cfg

    mock_session_state["epochs"] = 50
    mock_session_state["batch_size"] = 32
    mock_session_state["lr"] = 0.01
    mock_session_state["img_size"] = 640

    result = _build_cfg()
    assert result["training"]["epochs"] == 50
    assert result["training"]["batch_size"] == 32
    assert result["training"]["lr"] == 0.01
    assert result["training"]["img_size"] == 640


def test_apply_cfg_sets_model_keys(mock_session_state):
    from flashstudio.pages.model.utils.config import _apply_cfg
    from unittest.mock import patch

    config = {
        "model": {
            "family": "YOLOv8",
            "variant": "YOLOv8-S",
            "pretrained": "COCO pretrained",
        }
    }
    with patch("flashstudio.pages.model.utils.config.apply_training_config"):
        _apply_cfg(config)

    assert mock_session_state["arch_family"] == "YOLOv8"
    assert mock_session_state["model_arch"] == "YOLOv8-S"
    assert mock_session_state["pretrain_option"] == "COCO pretrained"


def test_apply_cfg_calls_apply_training_config(mock_session_state):
    from flashstudio.pages.model.utils.config import _apply_cfg
    from unittest.mock import patch

    config = {"training": {"epochs": 200}}
    with patch("flashstudio.pages.model.utils.config.apply_training_config") as mock_apply:
        _apply_cfg(config)
        mock_apply.assert_called_once_with(config)
