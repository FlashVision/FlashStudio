"""Tests for flashstudio.pages.model.utils.config — config build/apply helpers."""

import pytest
from unittest.mock import patch, MagicMock


class TestModelConfigBuild:
    def test_build_cfg_returns_dict(self, mock_session_state):
        from flashstudio.pages.model.utils.config import _build_cfg

        mock_session_state["arch_family"] = "FlashDet (recommended)"
        mock_session_state["model_arch"] = "FlashDet-S"
        mock_session_state["epochs"] = 100
        mock_session_state["batch_size"] = 16
        mock_session_state["lr"] = 0.001
        mock_session_state["img_size"] = 320

        cfg = _build_cfg()
        assert isinstance(cfg, dict)
        assert "model" in cfg
        assert "training" in cfg
        assert "augmentation" in cfg
        assert "advanced" in cfg

    def test_build_cfg_captures_model(self, mock_session_state):
        from flashstudio.pages.model.utils.config import _build_cfg

        mock_session_state["arch_family"] = "FlashDet (recommended)"
        mock_session_state["model_arch"] = "FlashDet-L"

        cfg = _build_cfg()
        assert cfg["model"]["family"] == "FlashDet (recommended)"
        assert cfg["model"]["variant"] == "FlashDet-L"

    def test_build_cfg_uses_defaults(self, mock_session_state):
        from flashstudio.pages.model.utils.config import _build_cfg
        from flashstudio.constants import TRAIN_EPOCHS, TRAIN_BATCH_SIZE

        cfg = _build_cfg()
        assert cfg["training"]["epochs"] == TRAIN_EPOCHS
        assert cfg["training"]["batch_size"] == TRAIN_BATCH_SIZE


class TestModelConfigApply:
    def test_apply_cfg_sets_model(self, mock_session_state):
        from flashstudio.pages.model.utils.config import _apply_cfg

        config = {
            "model": {"family": "YOLOv8", "variant": "YOLOv8-S"},
            "training": {"epochs": 250},
        }
        _apply_cfg(config)
        assert mock_session_state.get("arch_family") == "YOLOv8"
        assert mock_session_state.get("model_arch") == "YOLOv8-S"

    def test_apply_cfg_handles_empty_model(self, mock_session_state):
        from flashstudio.pages.model.utils.config import _apply_cfg

        _apply_cfg({"training": {"epochs": 50}})
        assert mock_session_state.get("epochs") == 50


class TestModelPageImports:
    def test_render_model_page(self):
        from flashstudio.pages.model import render_model_page
        assert callable(render_model_page)

    def test_page_module(self):
        from flashstudio.pages.model.page import render_model_page
        assert callable(render_model_page)

    def test_architecture_tab(self):
        from flashstudio.pages.model.architecture.tab import _tab_arch
        assert callable(_tab_arch)

    def test_hyperparams_tab(self):
        from flashstudio.pages.model.hyperparams.tab import _tab_hyper
        assert callable(_tab_hyper)

    def test_augment_tab(self):
        from flashstudio.pages.model.augment.tab import _tab_aug
        assert callable(_tab_aug)

    def test_advanced_tab(self):
        from flashstudio.pages.model.advanced.tab import _tab_adv
        assert callable(_tab_adv)
