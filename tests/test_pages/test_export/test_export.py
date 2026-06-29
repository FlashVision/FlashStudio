"""Tests for flashstudio.pages.export — config, settings, imports."""

import pytest
from unittest.mock import patch, MagicMock


class TestExportPageImports:
    def test_render_export_page(self):
        from flashstudio.pages.export import render_export_page
        assert callable(render_export_page)

    def test_page_module(self):
        from flashstudio.pages.export.page import render_export_page
        assert callable(render_export_page)


class TestExportWeightsSettings:
    def test_get_export_save_dir_from_session(self, mock_session_state):
        from flashstudio.pages.export.weights.settings import _get_export_save_dir

        mock_session_state["save_dir"] = "/custom/export/dir"
        assert _get_export_save_dir() == "/custom/export/dir"

    def test_get_export_save_dir_fallback(self, mock_session_state):
        from flashstudio.pages.export.weights.settings import _get_export_save_dir
        from flashstudio.constants import DEFAULT_SAVE_DIR

        result = _get_export_save_dir()
        assert result == DEFAULT_SAVE_DIR

    def test_render_export_settings_importable(self):
        from flashstudio.pages.export.weights.settings import render_export_settings
        assert callable(render_export_settings)


class TestExportEngineImport:
    def test_run_export_importable(self):
        from flashstudio.pages.export.weights.exporter import _run_export
        assert callable(_run_export)


class TestExportConstants:
    def test_export_formats(self):
        from flashstudio.constants import EXPORT_FORMATS
        assert "ONNX" in EXPORT_FORMATS
        assert "TorchScript" in EXPORT_FORMATS

    def test_export_weight_map(self):
        from flashstudio.constants import EXPORT_WEIGHT_MAP
        assert "Best (inference)" in EXPORT_WEIGHT_MAP
        assert "Last" in EXPORT_WEIGHT_MAP
        assert isinstance(EXPORT_WEIGHT_MAP["Best (inference)"], list)

    def test_export_img_sizes(self):
        from flashstudio.constants import EXPORT_IMG_SIZES
        assert 320 in EXPORT_IMG_SIZES
        assert 640 in EXPORT_IMG_SIZES
        assert EXPORT_IMG_SIZES == sorted(EXPORT_IMG_SIZES)

    def test_export_opset_range(self):
        from flashstudio.constants import EXPORT_OPSET_MIN, EXPORT_OPSET_MAX, EXPORT_OPSET_DEFAULT
        assert EXPORT_OPSET_MIN < EXPORT_OPSET_MAX
        assert EXPORT_OPSET_MIN <= EXPORT_OPSET_DEFAULT <= EXPORT_OPSET_MAX
