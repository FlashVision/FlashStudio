"""Tests for flashstudio.app — application-level helpers and constants."""

import os
import json
import pytest
from unittest.mock import patch, MagicMock


class TestAppSteps:
    def test_steps_defined(self):
        from flashstudio.app import STEPS
        assert isinstance(STEPS, list)
        assert len(STEPS) == 6

    def test_steps_have_required_keys(self):
        from flashstudio.app import STEPS
        for step in STEPS:
            assert "id" in step
            assert "label" in step
            assert "icon" in step

    def test_step_ids(self):
        from flashstudio.app import STEPS
        ids = [s["id"] for s in STEPS]
        assert ids == ["dashboard", "data", "model", "training", "export", "inference"]


class TestPageRenderers:
    def test_all_step_ids_have_renderers(self):
        from flashstudio.app import STEPS, PAGE_RENDERERS
        for step in STEPS:
            assert step["id"] in PAGE_RENDERERS, f"Missing renderer for {step['id']}"

    def test_renderers_are_callable(self):
        from flashstudio.app import PAGE_RENDERERS
        for name, func in PAGE_RENDERERS.items():
            assert callable(func), f"Renderer for {name} is not callable"


class TestEnsureConfigInSession:
    def test_restores_from_file(self, mock_session_state, tmp_dir):
        from flashstudio.app import _ensure_config_in_session

        state = {"epochs": 200, "batch_size": 32, "model_arch": "FlashDet-L"}
        state_file = os.path.join(tmp_dir, "session_state.json")
        with open(state_file, "w") as f:
            json.dump(state, f)

        with patch("flashstudio.components.project_manager.get_project_dir", return_value=tmp_dir):
            _ensure_config_in_session("test")

        assert mock_session_state.get("epochs") == 200
        assert mock_session_state.get("model_arch") == "FlashDet-L"

    def test_does_not_overwrite(self, mock_session_state, tmp_dir):
        from flashstudio.app import _ensure_config_in_session

        mock_session_state["epochs"] = 999

        state = {"epochs": 200, "batch_size": 32}
        state_file = os.path.join(tmp_dir, "session_state.json")
        with open(state_file, "w") as f:
            json.dump(state, f)

        with patch("flashstudio.components.project_manager.get_project_dir", return_value=tmp_dir):
            _ensure_config_in_session("test")

        assert mock_session_state["epochs"] == 999
        assert mock_session_state["batch_size"] == 32

    def test_missing_file_ok(self, mock_session_state, tmp_dir):
        from flashstudio.app import _ensure_config_in_session

        with patch("flashstudio.components.project_manager.get_project_dir", return_value=tmp_dir):
            _ensure_config_in_session("test")

    def test_invalid_json_ok(self, mock_session_state, tmp_dir):
        from flashstudio.app import _ensure_config_in_session

        state_file = os.path.join(tmp_dir, "session_state.json")
        with open(state_file, "w") as f:
            f.write("{{invalid json")

        with patch("flashstudio.components.project_manager.get_project_dir", return_value=tmp_dir):
            _ensure_config_in_session("test")


class TestInitConfigMirror:
    def test_captures_existing_keys(self, mock_session_state):
        from flashstudio.app import _init_config_mirror

        mock_session_state["epochs"] = 100
        mock_session_state["batch_size"] = 8
        mock_session_state["model_arch"] = "FlashDet-N"

        _init_config_mirror()
        mirror = mock_session_state.get("_config_mirror", {})
        assert mirror["epochs"] == 100
        assert mirror["batch_size"] == 8
        assert mirror["model_arch"] == "FlashDet-N"

    def test_skips_missing_keys(self, mock_session_state):
        from flashstudio.app import _init_config_mirror

        mock_session_state["epochs"] = 50
        _init_config_mirror()
        mirror = mock_session_state.get("_config_mirror", {})
        assert "epochs" in mirror
        assert "batch_size" not in mirror

    def test_empty_session(self, mock_session_state):
        from flashstudio.app import _init_config_mirror

        _init_config_mirror()
        assert "_config_mirror" not in mock_session_state


class TestMainImport:
    def test_main_callable(self):
        from flashstudio.app import main
        assert callable(main)
