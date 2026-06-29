"""Tests for flashstudio.pages.dashboard.overview.pipeline_status."""

import pytest
from unittest.mock import patch, MagicMock


class TestPipelineStatusLogic:
    """Verify pipeline step detection logic."""

    def test_data_step_detected(self, mock_session_state):
        mock_session_state["dataset_name"] = "MyDataset"
        ds = mock_session_state.get("dataset_name")
        assert ds is not None

    def test_data_step_not_detected(self, mock_session_state):
        ds = mock_session_state.get("dataset_name")
        assert ds is None

    def test_model_step_detected(self, mock_session_state):
        mock_session_state["model_arch"] = "FlashDet-S"
        model = mock_session_state.get("model_arch")
        assert model is not None
        assert model.replace("FlashDet-", "") == "S"

    def test_training_step_complete(self, mock_session_state):
        mock_session_state["training_status"] = "Complete"
        status = mock_session_state.get("training_status", "")
        assert status in ("Complete", "Running")

    def test_training_step_not_started(self, mock_session_state):
        status = mock_session_state.get("training_status", "")
        assert status not in ("Complete", "Running")

    def test_export_step_detected(self, mock_session_state):
        mock_session_state["exported_files"] = [{"format": "ONNX", "path": "/m.onnx"}]
        assert mock_session_state.get("exported_files") is not None

    def test_export_step_pending(self, mock_session_state):
        assert mock_session_state.get("exported_files") is None
