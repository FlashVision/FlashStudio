"""Tests for flashstudio.pages.training.launch.runner — run name generation."""

from datetime import datetime


class TestGenerateRunName:
    def test_default_name(self, mock_session_state):
        from flashstudio.pages.training.launch.runner import _generate_run_name

        name = _generate_run_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_includes_model_code(self, mock_session_state):
        from flashstudio.pages.training.launch.runner import _generate_run_name

        mock_session_state["model_arch"] = "FlashDet-Large"
        name = _generate_run_name()
        assert "larg" in name

    def test_includes_dataset_code(self, mock_session_state):
        from flashstudio.pages.training.launch.runner import _generate_run_name

        mock_session_state["dataset_name"] = "Traffic Detection"
        name = _generate_run_name()
        assert "traffic" in name.lower()

    def test_includes_timestamp(self, mock_session_state):
        from flashstudio.pages.training.launch.runner import _generate_run_name

        name = _generate_run_name()
        now = datetime.now()
        assert now.strftime("%m%d") in name

    def test_custom_when_no_dataset(self, mock_session_state):
        from flashstudio.pages.training.launch.runner import _generate_run_name

        name = _generate_run_name()
        assert "custom" in name


class TestRunnerImports:
    def test_start_training_importable(self):
        from flashstudio.pages.training.launch.runner import _start_training
        assert callable(_start_training)
