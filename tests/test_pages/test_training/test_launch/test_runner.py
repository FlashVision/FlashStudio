"""Tests for training launch runner."""

from flashstudio.pages.training.launch.runner import _generate_run_name


def test_generate_run_name_returns_non_empty(mock_session_state):
    name = _generate_run_name()
    assert isinstance(name, str)
    assert len(name) > 0


def test_generate_run_name_contains_model_size_code(mock_session_state):
    mock_session_state["model_arch"] = "FlashDet-Nano"
    name = _generate_run_name()
    assert "nano" in name


def test_generate_run_name_contains_dataset_code(mock_session_state):
    mock_session_state["dataset_name"] = "MyDataset (demo)"
    name = _generate_run_name()
    assert "mydata" in name.lower()


def test_generate_run_name_contains_timestamp(mock_session_state):
    name = _generate_run_name()
    parts = name.split("_")
    assert len(parts) >= 3
    # Timestamp part should have digits (MMDD_HHMM pattern)
    timestamp_part = "_".join(parts[-2:])
    assert any(c.isdigit() for c in timestamp_part)
