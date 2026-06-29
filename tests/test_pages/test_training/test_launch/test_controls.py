"""Tests for training launch controls."""

from flashstudio.pages.training.launch.controls import (
    _stop_training,
    _pause_training,
    _resume_active_training,
)


def test_stop_training_importable():
    assert _stop_training is not None


def test_pause_training_importable():
    assert _pause_training is not None


def test_resume_active_training_importable():
    assert _resume_active_training is not None


def test_all_callable():
    assert callable(_stop_training)
    assert callable(_pause_training)
    assert callable(_resume_active_training)
