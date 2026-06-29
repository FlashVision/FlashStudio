"""Tests for flashstudio.pages.training.launch.controls — training process control."""

import pytest


class TestControlImports:
    def test_stop_training_importable(self):
        from flashstudio.pages.training.launch.controls import _stop_training
        assert callable(_stop_training)

    def test_pause_training_importable(self):
        from flashstudio.pages.training.launch.controls import _pause_training
        assert callable(_pause_training)

    def test_resume_active_training_importable(self):
        from flashstudio.pages.training.launch.controls import _resume_active_training
        assert callable(_resume_active_training)
