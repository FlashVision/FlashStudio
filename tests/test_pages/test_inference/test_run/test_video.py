"""Tests for flashstudio.pages.inference.run.video — video processing utilities."""

import os
import tempfile

import pytest
from unittest.mock import patch, MagicMock


class TestImports:
    def test_run_video_importable(self):
        from flashstudio.pages.inference.run.video import _run_video
        assert _run_video is not None

    def test_run_demo_video_importable(self):
        from flashstudio.pages.inference.run.video import _run_demo_video
        assert _run_demo_video is not None

    def test_get_output_path_importable(self):
        from flashstudio.pages.inference.run.video import _get_output_path
        assert _get_output_path is not None

    def test_run_flashdet_video_importable(self):
        from flashstudio.pages.inference.run.video import _run_flashdet_video
        assert _run_flashdet_video is not None

    def test_create_solution_importable(self):
        from flashstudio.pages.inference.run.video import _create_solution
        assert _create_solution is not None


class TestCallable:
    def test_run_video_callable(self):
        from flashstudio.pages.inference.run.video import _run_video
        assert callable(_run_video)

    def test_run_demo_video_callable(self):
        from flashstudio.pages.inference.run.video import _run_demo_video
        assert callable(_run_demo_video)

    def test_get_output_path_callable(self):
        from flashstudio.pages.inference.run.video import _get_output_path
        assert callable(_get_output_path)

    def test_run_flashdet_video_callable(self):
        from flashstudio.pages.inference.run.video import _run_flashdet_video
        assert callable(_run_flashdet_video)

    def test_create_solution_callable(self):
        from flashstudio.pages.inference.run.video import _create_solution
        assert callable(_create_solution)


class TestGetOutputPath:
    def test_returns_string(self, mock_session_state):
        from flashstudio.pages.inference.run.video import _get_output_path
        result = _get_output_path("output.mp4")
        assert isinstance(result, str)

    def test_contains_filename(self, mock_session_state):
        from flashstudio.pages.inference.run.video import _get_output_path
        result = _get_output_path("output.mp4")
        assert result.endswith("output.mp4")

    def test_falls_back_to_tempdir(self, mock_session_state):
        from flashstudio.pages.inference.run.video import _get_output_path
        result = _get_output_path("result.mp4")
        assert tempfile.gettempdir() in result

    def test_uses_save_dir_when_valid(self, mock_session_state, tmp_dir):
        from flashstudio.pages.inference.run.video import _get_output_path
        mock_session_state["save_dir"] = tmp_dir
        result = _get_output_path("output.mp4")
        assert result == os.path.join(tmp_dir, "output.mp4")

    def test_unique_filenames_with_different_inputs(self, mock_session_state):
        from flashstudio.pages.inference.run.video import _get_output_path
        path_a = _get_output_path("video_a.mp4")
        path_b = _get_output_path("video_b.mp4")
        assert path_a != path_b

    def test_ignores_invalid_save_dir(self, mock_session_state):
        from flashstudio.pages.inference.run.video import _get_output_path
        mock_session_state["save_dir"] = "/nonexistent_dir_12345"
        result = _get_output_path("output.mp4")
        assert tempfile.gettempdir() in result
