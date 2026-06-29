"""Tests for training monitor ground truth verification."""

from flashstudio.pages.training.monitor.gt_verification import _render_gt_verification


def test_render_gt_verification_importable():
    assert _render_gt_verification is not None


def test_render_gt_verification_callable():
    assert callable(_render_gt_verification)
