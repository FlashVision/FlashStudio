"""Tests for training page."""

from flashstudio.pages.training.page import render_training_page


def test_render_training_page_importable():
    assert render_training_page is not None


def test_render_training_page_callable():
    assert callable(render_training_page)
