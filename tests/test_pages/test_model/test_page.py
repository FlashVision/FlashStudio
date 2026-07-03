"""Tests for flashstudio.pages.model.page."""


def test_render_model_page_importable():
    from flashstudio.pages.model.page import render_model_page
    assert render_model_page is not None


def test_render_model_page_callable():
    from flashstudio.pages.model.page import render_model_page
    assert callable(render_model_page)
