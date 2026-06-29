"""Tests for flashstudio.pages.inference.page — inference page entry point."""

import pytest


class TestImports:
    def test_render_inference_page_importable(self):
        from flashstudio.pages.inference.page import render_inference_page
        assert render_inference_page is not None


class TestCallable:
    def test_render_inference_page_callable(self):
        from flashstudio.pages.inference.page import render_inference_page
        assert callable(render_inference_page)
