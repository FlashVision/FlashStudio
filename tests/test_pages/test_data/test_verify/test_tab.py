"""Tests for flashstudio.pages.data.verify.tab."""

from flashstudio.pages.data.verify.tab import _render_verify


def test_render_verify_callable():
    assert callable(_render_verify)
