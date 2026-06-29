"""Tests for flashstudio.pages.data.upload.tab."""

from flashstudio.pages.data.upload.tab import (
    _render_class_config,
    _render_conversion_hint,
    _render_upload,
)


def test_render_upload_callable():
    assert callable(_render_upload)


def test_render_class_config_callable():
    assert callable(_render_class_config)


def test_render_conversion_hint_callable():
    assert callable(_render_conversion_hint)
