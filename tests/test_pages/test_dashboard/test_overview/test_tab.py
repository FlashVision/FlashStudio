"""Tests for overview tab module."""

from unittest.mock import patch, MagicMock

import pytest


def test_render_overview_importable():
    from flashstudio.pages.dashboard.overview.tab import render_overview

    assert render_overview is not None


def test_render_overview_callable():
    from flashstudio.pages.dashboard.overview.tab import render_overview

    assert callable(render_overview)
