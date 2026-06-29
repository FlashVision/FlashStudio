"""Tests for quick_actions module."""

from unittest.mock import patch, MagicMock

import pytest


def test_render_quick_actions_importable():
    from flashstudio.pages.dashboard.overview.quick_actions import render_quick_actions

    assert render_quick_actions is not None


def test_render_quick_actions_callable():
    from flashstudio.pages.dashboard.overview.quick_actions import render_quick_actions

    assert callable(render_quick_actions)
