"""Tests for metrics module."""

from unittest.mock import patch, MagicMock

import pytest


def test_render_metrics_importable():
    from flashstudio.pages.dashboard.overview.metrics import render_metrics

    assert render_metrics is not None


def test_render_metrics_callable():
    from flashstudio.pages.dashboard.overview.metrics import render_metrics

    assert callable(render_metrics)
