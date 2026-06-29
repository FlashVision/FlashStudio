"""Tests for flashstudio.utils.device."""

import pytest
from unittest.mock import patch, MagicMock
from flashstudio.utils.device import (
    is_colab, has_cuda, get_device, get_gpu_info, get_colab_runtime_type,
)


class TestIsColab:
    def test_not_colab_locally(self):
        assert is_colab() is False

    @patch.dict("sys.modules", {"google.colab": MagicMock()})
    def test_is_colab_when_module_exists(self):
        assert is_colab() is True


class TestHasCuda:
    def test_returns_bool(self):
        result = has_cuda()
        assert isinstance(result, bool)


class TestGetDevice:
    def test_returns_valid_device(self):
        device = get_device()
        assert device in ("cpu", "cuda", "mps")


class TestGetGpuInfo:
    def test_returns_dict(self):
        info = get_gpu_info()
        assert isinstance(info, dict)
        assert "available" in info
        assert "name" in info
        assert "memory_total" in info
        assert "memory_used" in info

    def test_cpu_fallback(self):
        with patch.dict("sys.modules", {"torch": None}):
            info = get_gpu_info()
            assert info["name"] == "CPU"


class TestGetColabRuntimeType:
    def test_local_when_not_colab(self):
        assert get_colab_runtime_type() == "local"
