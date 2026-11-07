"""Dependency install helper tests."""

from flashstudio.components.dependencies import (
    DEPENDENCIES,
    dependency_info,
    install_package,
)


def test_dependency_entries_cover_frameworks():
    for name in ("FlashDet", "Ultralytics", "YOLOX", "ONNX Runtime", "TensorRT"):
        assert name in DEPENDENCIES
        info = dependency_info(name)
        assert info["module"]
        assert info["pip"]
        assert isinstance(info["installed"], bool)


def test_install_package_dry_run_returns_command():
    ok, command = install_package("flashdet[export]", dry_run=True)
    assert ok
    assert "pip" in command
    assert "flashdet[export]" in command
