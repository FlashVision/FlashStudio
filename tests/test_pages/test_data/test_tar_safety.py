"""Archive extraction safety tests."""

import tarfile
import zipfile

import pytest

from flashstudio.pages.data.helpers import _validate_tar_members, _validate_zip_members


def test_rejects_path_traversal():
    member = tarfile.TarInfo("../../escaped.txt")
    with pytest.raises(ValueError):
        _validate_tar_members([member], "/tmp/flashstudio_extract")


def test_accepts_normal_member():
    member = tarfile.TarInfo("dataset/train/image.jpg")
    _validate_tar_members([member], "/tmp/flashstudio_extract")


def test_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "x")
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(ValueError):
            _validate_zip_members(zf.infolist(), str(tmp_path / "out"))


def test_zip_accepts_normal_member(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("dataset/train/image.jpg", "x")
    with zipfile.ZipFile(archive) as zf:
        _validate_zip_members(zf.infolist(), str(tmp_path / "out"))
