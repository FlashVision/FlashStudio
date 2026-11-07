"""TAR extraction safety tests."""

import tarfile

import pytest

from flashstudio.pages.data.helpers import _validate_tar_members


def test_rejects_path_traversal():
    member = tarfile.TarInfo("../../escaped.txt")
    with pytest.raises(ValueError):
        _validate_tar_members([member], "/tmp/flashstudio_extract")


def test_accepts_normal_member():
    member = tarfile.TarInfo("dataset/train/image.jpg")
    _validate_tar_members([member], "/tmp/flashstudio_extract")
