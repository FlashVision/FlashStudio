"""Tests for user-configurable storage locations."""

import json
import os

import pytest

from flashstudio.utils import storage


@pytest.fixture()
def storage_env(tmp_path, monkeypatch):
    """Isolate storage settings and project folders inside tmp_path."""
    base = tmp_path / "home" / ".flashstudio"
    base.mkdir(parents=True)
    monkeypatch.setattr(storage, "BASE_DIR", str(base))
    monkeypatch.setattr(storage, "STORAGE_FILE", str(base / "storage.json"))
    monkeypatch.setattr(storage, "DEFAULT_WEIGHTS_DIR", str(base / "weights"))

    project = tmp_path / "project"
    data = project / "data"
    runs = project / "runs"
    data.mkdir(parents=True)
    runs.mkdir(parents=True)
    monkeypatch.setattr(storage, "_project_data_dir", lambda: str(data))
    monkeypatch.setattr(storage, "_project_runs_dir", lambda: str(runs))
    return {"base": base, "data": data, "runs": runs, "tmp": tmp_path}


class TestCleanPath:
    def test_empty_returns_empty(self):
        assert storage.clean_path("") == ""
        assert storage.clean_path(None) == ""

    def test_expands_user(self, monkeypatch):
        monkeypatch.setattr(storage.os.path, "expanduser", lambda p: "/home/u" + p[1:])
        assert storage.clean_path("~/datasets") == os.path.abspath("/home/u/datasets")

    def test_relative_resolves_against_base(self, storage_env):
        result = storage.clean_path("my-data", base=str(storage_env["data"]))
        assert result == os.path.abspath(str(storage_env["data"] / "my-data"))

    def test_absolute_is_kept(self, storage_env):
        absolute = str(storage_env["tmp"] / "absolute")
        assert storage.clean_path(absolute, base=str(storage_env["data"])) == absolute


class TestDefaults:
    def test_dataset_default_inside_project(self, storage_env):
        assert storage.default_datasets_root() == str(storage_env["data"] / "datasets")

    def test_runs_default_inside_project(self, storage_env):
        assert storage.default_runs_root() == str(storage_env["runs"])

    def test_weights_default_in_home(self, storage_env):
        assert storage.get_weights_root() == str(storage_env["base"] / "weights")


class TestDatasetOverride:
    def test_set_and_get(self, storage_env, mock_session_state):
        custom = storage_env["tmp"] / "datasets-drive"
        resolved = storage.set_datasets_root(str(custom))
        assert resolved == str(custom.resolve())
        assert storage.get_datasets_root() == str(custom.resolve())
        assert mock_session_state["download_output_dir"] == str(custom.resolve())

    def test_empty_uses_default(self, storage_env, mock_session_state):
        assert storage.set_datasets_root("") == storage.default_datasets_root()
        assert storage.get_datasets_root() == storage.default_datasets_root()

    def test_relative_uses_project_data(self, storage_env, mock_session_state):
        storage.set_datasets_root("relative-datasets")
        assert storage.get_datasets_root() == str(
            storage_env["data"] / "relative-datasets"
        )


class TestRunsOverride:
    def test_set_and_get(self, storage_env, mock_session_state):
        custom = storage_env["tmp"] / "runs-drive"
        resolved = storage.set_runs_root(str(custom))
        assert resolved == str(custom.resolve())
        assert storage.get_runs_root() == str(custom.resolve())
        assert mock_session_state["save_dir"] == str(custom.resolve())

    def test_updates_config_mirror(self, storage_env, mock_session_state):
        mock_session_state["_config_mirror"] = {"epochs": 10}
        custom = storage_env["tmp"] / "mirror-runs"
        storage.set_runs_root(str(custom))
        assert mock_session_state["_config_mirror"]["save_dir"] == str(custom.resolve())


class TestWeightsOverride:
    def test_persists_to_storage_file(self, storage_env):
        custom = storage_env["tmp"] / "weights-drive"
        assert storage.set_weights_root(str(custom)) == str(custom.resolve())

        with open(storage.STORAGE_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["weights_root"] == str(custom.resolve())
        assert storage.get_weights_root() == str(custom.resolve())

    def test_default_removes_override(self, storage_env):
        storage.set_weights_root(str(storage_env["tmp"] / "weights-drive"))
        storage.set_weights_root("")
        assert storage.get_weights_root() == storage.DEFAULT_WEIGHTS_DIR
        with open(storage.STORAGE_FILE, "r", encoding="utf-8") as f:
            assert "weights_root" not in json.load(f)

    def test_ensure_weights_root_creates_folder(self, storage_env):
        custom = storage_env["tmp"] / "new-weights"
        storage.set_weights_root(str(custom))
        assert storage.ensure_weights_root() == str(custom.resolve())
        assert custom.is_dir()


class TestEnsureDir:
    def test_creates_missing_folder(self, storage_env):
        target = storage_env["tmp"] / "new" / "nested"
        ok, message = storage.ensure_dir(str(target))
        assert ok and target.is_dir() and message == ""

    def test_existing_folder_is_ok(self, storage_env):
        ok, _ = storage.ensure_dir(str(storage_env["data"]))
        assert ok

    def test_file_path_is_rejected(self, storage_env):
        target = storage_env["tmp"] / "not-a-folder"
        target.write_text("x")
        ok, message = storage.ensure_dir(str(target))
        assert not ok
        assert "not a folder" in message
