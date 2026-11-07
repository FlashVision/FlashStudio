"""Project isolation: state is cleared/rebuilt instead of leaking between projects."""

import os

import flashstudio.components.project_manager as pm


def _patch_roots(monkeypatch, tmp_path):
    monkeypatch.setattr(pm, "PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setattr(pm, "PROJECTS_INDEX", str(tmp_path / "projects.json"))


def test_clear_removes_config_and_mirror(mock_session_state, monkeypatch, tmp_path):
    _patch_roots(monkeypatch, tmp_path)
    pm.create_project("A")
    mock_session_state.update({
        "epochs": 999,
        "lr": 0.05,
        "model_arch": "FlashDet-X",
        "_config_mirror": {"epochs": 999, "lr": 0.05},
    })
    pm._clear_session_for_new_project()
    assert "epochs" not in mock_session_state
    assert "lr" not in mock_session_state
    assert "model_arch" not in mock_session_state
    assert "_config_mirror" not in mock_session_state


def test_load_seeds_project_runs_dir(mock_session_state, monkeypatch, tmp_path):
    _patch_roots(monkeypatch, tmp_path)
    project = pm.create_project("B")
    pm.load_project_state(project["id"])
    assert mock_session_state["save_dir"] == pm.get_project_runs_dir(project["id"])
    assert os.path.isdir(mock_session_state["save_dir"])


def test_switch_project_rebuilds_mirror(mock_session_state, monkeypatch, tmp_path):
    _patch_roots(monkeypatch, tmp_path)
    a = pm.create_project("A")
    mock_session_state["epochs"] = 11
    pm.save_project_state()

    b = pm.create_project("B")
    mock_session_state["epochs"] = 22
    pm.save_project_state()

    pm.switch_project(a["id"])
    from flashstudio.utils import get_state
    assert get_state("epochs") == 11
    assert mock_session_state["_config_mirror"]["epochs"] == 11

    pm.switch_project(b["id"])
    assert get_state("epochs") == 22
