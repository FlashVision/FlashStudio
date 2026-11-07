"""Training process lifecycle: completed children are reaped, not zombies."""

import os
import subprocess
import time

import pytest

from flashstudio.constants import CKPT_FINAL_INFERENCE


def test_completed_process_marks_run_complete(mock_session_state, tmp_path):
    from flashstudio.pages.training.page import _check_training_process

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / CKPT_FINAL_INFERENCE).write_bytes(b"\x00")
    log = run_dir / "train_1.log"
    log.write_text("Training Complete!\n")

    proc = subprocess.Popen(["true"])
    try:
        proc.wait(timeout=5)
    except Exception:
        pytest.skip("could not start a helper process")

    mock_session_state.update({
        "training_active": True,
        "training_pid": proc.pid,
        "_training_process": proc,
        "active_run_path": str(run_dir),
        "training_log_file": str(log),
    })
    _check_training_process()
    assert mock_session_state["training_status"] == "Complete"
    assert mock_session_state["_training_process"] is None
    assert mock_session_state["training_active"] is False
