"""Shared fixtures for FlashStudio tests."""

import os
import sys
import tempfile
import shutil
from unittest.mock import MagicMock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture()
def tmp_dir():
    """Provide a temporary directory, cleaned up after the test."""
    d = tempfile.mkdtemp(prefix="flashstudio_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def mock_session_state(monkeypatch):
    """Replace streamlit.session_state with a plain dict for unit tests."""
    state = {}

    class _FakeSessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError:
                raise AttributeError(name)

    fake = _FakeSessionState(state)

    st_mock = MagicMock()
    st_mock.session_state = fake
    monkeypatch.setattr("streamlit.session_state", fake)
    return fake


@pytest.fixture()
def sample_training_csv(tmp_dir):
    """Create a sample training_log.csv for parser tests."""
    csv_path = os.path.join(tmp_dir, "training_log.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,lr,val_loss,mAP@0.5\n")
        f.write("1,5.432100,0.00100000,,\n")
        f.write("2,4.321000,0.00098000,,\n")
        f.write("3,3.210000,0.00096000,3.500000,0.120000\n")
        f.write("4,2.800000,0.00094000,,\n")
        f.write("5,2.500000,0.00092000,2.900000,0.250000\n")
    return csv_path


@pytest.fixture()
def sample_training_log(tmp_dir):
    """Create a sample train_*.log for parser tests."""
    import time
    log_path = os.path.join(tmp_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    with open(log_path, "w") as f:
        f.write("FlashDet Training\n")
        f.write("Model: FlashDetN, Input: (320, 320)\n")
        f.write("Device: cuda\n")
        f.write("Epochs: 100, Batch: 16, LR: 0.001\n")
        f.write("Classes (3): ['cat', 'dog', 'bird']\n")
        f.write("\n")
        f.write("Epoch 1/100 (lr=0.001000)\n")
        f.write("  Val Loss: 5.43 (loss_total: 5.43, o2m_cls: 2.1, o2m_box: 1.5, o2o_cls: 1.0, o2o_box: 0.83) | mAP@0.5: 0.012\n")
        f.write("Epoch 2/100 (lr=0.000980)\n")
        f.write("  Val Loss: 4.32 (loss_total: 4.32, o2m_cls: 1.8, o2m_box: 1.2, o2o_cls: 0.8, o2o_box: 0.52) | mAP@0.5: 0.045\n")
    return log_path
