"""Tests for training monitor run_meta."""

import os

from flashstudio.pages.training.monitor.run_meta import _get_run_meta


def test_empty_dir_status(tmp_dir):
    meta = _get_run_meta(tmp_dir)
    assert meta["status"] == "Empty"
    assert meta["epochs"] == "?"


def test_dir_with_final_inference_complete(tmp_dir):
    open(os.path.join(tmp_dir, "model_final_inference.pth"), "w").close()
    meta = _get_run_meta(tmp_dir)
    assert meta["status"] == "Complete"


def test_dir_with_csv_has_epochs_and_map(tmp_dir):
    csv_path = os.path.join(tmp_dir, "training_log.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,train_loss,lr,val_loss,mAP@0.5\n")
        f.write("1,5.0,0.001,4.5,0.1\n")
        f.write("2,4.0,0.001,3.5,0.2\n")
        f.write("3,3.0,0.001,2.5,0.35\n")

    meta = _get_run_meta(tmp_dir)
    assert meta["epochs"] == 3
    assert meta["mAP"] is not None
    assert meta["mAP"] == 0.35


def test_dir_with_log_parses_model_info(tmp_dir):
    log_path = os.path.join(tmp_dir, "train_20250101_120000.log")
    with open(log_path, "w") as f:
        f.write("FlashDet Training\n")
        f.write("Model Size: FlashDet-N\n")
        f.write("Device: cuda\n")
        f.write("Epochs: 50\n")

    meta = _get_run_meta(tmp_dir)
    assert meta["model"] == "FlashDet-N"
    assert meta["epochs"] == "50"
