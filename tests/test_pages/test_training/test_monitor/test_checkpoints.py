"""Tests for training monitor checkpoints file type classification."""

from flashstudio.pages.training.monitor.checkpoints import _file_type


def test_best_inference():
    result = _file_type("best_inference.pth")
    assert "Best" in result
    assert "inference" in result.lower()


def test_best_checkpoint():
    result = _file_type("checkpoint_best.pth")
    assert "Best" in result


def test_final_fp16():
    result = _file_type("final_fp16.pth")
    assert "FP16" in result


def test_final_inference():
    result = _file_type("model_final_inference.pth")
    assert "Final" in result


def test_checkpoint_last():
    result = _file_type("checkpoint_last.pth")
    assert "Latest" in result


def test_model_onnx():
    result = _file_type("model.onnx")
    assert "ONNX" in result


def test_training_log_csv():
    result = _file_type("training_log.csv")
    assert "CSV" in result


def test_training_log_file():
    result = _file_type("train_20250101.log")
    assert "log" in result.lower()


def test_results_json():
    result = _file_type("results.json")
    assert "result" in result.lower() or "Report" in result


def test_summary_txt():
    result = _file_type("summary.txt")
    assert "Summary" in result


def test_unknown_file():
    result = _file_type("unknown_file.xyz")
    assert "Other" in result
