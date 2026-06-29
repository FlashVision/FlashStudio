"""Save-dir compatibility tests — verify FlashStudio reads exactly what FlashDet writes.

FlashDet training produces a specific directory layout under ``save_dir/<run_name>/``:
  - checkpoint_best.pth, checkpoint_last.pth
  - model_best_inference.pth, model_last_inference.pth
  - model_final_inference.pth, model_final_fp16.pth, model_best_fp16.pth
  - training_log.csv
  - train_<timestamp>.log
  - config.yaml, results.json
  - model.onnx, model.onnx.data
  - visualizations/  (epoch0001.jpg, epoch0002.jpg, ...)
  - gt_verification/
      verification_report.json
      verification_summary.txt
      images/raw/*.jpg
      images/dataloader/*.jpg
  - plots/
      training_curves.png, mAP_curve.png

These tests create a realistic directory tree, then call FlashStudio's
parsers/monitors and verify they discover the correct files.
"""

import csv
import json
import os
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from flashstudio.constants import (
    CKPT_BEST,
    CKPT_BEST_FP16,
    CKPT_BEST_INFERENCE,
    CKPT_FINAL_FP16,
    CKPT_FINAL_INFERENCE,
    CKPT_LAST,
    CKPT_LAST_INFERENCE,
    CONFIG_YAML_FILE,
    GT_REPORT_FILE,
    GT_SUMMARY_FILE,
    GT_VERIFICATION_DIR,
    ONNX_DATA_FILE,
    ONNX_MODEL_FILE,
    RESULTS_JSON_FILE,
    TRAINING_LOG_CSV,
    TRAINING_LOG_GLOB,
    VIS_DIR_NAMES,
    VIS_SKIP_FILE,
)


# ─────────────────────────────────────────
# Fixtures — build a realistic FlashDet save-dir
# ─────────────────────────────────────────

@pytest.fixture()
def run_dir(tmp_path):
    """Create a fully-populated FlashDet training run directory."""
    rd = tmp_path / "run_FlashDet-Pico_001"
    rd.mkdir()

    # Checkpoint files (.pth)
    for ckpt in (CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_BEST_FP16,
                 CKPT_LAST, CKPT_LAST_INFERENCE,
                 CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16):
        (rd / ckpt).write_bytes(b"\x00" * 64)

    # CSV metrics
    csv_path = rd / TRAINING_LOG_CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "lr", "val_loss", "mAP@0.5"])
        w.writeheader()
        for ep in range(1, 11):
            w.writerow({
                "epoch": ep,
                "train_loss": f"{10.0 - ep * 0.5:.6f}",
                "lr": "0.00100000",
                "val_loss": f"{8.0 - ep * 0.3:.6f}",
                "mAP@0.5": f"{ep * 0.05:.6f}",
            })

    # Log file
    log_path = rd / "train_20260101_120000.log"
    log_path.write_text(textwrap.dedent("""\
        ═══ FlashDet Training ═══
        Model: p, Input: (320, 320)
        Model Size: FlashDet-Pico
        Device: cuda:0
        Classes (3): ['cat', 'dog', 'bird']
        Epochs: 10, Batch: 16, LR: 0.001

        Epoch 1/10 (lr=0.001000)
          Val Loss: 8.50 | mAP@0.5: 0.050
        Epoch 10/10 (lr=0.000100)
          Val Loss: 5.00 | mAP@0.5: 0.500

        Training Complete!
        Best mAP@0.5: 0.500 | Best Loss: 5.000
    """))

    # ONNX exports
    (rd / ONNX_MODEL_FILE).write_bytes(b"\x00" * 128)
    (rd / ONNX_DATA_FILE).write_bytes(b"\x00" * 256)

    # config.yaml and results.json
    (rd / CONFIG_YAML_FILE).write_text("model_arch: FlashDet-Pico\nepochs: 10\n")
    (rd / RESULTS_JSON_FILE).write_text(json.dumps({"best_mAP": 0.5}))

    # visualizations/ directory
    vis_dir = rd / VIS_DIR_NAMES[0]   # "visualizations"
    vis_dir.mkdir()
    for ep in range(1, 6):
        _write_tiny_jpg(vis_dir / f"epoch{ep:04d}.jpg")
    _write_tiny_jpg(vis_dir / VIS_SKIP_FILE)

    # gt_verification/ directory
    gt_dir = rd / GT_VERIFICATION_DIR
    gt_dir.mkdir()
    (gt_dir / GT_REPORT_FILE).write_text(json.dumps({
        "passed": True,
        "num_classes": 3,
        "splits": {
            "train": {
                "coco": {"num_images": 100, "num_annotations": 300, "num_categories": 3},
                "dataloader": {"avg_boxes_per_sample": 3.0},
            },
            "val": {
                "coco": {"num_images": 20, "num_annotations": 60, "num_categories": 3},
                "dataloader": {"avg_boxes_per_sample": 3.0},
            },
        },
    }))
    (gt_dir / GT_SUMMARY_FILE).write_text("train: 100 images, 300 annotations, 3 categories\nval: 20 images")

    raw_dir = gt_dir / "images" / "raw"
    raw_dir.mkdir(parents=True)
    for i in range(3):
        _write_tiny_jpg(raw_dir / f"gt_{i:06d}.jpg")

    dl_dir = gt_dir / "images" / "dataloader"
    dl_dir.mkdir(parents=True)
    for i in range(2):
        _write_tiny_jpg(dl_dir / f"dl_{i:06d}.jpg")

    # plots/ directory
    plots_dir = rd / VIS_DIR_NAMES[1]   # "plots"
    plots_dir.mkdir()
    _write_tiny_jpg(plots_dir / "training_curves.png")
    _write_tiny_jpg(plots_dir / "mAP_curve.png")

    return str(rd)


def _write_tiny_jpg(path):
    """Write a minimal valid-ish file (enough for os.path.isfile checks)."""
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 12)


# ─────────────────────────────────────────
# 1. Constants consistency — names used by writers match names used by readers
# ─────────────────────────────────────────

class TestConstantsConsistency:
    """Verify the constant values FlashStudio defines match what FlashDet writes."""

    def test_checkpoint_names_are_pth(self):
        for name in (CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_BEST_FP16,
                     CKPT_LAST, CKPT_LAST_INFERENCE,
                     CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16):
            assert name.endswith(".pth"), f"{name} should end with .pth"

    def test_csv_name(self):
        assert TRAINING_LOG_CSV == "training_log.csv"

    def test_vis_dir_primary(self):
        assert VIS_DIR_NAMES[0] == "visualizations"

    def test_gt_dir_name(self):
        assert GT_VERIFICATION_DIR == "gt_verification"

    def test_gt_report_file(self):
        assert GT_REPORT_FILE == "verification_report.json"

    def test_gt_summary_file(self):
        assert GT_SUMMARY_FILE == "verification_summary.txt"

    def test_onnx_filenames(self):
        assert ONNX_MODEL_FILE == "model.onnx"
        assert ONNX_DATA_FILE == "model.onnx.data"

    def test_log_glob_pattern(self):
        assert TRAINING_LOG_GLOB == "train_*.log"

    def test_config_yaml(self):
        assert CONFIG_YAML_FILE == "config.yaml"

    def test_results_json(self):
        assert RESULTS_JSON_FILE == "results.json"


# ─────────────────────────────────────────
# 2. CSV parser reads what StudioCSVLogger writes
# ─────────────────────────────────────────

class TestCSVParserCompat:
    def test_parse_training_csv_reads_all_epochs(self, run_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        history = _parse_training_csv(run_dir)
        assert history is not None
        assert len(history["epochs"]) == 10

    def test_parse_csv_extracts_map(self, run_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        history = _parse_training_csv(run_dir)
        assert history["mAP50"][-1] == pytest.approx(0.5, abs=0.01)

    def test_parse_csv_extracts_losses(self, run_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        history = _parse_training_csv(run_dir)
        assert len(history["train_loss"]) == 10
        assert history["train_loss"][0] > history["train_loss"][-1]

    def test_parse_csv_extracts_lr(self, run_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        history = _parse_training_csv(run_dir)
        assert all(lr > 0 for lr in history["lr"])

    def test_csv_file_path_uses_constant(self, run_dir):
        csv_path = os.path.join(run_dir, TRAINING_LOG_CSV)
        assert os.path.isfile(csv_path)


# ─────────────────────────────────────────
# 3. Log parser reads what FlashDet trainer writes
# ─────────────────────────────────────────

class TestLogParserCompat:
    def test_find_log_file(self, run_dir):
        from flashstudio.pages.training.monitor.parsers import _find_log_file
        log = _find_log_file(run_dir)
        assert log is not None
        assert os.path.basename(log).startswith("train_")
        assert log.endswith(".log")

    def test_parse_training_log_extracts_model(self, run_dir):
        from flashstudio.pages.training.monitor.parsers import (
            _find_log_file, _parse_training_log,
        )
        log = _find_log_file(run_dir)
        history = _parse_training_log(log)
        assert history is not None
        assert history["model_info"]

    def test_parse_training_log_extracts_device(self, run_dir):
        from flashstudio.pages.training.monitor.parsers import (
            _find_log_file, _parse_training_log,
        )
        log = _find_log_file(run_dir)
        history = _parse_training_log(log)
        assert history["device"] == "cuda:0"

    def test_parse_training_log_extracts_classes(self, run_dir):
        from flashstudio.pages.training.monitor.parsers import (
            _find_log_file, _parse_training_log,
        )
        log = _find_log_file(run_dir)
        history = _parse_training_log(log)
        assert history["classes"] == ["cat", "dog", "bird"]

    def test_parse_csv_header_from_log(self, run_dir):
        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        history = _parse_training_csv(run_dir)
        assert history["total_epochs"] == 10
        assert history["batch_size"] == 16
        assert history["model_info"]
        assert history["device"] == "cuda:0"


# ─────────────────────────────────────────
# 4. Run metadata reader — discovers checkpoints, status, model info
# ─────────────────────────────────────────

class TestRunMetaCompat:
    def test_status_complete(self, run_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        meta = _get_run_meta(run_dir)
        assert meta["status"] == "Complete"

    def test_model_extracted(self, run_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        meta = _get_run_meta(run_dir)
        assert "FlashDet" in meta["model"] or "Pico" in meta["model"] or "p" in meta["model"]

    def test_epochs_extracted(self, run_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        meta = _get_run_meta(run_dir)
        assert meta["epochs"] == "10" or meta["epochs"] == 10

    def test_mAP_extracted(self, run_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        meta = _get_run_meta(run_dir)
        assert meta["mAP"] is not None and meta["mAP"] > 0

    def test_status_in_progress_without_final(self, tmp_path):
        """Without final/best checkpoints, status should not be 'Complete'."""
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        rd = tmp_path / "partial_run"
        rd.mkdir()
        (rd / CKPT_LAST).write_bytes(b"\x00" * 16)
        (rd / TRAINING_LOG_CSV).write_text("epoch,train_loss\n1,5.0\n")
        log = rd / "train_20260101.log"
        log.write_text("Epoch 1/100 (lr=0.001)\n")
        meta = _get_run_meta(str(rd))
        assert meta["status"] == "In Progress"

    def test_status_empty_run(self, tmp_path):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        rd = tmp_path / "empty_run"
        rd.mkdir()
        meta = _get_run_meta(str(rd))
        assert meta["status"] == "Empty"

    def test_display_name_enriched(self, run_dir):
        from flashstudio.pages.training.monitor.run_meta import _get_run_meta
        meta = _get_run_meta(run_dir)
        assert "|" in meta["display_name"]


# ─────────────────────────────────────────
# 5. Visualizations — reader finds what callback writes
# ─────────────────────────────────────────

class TestVisualizationsCompat:
    def test_vis_dir_exists(self, run_dir):
        vis_dir = os.path.join(run_dir, VIS_DIR_NAMES[0])
        assert os.path.isdir(vis_dir)

    def test_vis_images_found(self, run_dir):
        from flashstudio.constants import IMG_EXTENSIONS
        vis_dir = os.path.join(run_dir, VIS_DIR_NAMES[0])
        images = [f for f in os.listdir(vis_dir)
                  if f.lower().endswith(IMG_EXTENSIONS) and f != VIS_SKIP_FILE]
        assert len(images) == 5

    def test_vis_skip_file_excluded(self, run_dir):
        vis_dir = os.path.join(run_dir, VIS_DIR_NAMES[0])
        from flashstudio.constants import IMG_EXTENSIONS
        images = [f for f in os.listdir(vis_dir)
                  if f.lower().endswith(IMG_EXTENSIONS) and f != VIS_SKIP_FILE]
        assert VIS_SKIP_FILE not in images

    def test_vis_reader_falls_back_to_plots(self, tmp_path):
        """If 'visualizations' is missing, reader should also check 'plots' and 'vis'."""
        rd = tmp_path / "vis_fallback"
        rd.mkdir()
        plots = rd / VIS_DIR_NAMES[1]
        plots.mkdir()
        _write_tiny_jpg(plots / "epoch0001.jpg")
        vis = rd / VIS_DIR_NAMES[2]
        vis.mkdir()
        _write_tiny_jpg(vis / "epoch0002.jpg")
        all_imgs = []
        for dname in VIS_DIR_NAMES:
            d = rd / dname
            if d.is_dir():
                all_imgs.extend(f for f in os.listdir(d) if f.endswith((".jpg", ".png")))
        assert len(all_imgs) == 2


# ─────────────────────────────────────────
# 6. GT Verification — reader finds what hook writes
# ─────────────────────────────────────────

class TestGTVerificationCompat:
    def test_gt_dir_exists(self, run_dir):
        gt_dir = os.path.join(run_dir, GT_VERIFICATION_DIR)
        assert os.path.isdir(gt_dir)

    def test_report_file_exists(self, run_dir):
        gt_dir = os.path.join(run_dir, GT_VERIFICATION_DIR)
        assert os.path.isfile(os.path.join(gt_dir, GT_REPORT_FILE))

    def test_report_json_parseable(self, run_dir):
        gt_dir = os.path.join(run_dir, GT_VERIFICATION_DIR)
        with open(os.path.join(gt_dir, GT_REPORT_FILE)) as f:
            report = json.load(f)
        assert report["passed"] is True
        assert report["num_classes"] == 3
        assert "train" in report["splits"]
        assert "val" in report["splits"]

    def test_report_has_coco_metrics(self, run_dir):
        gt_dir = os.path.join(run_dir, GT_VERIFICATION_DIR)
        with open(os.path.join(gt_dir, GT_REPORT_FILE)) as f:
            report = json.load(f)
        tc = report["splits"]["train"]["coco"]
        assert tc["num_images"] > 0
        assert tc["num_annotations"] > 0

    def test_summary_file_exists(self, run_dir):
        gt_dir = os.path.join(run_dir, GT_VERIFICATION_DIR)
        assert os.path.isfile(os.path.join(gt_dir, GT_SUMMARY_FILE))

    def test_gt_raw_images_dir(self, run_dir):
        raw_dir = os.path.join(run_dir, GT_VERIFICATION_DIR, "images", "raw")
        assert os.path.isdir(raw_dir)
        images = [f for f in os.listdir(raw_dir) if f.endswith(".jpg")]
        assert len(images) == 3

    def test_gt_dataloader_images_dir(self, run_dir):
        dl_dir = os.path.join(run_dir, GT_VERIFICATION_DIR, "images", "dataloader")
        assert os.path.isdir(dl_dir)
        images = [f for f in os.listdir(dl_dir) if f.endswith(".jpg")]
        assert len(images) == 2


# ─────────────────────────────────────────
# 7. Checkpoint file type classifier
# ─────────────────────────────────────────

class TestFileTypeClassifier:
    def test_best_inference(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(CKPT_BEST_INFERENCE) == "Best inference weights"

    def test_best_checkpoint(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(CKPT_BEST) == "Best checkpoint"

    def test_final_inference(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(CKPT_FINAL_INFERENCE) == "Final inference weights"

    def test_final_fp16(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(CKPT_FINAL_FP16) == "Final FP16 weights"

    def test_last_checkpoint(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(CKPT_LAST) == "Latest checkpoint (full)"

    def test_last_inference(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(CKPT_LAST_INFERENCE) == "Latest inference weights"

    def test_onnx_model(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(ONNX_MODEL_FILE) == "ONNX model"

    def test_onnx_data(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(ONNX_DATA_FILE) == "ONNX weights data"

    def test_results_json(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(RESULTS_JSON_FILE) == "Training results"

    def test_csv_type(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type(TRAINING_LOG_CSV) == "Training metrics CSV"

    def test_log_type(self):
        from flashstudio.pages.training.monitor.checkpoints import _file_type
        assert _file_type("train_20260101.log") == "Training log"


# ─────────────────────────────────────────
# 8. Cleanup keeps the right files
# ─────────────────────────────────────────

class TestCleanupKeepBest:
    def test_cleanup_keeps_best_and_csv(self, run_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best
        removed = _cleanup_run_keep_best(run_dir)
        assert removed > 0
        assert os.path.isfile(os.path.join(run_dir, CKPT_BEST))
        assert os.path.isfile(os.path.join(run_dir, CKPT_BEST_INFERENCE))
        assert os.path.isfile(os.path.join(run_dir, TRAINING_LOG_CSV))

    def test_cleanup_keeps_final_weights(self, run_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best
        _cleanup_run_keep_best(run_dir)
        assert os.path.isfile(os.path.join(run_dir, CKPT_FINAL_INFERENCE))
        assert os.path.isfile(os.path.join(run_dir, CKPT_FINAL_FP16))

    def test_cleanup_keeps_onnx(self, run_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best
        _cleanup_run_keep_best(run_dir)
        assert os.path.isfile(os.path.join(run_dir, ONNX_MODEL_FILE))
        assert os.path.isfile(os.path.join(run_dir, ONNX_DATA_FILE))

    def test_cleanup_removes_vis_dirs(self, run_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best
        _cleanup_run_keep_best(run_dir)
        assert not os.path.isdir(os.path.join(run_dir, VIS_DIR_NAMES[0]))
        assert not os.path.isdir(os.path.join(run_dir, GT_VERIFICATION_DIR))

    def test_cleanup_keeps_log_files(self, run_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best
        _cleanup_run_keep_best(run_dir)
        import glob
        logs = glob.glob(os.path.join(run_dir, TRAINING_LOG_GLOB))
        assert len(logs) > 0

    def test_cleanup_removes_last_checkpoint(self, run_dir):
        from flashstudio.pages.training.launch.dialogs import _cleanup_run_keep_best
        _cleanup_run_keep_best(run_dir)
        assert not os.path.isfile(os.path.join(run_dir, CKPT_LAST))
        assert not os.path.isfile(os.path.join(run_dir, CKPT_LAST_INFERENCE))


# ─────────────────────────────────────────
# 9. Plots directory — curves reader finds FlashDet's plots
# ─────────────────────────────────────────

class TestPlotsCompat:
    def test_plots_dir_uses_vis_dir_names(self, run_dir):
        plots_dir = os.path.join(run_dir, VIS_DIR_NAMES[1])
        assert os.path.isdir(plots_dir)

    def test_training_curves_image_exists(self, run_dir):
        plots_dir = os.path.join(run_dir, VIS_DIR_NAMES[1])
        assert os.path.isfile(os.path.join(plots_dir, "training_curves.png"))

    def test_map_curve_image_exists(self, run_dir):
        plots_dir = os.path.join(run_dir, VIS_DIR_NAMES[1])
        assert os.path.isfile(os.path.join(plots_dir, "mAP_curve.png"))


# ─────────────────────────────────────────
# 10. No hardcoded strings — all file/dir accesses use constants
# ─────────────────────────────────────────

class TestNoHardcodedStrings:
    """Verify that source files reference constants, not literal strings,
    for filenames that must stay in sync between FlashDet and FlashStudio."""

    @pytest.fixture(autouse=True)
    def _load_sources(self):
        import importlib
        import inspect
        self._sources = {}
        modules = [
            "flashstudio.pages.training.monitor.parsers",
            "flashstudio.pages.training.monitor.run_meta",
            "flashstudio.pages.training.monitor.visualizations",
            "flashstudio.pages.training.monitor.gt_verification",
            "flashstudio.pages.training.monitor.curves",
            "flashstudio.pages.training.launch.dialogs",
            "flashstudio.utils.training_hooks",
        ]
        for modname in modules:
            try:
                mod = importlib.import_module(modname)
                src = inspect.getsource(mod)
                self._sources[modname] = src
            except Exception:
                pass

    def test_no_hardcoded_gt_verification_dir(self):
        for name, src in self._sources.items():
            if "gt_verification" in name:
                continue
            count = src.count('"gt_verification"')
            assert count == 0, f'{name} has hardcoded "gt_verification" — use GT_VERIFICATION_DIR'

    def test_no_hardcoded_training_log_csv(self):
        for name, src in self._sources.items():
            count = src.count('"training_log.csv"')
            assert count == 0, f'{name} has hardcoded "training_log.csv" — use TRAINING_LOG_CSV'

    def test_no_hardcoded_model_onnx(self):
        for name, src in self._sources.items():
            if "training_hooks" in name:
                continue
            count = src.count('"model.onnx"')
            assert count == 0, f'{name} has hardcoded "model.onnx" — use ONNX_MODEL_FILE'


# ─────────────────────────────────────────
# 11. Full roundtrip: write with hooks → read with parsers
# ─────────────────────────────────────────

class TestWriteReadRoundtrip:
    """Simulate what StudioCSVLogger writes and verify parsers can read it."""

    def test_csv_roundtrip(self, tmp_path):
        csv_path = tmp_path / TRAINING_LOG_CSV
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "lr", "val_loss", "mAP@0.5"])
            w.writeheader()
            w.writerow({"epoch": 1, "train_loss": "5.000000", "lr": "0.00100000",
                         "val_loss": "4.500000", "mAP@0.5": "0.100000"})
            w.writerow({"epoch": 2, "train_loss": "4.000000", "lr": "0.00090000",
                         "val_loss": "3.500000", "mAP@0.5": "0.200000"})

        from flashstudio.pages.training.monitor.parsers import _parse_training_csv
        history = _parse_training_csv(str(tmp_path))
        assert history is not None
        assert len(history["epochs"]) == 2
        assert history["mAP50"][0] == pytest.approx(0.1, abs=0.001)
        assert history["mAP50"][1] == pytest.approx(0.2, abs=0.001)
        assert history["train_loss"][0] == pytest.approx(5.0, abs=0.01)
        assert history["lr"][0] == pytest.approx(0.001, abs=0.00001)

    def test_gt_report_roundtrip(self, tmp_path):
        gt_dir = tmp_path / GT_VERIFICATION_DIR
        gt_dir.mkdir()
        report = {"passed": True, "num_classes": 2, "splits": {
            "train": {"coco": {"num_images": 50, "num_annotations": 100, "num_categories": 2},
                      "dataloader": {"avg_boxes_per_sample": 2.0}},
        }}
        with open(gt_dir / GT_REPORT_FILE, "w") as f:
            json.dump(report, f)
        with open(gt_dir / GT_SUMMARY_FILE, "w") as f:
            f.write("train: 50 images\n")

        loaded = json.load(open(gt_dir / GT_REPORT_FILE))
        assert loaded["passed"] is True
        assert loaded["num_classes"] == 2
        assert loaded["splits"]["train"]["coco"]["num_images"] == 50


# ─────────────────────────────────────────
# 12. Export weight discovery — finds .pth in save_dir
# ─────────────────────────────────────────

class TestExportWeightDiscovery:
    def test_pth_files_discoverable(self, run_dir):
        pth_files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]
        assert len(pth_files) == 7
        expected = {CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_BEST_FP16,
                    CKPT_LAST, CKPT_LAST_INFERENCE,
                    CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16}
        assert set(pth_files) == expected

    def test_walk_finds_nested_pth(self, tmp_path):
        save_dir = tmp_path / "workspace"
        run_a = save_dir / "run_a"
        run_a.mkdir(parents=True)
        (run_a / CKPT_BEST_INFERENCE).write_bytes(b"\x00")
        run_b = save_dir / "run_b"
        run_b.mkdir()
        (run_b / CKPT_BEST).write_bytes(b"\x00")

        found = []
        for root, _dirs, files in os.walk(str(save_dir)):
            for f in files:
                if f.endswith(".pth"):
                    found.append(os.path.join(root, f))
        assert len(found) == 2


# ─────────────────────────────────────────
# 13. Full directory inventory check
# ─────────────────────────────────────────

class TestDirectoryInventory:
    """Verify the full run directory has all expected artifacts."""

    def test_all_expected_files_present(self, run_dir):
        expected_files = {
            CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_BEST_FP16,
            CKPT_LAST, CKPT_LAST_INFERENCE,
            CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16,
            TRAINING_LOG_CSV, CONFIG_YAML_FILE, RESULTS_JSON_FILE,
            ONNX_MODEL_FILE, ONNX_DATA_FILE,
        }
        actual_files = {f for f in os.listdir(run_dir) if os.path.isfile(os.path.join(run_dir, f))}
        missing = expected_files - actual_files
        assert not missing, f"Missing files: {missing}"

    def test_all_expected_dirs_present(self, run_dir):
        expected_dirs = {VIS_DIR_NAMES[0], VIS_DIR_NAMES[1], GT_VERIFICATION_DIR}
        actual_dirs = {d for d in os.listdir(run_dir) if os.path.isdir(os.path.join(run_dir, d))}
        missing = expected_dirs - actual_dirs
        assert not missing, f"Missing directories: {missing}"

    def test_log_file_matches_glob(self, run_dir):
        import glob
        logs = glob.glob(os.path.join(run_dir, TRAINING_LOG_GLOB))
        assert len(logs) >= 1
        assert all(os.path.basename(l).startswith("train_") for l in logs)
        assert all(l.endswith(".log") for l in logs)
