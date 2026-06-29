"""Tests for flashstudio.constants — verify all constants have expected types."""

import os
import pytest
from flashstudio.constants import (
    IMG_EXTENSIONS, VIDEO_EXTENSIONS, WEIGHT_EXTENSIONS, ARCHIVE_EXTENSIONS,
    CKPT_BEST, CKPT_LAST, CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16,
    CKPT_BEST_INFERENCE, CKPT_BEST_FP16, CKPT_LAST_INFERENCE,
    COMPLETE_MARKERS, BEST_WEIGHT_PRIORITY, TRAINING_LOG_CSV,
    PROJECT_ROOT, DEFAULT_SAVE_DIR, DEFAULT_DATA_DIR,
    COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR,
    BBOX_COLORS_RGB, BBOX_COLORS_HEX,
    INFER_CONF_THRESHOLD, INFER_NMS_THRESHOLD, INFER_IMG_SIZE,
    INFER_NUM_CLASSES, INFER_MAX_FRAMES,
    EXPORT_IMG_SIZES, EXPORT_OPSET_MIN, EXPORT_OPSET_MAX, EXPORT_OPSET_DEFAULT,
    EXPORT_FORMATS, EXPORT_WEIGHT_MAP,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_IMG_SIZE,
    BATCH_SIZE_OPTIONS, IMG_SIZE_OPTIONS,
    FLASHDET_MODELS, ARCH_FAMILIES, OPTIMIZERS,
    COCO_CLASSES, SIZE_GB, SIZE_MB, SIZE_KB,
    format_bytes,
)


class TestFileExtensions:
    def test_img_extensions_are_tuples(self):
        assert isinstance(IMG_EXTENSIONS, tuple)
        assert all(ext.startswith(".") for ext in IMG_EXTENSIONS)

    def test_video_extensions_are_tuples(self):
        assert isinstance(VIDEO_EXTENSIONS, tuple)
        assert ".mp4" in VIDEO_EXTENSIONS

    def test_weight_extensions(self):
        assert ".pth" in WEIGHT_EXTENSIONS
        assert ".onnx" in WEIGHT_EXTENSIONS


class TestCheckpointNames:
    def test_checkpoint_names_are_strings(self):
        for name in [CKPT_BEST, CKPT_LAST, CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16]:
            assert isinstance(name, str)
            assert name.endswith(".pth")

    def test_complete_markers_nonempty(self):
        assert len(COMPLETE_MARKERS) > 0

    def test_best_weight_priority_nonempty(self):
        assert len(BEST_WEIGHT_PRIORITY) > 0
        assert CKPT_BEST_INFERENCE in BEST_WEIGHT_PRIORITY


class TestPaths:
    def test_project_root_is_absolute(self):
        assert os.path.isabs(PROJECT_ROOT)

    def test_default_save_dir_is_absolute(self):
        assert os.path.isabs(DEFAULT_SAVE_DIR)

    def test_default_data_dir_is_absolute(self):
        assert os.path.isabs(DEFAULT_DATA_DIR)


class TestColors:
    def test_hex_colors_start_with_hash(self):
        for color in [COLOR_PRIMARY, COLOR_SUCCESS, COLOR_WARNING, COLOR_ERROR]:
            assert color.startswith("#")
            assert len(color) == 7

    def test_bbox_rgb_valid(self):
        assert len(BBOX_COLORS_RGB) >= 1
        for r, g, b in BBOX_COLORS_RGB:
            assert 0 <= r <= 255
            assert 0 <= g <= 255
            assert 0 <= b <= 255

    def test_bbox_hex_valid(self):
        assert len(BBOX_COLORS_HEX) >= 1
        for color in BBOX_COLORS_HEX:
            assert color.startswith("#")


class TestInferenceDefaults:
    def test_confidence_in_range(self):
        assert 0.0 < INFER_CONF_THRESHOLD < 1.0

    def test_nms_in_range(self):
        assert 0.0 < INFER_NMS_THRESHOLD < 1.0

    def test_img_size_positive(self):
        assert INFER_IMG_SIZE > 0

    def test_num_classes_positive(self):
        assert INFER_NUM_CLASSES > 0


class TestExportDefaults:
    def test_img_sizes_sorted(self):
        assert EXPORT_IMG_SIZES == sorted(EXPORT_IMG_SIZES)

    def test_opset_range_valid(self):
        assert EXPORT_OPSET_MIN < EXPORT_OPSET_MAX
        assert EXPORT_OPSET_MIN <= EXPORT_OPSET_DEFAULT <= EXPORT_OPSET_MAX

    def test_formats_nonempty(self):
        assert "ONNX" in EXPORT_FORMATS

    def test_weight_map_keys(self):
        assert "Best (inference)" in EXPORT_WEIGHT_MAP
        assert "Last" in EXPORT_WEIGHT_MAP


class TestTrainingDefaults:
    def test_epochs_positive(self):
        assert TRAIN_EPOCHS > 0

    def test_batch_size_positive(self):
        assert TRAIN_BATCH_SIZE > 0

    def test_lr_positive(self):
        assert TRAIN_LR > 0

    def test_img_size_in_options(self):
        assert TRAIN_IMG_SIZE in IMG_SIZE_OPTIONS

    def test_batch_size_in_options(self):
        assert TRAIN_BATCH_SIZE in BATCH_SIZE_OPTIONS


class TestModelArchitecture:
    def test_flashdet_models_has_all_sizes(self):
        expected = ["FlashDet-Pico", "FlashDet-Nano", "FlashDet-Small",
                     "FlashDet-Medium", "FlashDet-Large", "FlashDet-X"]
        for name in expected:
            assert name in FLASHDET_MODELS

    def test_flashdet_model_keys(self):
        for name, info in FLASHDET_MODELS.items():
            assert "size" in info
            assert "params" in info
            assert "speed" in info

    def test_arch_families_nonempty(self):
        assert len(ARCH_FAMILIES) >= 2
        assert any("FlashDet" in f for f in ARCH_FAMILIES)

    def test_optimizers_nonempty(self):
        assert "AdamW" in OPTIMIZERS


class TestCocoClasses:
    def test_coco_80_classes(self):
        assert len(COCO_CLASSES) == 80

    def test_common_classes_present(self):
        for cls in ["person", "car", "dog", "cat"]:
            assert cls in COCO_CLASSES


class TestFormatBytes:
    def test_gb(self):
        assert "GB" in format_bytes(2_000_000_000)

    def test_mb(self):
        assert "MB" in format_bytes(50_000_000)

    def test_kb(self):
        assert "KB" in format_bytes(500)

    def test_zero(self):
        result = format_bytes(0)
        assert "KB" in result


class TestSizeConstants:
    def test_size_hierarchy(self):
        assert SIZE_KB < SIZE_MB < SIZE_GB
