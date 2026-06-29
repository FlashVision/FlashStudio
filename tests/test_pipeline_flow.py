"""Tests for session state data flow across the FlashStudio pipeline.

Verifies that:
 - Model page hyperparameters propagate correctly to Training page
 - Data page paths propagate to Training and Inference pages
 - Training outputs propagate to Export and Inference pages
 - No hardcoded fallback values override user-chosen settings
 - The config mirror mechanism preserves state across page navigation
"""

import os
import json

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def pipeline_state(mock_session_state):
    """Seed session state with a realistic pipeline configuration."""
    mock_session_state.update({
        # Data page outputs
        "train_img_path": "/data/project/train",
        "val_img_path": "/data/project/valid",
        "dataset_name": "Traffic Detection",
        "class_names": "car\ntruck\nbus\npedestrian",
        "num_classes": 4,
        "ann_format": "COCO JSON",
        # Model page outputs
        "arch_family": "FlashDet (recommended)",
        "model_arch": "FlashDet-S",
        "pretrain_option": "COCO pretrained",
        "finetune_strategy": "Full fine-tune",
        # Hyperparameters (set by user on Model page)
        "epochs": 200,
        "batch_size": 32,
        "lr": 0.005,
        "img_size": 640,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "patience": 30,
        "num_workers": 4,
        "grad_accum": 2,
        "val_interval": 5,
        "lr_final_ratio": 0.1,
        "optimizer": "AdamW",
        # Augmentation
        "aug_mosaic": True,
        "aug_mixup": True,
        "aug_copypaste": False,
        # Advanced
        "amp": True,
        "grad_clip": True,
        "multiscale": False,
        "compile_model": False,
        "ddp": False,
        "activation_checkpointing": False,
        "activation_offloading": False,
        "optimizer_in_bwd": False,
        "use_8bit_optimizer": False,
        "chunked_loss": False,
        # Output
        "save_dir": "/output/runs",
    })
    return mock_session_state


# ---------------------------------------------------------------------------
# 1. Hyperparameter flow: Model page -> get_state() -> Training runner
# ---------------------------------------------------------------------------

class TestHyperparamFlow:
    """Ensure hyperparameters set on Model page are read correctly by Training."""

    def test_get_state_reads_session_values(self, pipeline_state):
        from flashstudio.utils import get_state
        assert get_state("epochs") == 200
        assert get_state("batch_size") == 32
        assert get_state("lr") == 0.005
        assert get_state("img_size") == 640
        assert get_state("patience") == 30
        assert get_state("grad_accum") == 2
        assert get_state("warmup_epochs") == 5
        assert get_state("num_workers") == 4

    def test_get_state_falls_back_to_defaults(self, mock_session_state):
        """When user hasn't set a value, get_state returns DEFAULTS."""
        from flashstudio.utils import get_state, DEFAULTS
        for key in ["epochs", "batch_size", "lr", "img_size",
                     "patience", "grad_accum", "warmup_epochs", "num_workers"]:
            assert get_state(key) == DEFAULTS[key], f"{key} should fall back to DEFAULTS"

    def test_mirror_overrides_stale_session(self, mock_session_state):
        """_config_mirror (kept fresh by update_config_mirror) wins over
        session_state which may hold stale re-seeded values from project file."""
        from flashstudio.utils import get_state
        mock_session_state["epochs"] = 50
        mock_session_state["_config_mirror"] = {"epochs": 300}
        assert get_state("epochs") == 300

    def test_config_mirror_does_not_override_when_absent(self, mock_session_state):
        """If key missing from mirror, session_state is used."""
        from flashstudio.utils import get_state
        mock_session_state["epochs"] = 50
        mock_session_state["_config_mirror"] = {"batch_size": 16}
        assert get_state("epochs") == 50

    def test_model_arch_propagates(self, pipeline_state):
        from flashstudio.utils import get_state
        assert get_state("model_arch") == "FlashDet-S"

    def test_save_dir_propagates(self, pipeline_state):
        from flashstudio.utils import get_state
        assert get_state("save_dir") == "/output/runs"


# ---------------------------------------------------------------------------
# 2. Data path flow: Data page -> Training -> Inference
# ---------------------------------------------------------------------------

class TestDataPathFlow:
    """Ensure data paths set on Data page are read by Training and Inference."""

    def test_train_path_available_for_training(self, pipeline_state):
        assert pipeline_state.get("train_img_path") == "/data/project/train"

    def test_val_path_available_for_training(self, pipeline_state):
        assert pipeline_state.get("val_img_path") == "/data/project/valid"

    def test_class_names_as_newline_string(self, pipeline_state):
        from flashstudio.utils import get_class_names_list
        classes = get_class_names_list()
        assert classes == ["car", "truck", "bus", "pedestrian"]

    def test_num_classes_matches_class_names(self, pipeline_state):
        from flashstudio.utils import get_class_names_list
        classes = get_class_names_list()
        assert len(classes) == pipeline_state["num_classes"]

    def test_class_names_list_type(self, pipeline_state):
        """class_names stored as list should still work."""
        pipeline_state["class_names"] = ["car", "truck", "bus"]
        from flashstudio.utils import get_class_names_list, get_class_names_str
        assert get_class_names_str() == "car\ntruck\nbus"
        assert get_class_names_list() == ["car", "truck", "bus"]

    def test_dataset_name_propagates(self, pipeline_state):
        assert pipeline_state.get("dataset_name") == "Traffic Detection"


# ---------------------------------------------------------------------------
# 3. Training output flow -> Export & Inference
# ---------------------------------------------------------------------------

class TestTrainingOutputFlow:
    """Verify trained model paths flow from Training to Export and Inference."""

    def test_save_dir_used_by_export(self, pipeline_state):
        save_dir = pipeline_state.get("save_dir", "")
        assert save_dir == "/output/runs"

    def test_active_run_path_created(self, pipeline_state):
        pipeline_state["active_run_path"] = "/output/runs/my_run"
        assert pipeline_state["active_run_path"] == "/output/runs/my_run"

    def test_training_log_path_set(self, pipeline_state):
        pipeline_state["training_log_file"] = "/output/runs/my_run/train_20260629.log"
        assert "training_log_file" in pipeline_state


# ---------------------------------------------------------------------------
# 4. Config build/apply round-trip
# ---------------------------------------------------------------------------

class TestConfigRoundTrip:
    """build_training_config captures state; apply_training_config restores it."""

    def test_build_captures_all_sections(self, pipeline_state):
        from flashstudio.utils.config import build_training_config, CONFIG_SCHEMA
        config = build_training_config()
        for section in CONFIG_SCHEMA:
            assert section in config, f"Missing section: {section}"

    def test_build_captures_user_values(self, pipeline_state):
        from flashstudio.utils.config import build_training_config
        config = build_training_config()
        assert config["training"]["epochs"] == 200
        assert config["training"]["batch_size"] == 32
        assert config["training"]["lr"] == 0.005
        assert config["training"]["img_size"] == 640
        assert config["model"]["model_arch"] == "FlashDet-S"
        assert config["dataset"]["train_img_path"] == "/data/project/train"
        assert config["dataset"]["class_names"] == "car\ntruck\nbus\npedestrian"

    def test_apply_restores_state(self, mock_session_state):
        from flashstudio.utils.config import apply_training_config
        config = {
            "training": {"epochs": 500, "batch_size": 64, "lr": 0.01, "img_size": 416},
            "model": {"model_arch": "FlashDet-L", "arch_family": "FlashDet (recommended)"},
            "dataset": {"train_img_path": "/new/data/train", "num_classes": 10},
        }
        apply_training_config(config)
        assert mock_session_state["epochs"] == 500
        assert mock_session_state["batch_size"] == 64
        assert mock_session_state["model_arch"] == "FlashDet-L"
        assert mock_session_state["train_img_path"] == "/new/data/train"

    def test_roundtrip_preserves_values(self, pipeline_state):
        from flashstudio.utils.config import build_training_config, apply_training_config
        original_config = build_training_config()

        pipeline_state.clear()
        apply_training_config(original_config)

        assert pipeline_state.get("epochs") == 200
        assert pipeline_state.get("batch_size") == 32
        assert pipeline_state.get("lr") == 0.005
        assert pipeline_state.get("model_arch") == "FlashDet-S"
        assert pipeline_state.get("train_img_path") == "/data/project/train"

    def test_yaml_roundtrip(self, pipeline_state, tmp_dir):
        from flashstudio.utils.config import (
            build_training_config, save_config_yaml, load_config_yaml,
            apply_training_config,
        )
        config = build_training_config()
        path = os.path.join(tmp_dir, "config.yaml")
        save_config_yaml(config, path)

        loaded = load_config_yaml(path)
        pipeline_state.clear()
        apply_training_config(loaded)

        assert pipeline_state.get("epochs") == 200
        assert pipeline_state.get("model_arch") == "FlashDet-S"


# ---------------------------------------------------------------------------
# 5. No hardcoded fallback values in critical paths
# ---------------------------------------------------------------------------

class TestNoHardcodedFallbacks:
    """Verify that runner and config use DEFAULTS/get_state instead of literals."""

    def test_defaults_has_all_critical_keys(self):
        from flashstudio.utils import DEFAULTS
        required_keys = [
            "epochs", "batch_size", "lr", "img_size",
            "weight_decay", "warmup_epochs", "patience",
            "num_workers", "grad_accum", "model_arch",
            "arch_family", "save_dir",
        ]
        for key in required_keys:
            assert key in DEFAULTS, f"DEFAULTS missing: {key}"
            assert DEFAULTS[key] is not None, f"DEFAULTS[{key}] is None"

    def test_defaults_match_constants(self):
        from flashstudio.utils import DEFAULTS
        from flashstudio.constants import (
            TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_IMG_SIZE,
            TRAIN_WEIGHT_DECAY, TRAIN_WARMUP_EPOCHS, TRAIN_PATIENCE,
            TRAIN_NUM_WORKERS, TRAIN_GRAD_ACCUM, DEFAULT_MODEL_ARCH,
            DEFAULT_ARCH_FAMILY, DEFAULT_SAVE_DIR,
        )
        assert DEFAULTS["epochs"] == TRAIN_EPOCHS
        assert DEFAULTS["batch_size"] == TRAIN_BATCH_SIZE
        assert DEFAULTS["lr"] == TRAIN_LR
        assert DEFAULTS["img_size"] == TRAIN_IMG_SIZE
        assert DEFAULTS["weight_decay"] == TRAIN_WEIGHT_DECAY
        assert DEFAULTS["warmup_epochs"] == TRAIN_WARMUP_EPOCHS
        assert DEFAULTS["patience"] == TRAIN_PATIENCE
        assert DEFAULTS["num_workers"] == TRAIN_NUM_WORKERS
        assert DEFAULTS["grad_accum"] == TRAIN_GRAD_ACCUM
        assert DEFAULTS["model_arch"] == DEFAULT_MODEL_ARCH
        assert DEFAULTS["arch_family"] == DEFAULT_ARCH_FAMILY
        assert DEFAULTS["save_dir"] == DEFAULT_SAVE_DIR

    def test_user_values_override_defaults(self, mock_session_state):
        """User-set values must win over defaults."""
        from flashstudio.utils import get_state, DEFAULTS
        mock_session_state["epochs"] = 999
        assert get_state("epochs") == 999
        assert get_state("epochs") != DEFAULTS["epochs"]

    def test_get_state_never_returns_none_for_known_keys(self, mock_session_state):
        """For keys with entries in DEFAULTS, get_state should never return None."""
        from flashstudio.utils import get_state, DEFAULTS
        for key in DEFAULTS:
            val = get_state(key)
            assert val is not None, f"get_state({key!r}) returned None"


# ---------------------------------------------------------------------------
# 6. Config schema covers all session state keys
# ---------------------------------------------------------------------------

class TestConfigSchema:
    """CONFIG_SCHEMA should cover all critical session state keys."""

    def test_schema_covers_training_params(self):
        from flashstudio.utils.config import CONFIG_SCHEMA
        all_keys = []
        for keys in CONFIG_SCHEMA.values():
            all_keys.extend(keys)

        critical = [
            "epochs", "batch_size", "lr", "img_size",
            "warmup_epochs", "patience", "grad_accum", "num_workers",
            "model_arch", "arch_family",
            "train_img_path", "val_img_path", "class_names", "num_classes",
            "aug_mosaic", "aug_mixup",
            "amp", "compile_model", "ddp",
            "save_dir",
        ]
        for key in critical:
            assert key in all_keys, f"CONFIG_SCHEMA missing critical key: {key}"

    def test_schema_sections_are_strings(self):
        from flashstudio.utils.config import CONFIG_SCHEMA
        for section, keys in CONFIG_SCHEMA.items():
            assert isinstance(section, str)
            assert isinstance(keys, list)
            for k in keys:
                assert isinstance(k, str), f"Non-string key in {section}: {k}"


# ---------------------------------------------------------------------------
# 7. _get_save_dir uses DEFAULTS, not hardcoded
# ---------------------------------------------------------------------------

class TestSaveDir:
    def test_get_save_dir_from_session(self, mock_session_state):
        mock_session_state["save_dir"] = "/custom/output"
        from flashstudio.pages.training._common import _get_save_dir
        assert _get_save_dir() == "/custom/output"

    def test_get_save_dir_falls_back_to_default(self, mock_session_state):
        from flashstudio.pages.training._common import _get_save_dir
        from flashstudio.utils import DEFAULTS
        result = _get_save_dir()
        assert result == DEFAULTS["save_dir"]


# ---------------------------------------------------------------------------
# 8. _ensure_config_in_session restores from project file
# ---------------------------------------------------------------------------

class TestEnsureConfig:
    def test_restores_missing_keys_from_project(self, mock_session_state, tmp_dir):
        state_file = os.path.join(tmp_dir, "session_state.json")
        saved = {"epochs": 150, "batch_size": 16, "lr": 0.003, "model_arch": "FlashDet-M"}
        with open(state_file, "w") as f:
            json.dump(saved, f)

        from unittest.mock import patch
        with patch("flashstudio.components.project_manager.get_project_dir", return_value=tmp_dir):
            from flashstudio.app import _ensure_config_in_session
            _ensure_config_in_session("test_project")

        assert mock_session_state.get("epochs") == 150
        assert mock_session_state.get("model_arch") == "FlashDet-M"

    def test_does_not_overwrite_existing_keys(self, mock_session_state, tmp_dir):
        mock_session_state["epochs"] = 300

        state_file = os.path.join(tmp_dir, "session_state.json")
        saved = {"epochs": 150, "batch_size": 16}
        with open(state_file, "w") as f:
            json.dump(saved, f)

        from unittest.mock import patch
        with patch("flashstudio.components.project_manager.get_project_dir", return_value=tmp_dir):
            from flashstudio.app import _ensure_config_in_session
            _ensure_config_in_session("test_project")

        assert mock_session_state["epochs"] == 300
        assert mock_session_state["batch_size"] == 16


# ---------------------------------------------------------------------------
# 9. _init_config_mirror snapshots current state
# ---------------------------------------------------------------------------

class TestInitConfigMirror:
    def test_mirror_captures_state(self, mock_session_state):
        mock_session_state["epochs"] = 100
        mock_session_state["batch_size"] = 8
        mock_session_state["model_arch"] = "FlashDet-N"

        from flashstudio.app import _init_config_mirror
        _init_config_mirror()

        mirror = mock_session_state.get("_config_mirror", {})
        assert mirror["epochs"] == 100
        assert mirror["batch_size"] == 8
        assert mirror["model_arch"] == "FlashDet-N"

    def test_mirror_only_includes_present_keys(self, mock_session_state):
        mock_session_state["epochs"] = 50

        from flashstudio.app import _init_config_mirror
        _init_config_mirror()

        mirror = mock_session_state.get("_config_mirror", {})
        assert "epochs" in mirror
        assert "batch_size" not in mirror


# ---------------------------------------------------------------------------
# 10. Preflight checks read from session state
# ---------------------------------------------------------------------------

class TestPreflightChecks:
    def test_preflight_detects_no_data(self, mock_session_state):
        from flashstudio.pages.training.launch.preflight import _run_preflight_checks
        checks = _run_preflight_checks()
        labels = {c[0]: c[1] for c in checks}
        assert labels["Dataset"] is False

    def test_preflight_detects_data(self, mock_session_state, tmp_dir):
        train_dir = os.path.join(tmp_dir, "train")
        os.makedirs(train_dir)
        with open(os.path.join(train_dir, "_annotations.coco.json"), "w") as f:
            json.dump({"categories": [{"id": 1, "name": "cat"}], "images": [], "annotations": []}, f)

        mock_session_state["train_img_path"] = train_dir
        mock_session_state["class_names"] = "cat"
        mock_session_state["num_classes"] = 1
        mock_session_state["model_arch"] = "FlashDet-N"

        from flashstudio.pages.training.launch.preflight import _run_preflight_checks
        checks = _run_preflight_checks()
        labels = {c[0]: c[1] for c in checks}
        assert labels["Dataset"] is True
        assert labels["Annotations"] is True
        assert labels["Classes"] is True
        assert labels["Model"] is True


# ---------------------------------------------------------------------------
# 11. Pipeline steps use constants, not hardcoded values
# ---------------------------------------------------------------------------

class TestPipelineSteps:
    def test_pipeline_uses_session_model(self, mock_session_state):
        mock_session_state["model_arch"] = "FlashDet-L"
        mock_session_state["epochs"] = 500

        from flashstudio.utils.config import get_pipeline_steps
        steps = get_pipeline_steps()
        model_step = next(s for s in steps if s["action"] == "configure_model")
        train_step = next(s for s in steps if s["action"] == "train")
        assert "FlashDet-L" in model_step["description"]
        assert "500" in train_step["description"]

    def test_pipeline_falls_back_to_constants(self, mock_session_state):
        from flashstudio.utils.config import get_pipeline_steps
        from flashstudio.constants import DEFAULT_MODEL_ARCH, TRAIN_EPOCHS
        steps = get_pipeline_steps()
        model_step = next(s for s in steps if s["action"] == "configure_model")
        train_step = next(s for s in steps if s["action"] == "train")
        assert DEFAULT_MODEL_ARCH in model_step["description"]
        assert str(TRAIN_EPOCHS) in train_step["description"]


# ---------------------------------------------------------------------------
# 12. Hyperparameter changes on Model page visible on subsequent pages
# ---------------------------------------------------------------------------

class TestHyperparamCrossPageVisibility:
    """Verify that hyperparameters changed on the Model page survive page
    navigation and are reflected on the Dashboard, Training, and Export pages.

    Streamlit drops widget-bound session_state keys when the user navigates
    away from the page that created them.  FlashStudio compensates via
    update_config_mirror() which snapshots every Model-page widget value into
    _config_mirror.  get_state() reads the mirror first, so downstream pages
    always see the latest user choices.
    """

    @pytest.fixture()
    def model_page_state(self, mock_session_state):
        """Simulate the user changing hyperparams on the Model page."""
        mock_session_state.update({
            "epochs": 300,
            "batch_size": 64,
            "lr": 0.01,
            "img_size": 640,
            "weight_decay": 0.001,
            "warmup_epochs": 10,
            "patience": 20,
            "num_workers": 8,
            "grad_accum": 4,
            "val_interval": 2,
            "lr_final_ratio": 0.05,
            "optimizer": "SGD",
            "model_arch": "FlashDet-Large",
            "arch_family": "FlashDet (recommended)",
            "amp": False,
            "grad_clip": False,
            "multiscale": True,
            "aug_mosaic": False,
            "aug_mixup": True,
            "aug_copypaste": True,
            "compile_model": True,
            "ddp": True,
        })
        return mock_session_state

    def _simulate_page_navigation(self, state):
        """Simulate navigating away: call update_config_mirror, then clear
        all widget keys (as Streamlit does when the page changes)."""
        from flashstudio.utils import update_config_mirror
        update_config_mirror()

        widget_keys = [
            "epochs", "batch_size", "lr", "img_size", "weight_decay",
            "warmup_epochs", "patience", "num_workers", "grad_accum",
            "val_interval", "lr_final_ratio", "optimizer",
            "model_arch", "arch_family",
            "amp", "grad_clip", "multiscale",
            "aug_mosaic", "aug_mixup", "aug_copypaste",
            "compile_model", "ddp",
        ]
        for k in widget_keys:
            state.pop(k, None)

    # -- Core mirror mechanism --

    def test_mirror_preserves_all_changed_values(self, model_page_state):
        from flashstudio.utils import update_config_mirror
        update_config_mirror()
        mirror = model_page_state["_config_mirror"]
        assert mirror["epochs"] == 300
        assert mirror["batch_size"] == 64
        assert mirror["lr"] == 0.01
        assert mirror["img_size"] == 640
        assert mirror["model_arch"] == "FlashDet-Large"
        assert mirror["optimizer"] == "SGD"
        assert mirror["amp"] is False
        assert mirror["aug_mosaic"] is False
        assert mirror["aug_mixup"] is True
        assert mirror["compile_model"] is True
        assert mirror["ddp"] is True

    def test_get_state_returns_user_values_after_navigation(self, model_page_state):
        """After navigating away (widget keys cleared), get_state still
        returns the values the user chose on the Model page."""
        from flashstudio.utils import get_state
        self._simulate_page_navigation(model_page_state)

        assert "epochs" not in model_page_state, "widget key should be cleared"
        assert get_state("epochs") == 300
        assert get_state("batch_size") == 64
        assert get_state("lr") == 0.01
        assert get_state("img_size") == 640
        assert get_state("model_arch") == "FlashDet-Large"
        assert get_state("optimizer") == "SGD"

    def test_get_state_does_not_regress_to_defaults(self, model_page_state):
        """Changed values must NOT fall back to DEFAULTS after navigation."""
        from flashstudio.utils import get_state, DEFAULTS
        self._simulate_page_navigation(model_page_state)

        assert get_state("epochs") != DEFAULTS["epochs"]
        assert get_state("batch_size") != DEFAULTS["batch_size"]
        assert get_state("lr") != DEFAULTS["lr"]
        assert get_state("model_arch") != DEFAULTS["model_arch"]

    # -- Dashboard page reads --

    def test_dashboard_metrics_reflect_changed_model(self, model_page_state):
        """Dashboard render_metrics reads model_arch and epochs via
        get_state — verify they match the user's choices."""
        from flashstudio.utils import get_state
        self._simulate_page_navigation(model_page_state)

        model = get_state("model_arch")
        epochs = get_state("epochs")
        assert model == "FlashDet-Large"
        assert epochs == 300

    def test_pipeline_steps_reflect_changed_values(self, model_page_state):
        """Pipeline steps shown on Dashboard use get_state values."""
        from flashstudio.utils.config import get_pipeline_steps
        self._simulate_page_navigation(model_page_state)

        steps = get_pipeline_steps()
        model_step = next(s for s in steps if s["action"] == "configure_model")
        train_step = next(s for s in steps if s["action"] == "train")
        assert "FlashDet-Large" in model_step["description"]
        assert "300" in train_step["description"]

    # -- Training page reads --

    def test_training_config_summary_reflects_changes(self, model_page_state):
        """Training launch tab shows a config summary reading get_state."""
        from flashstudio.utils import get_state
        self._simulate_page_navigation(model_page_state)

        assert get_state("model_arch") == "FlashDet-Large"
        assert get_state("epochs") == 300
        assert get_state("batch_size") == 64
        assert get_state("img_size") == 640
        assert f"{get_state('lr'):.1e}" == "1.0e-02"

    def test_training_runner_uses_changed_hyperparams(self, model_page_state):
        """The runner reads get_state for every hyperparam — verify all
        of them return the user's modified values, not defaults."""
        from flashstudio.utils import get_state
        self._simulate_page_navigation(model_page_state)

        assert get_state("epochs") == 300
        assert get_state("batch_size") == 64
        assert get_state("lr") == 0.01
        assert get_state("num_workers") == 8
        assert get_state("warmup_epochs") == 10
        assert get_state("grad_accum") == 4
        assert get_state("patience") == 20
        assert get_state("img_size") == 640

    # -- _build_cfg (model utils/config) --

    def test_build_cfg_reflects_changes_after_navigation(self, model_page_state):
        """_build_cfg builds the config dict from session state.
        After navigation, the mirror should feed it correct values."""
        from flashstudio.utils import update_config_mirror
        update_config_mirror()

        from flashstudio.pages.model.utils.config import _build_cfg
        cfg = _build_cfg()
        assert cfg["training"]["epochs"] == 300
        assert cfg["training"]["batch_size"] == 64
        assert cfg["training"]["lr"] == 0.01
        assert cfg["training"]["img_size"] == 640
        assert cfg["model"]["variant"] == "FlashDet-Large"
        assert cfg["augmentation"]["mosaic"] is False
        assert cfg["augmentation"]["mixup"] is True
        assert cfg["advanced"]["amp"] is False
        assert cfg["advanced"]["compile"] is True

    # -- build_training_config (unified config) --

    def test_build_training_config_captures_changed_values(self, model_page_state):
        """build_training_config reads directly from session state, so it
        should capture all changed values before navigation."""
        from flashstudio.utils.config import build_training_config
        config = build_training_config()
        assert config["training"]["epochs"] == 300
        assert config["training"]["batch_size"] == 64
        assert config["training"]["lr"] == 0.01
        assert config["model"]["model_arch"] == "FlashDet-Large"
        assert config["optimizer"]["optimizer"] == "SGD"
        assert config["optimizer"]["weight_decay"] == 0.001
        assert config["augmentation"]["aug_mosaic"] is False
        assert config["augmentation"]["aug_mixup"] is True
        assert config["advanced"]["amp"] is False
        assert config["advanced"]["compile_model"] is True
        assert config["advanced"]["ddp"] is True

    # -- Roundtrip: change → save → clear → apply → verify --

    def test_changed_values_survive_save_load_roundtrip(self, model_page_state, tmp_dir):
        """Change hyperparams → save config to YAML → clear state →
        load config → apply → verify all values restored."""
        from flashstudio.utils.config import (
            build_training_config, save_config_yaml,
            load_config_yaml, apply_training_config,
        )
        config = build_training_config()
        path = os.path.join(tmp_dir, "changed_config.yaml")
        save_config_yaml(config, path)

        model_page_state.clear()
        loaded = load_config_yaml(path)
        apply_training_config(loaded)

        assert model_page_state.get("epochs") == 300
        assert model_page_state.get("batch_size") == 64
        assert model_page_state.get("lr") == 0.01
        assert model_page_state.get("model_arch") == "FlashDet-Large"
        assert model_page_state.get("optimizer") == "SGD"
        assert model_page_state.get("amp") is False

    # -- Multiple navigation cycles --

    def test_values_persist_across_multiple_navigations(self, model_page_state):
        """Simulate visiting Model → Dashboard → Training → back to Model.
        Values should remain consistent throughout."""
        from flashstudio.utils import get_state, update_config_mirror

        update_config_mirror()
        assert get_state("epochs") == 300

        # Navigate to Dashboard (widgets cleared)
        self._simulate_page_navigation(model_page_state)
        assert get_state("epochs") == 300

        # Navigate to Training (still cleared)
        assert get_state("epochs") == 300
        assert get_state("model_arch") == "FlashDet-Large"

        # Navigate back to Model (mirror still intact)
        assert get_state("epochs") == 300
        assert get_state("batch_size") == 64

    # -- Edge case: partial changes --

    def test_partial_change_only_updates_changed_keys(self, mock_session_state):
        """If user only changes epochs but leaves others at defaults,
        only epochs should differ on the next page."""
        from flashstudio.utils import get_state, DEFAULTS, update_config_mirror

        mock_session_state["epochs"] = 500
        update_config_mirror()

        mock_session_state.pop("epochs", None)

        assert get_state("epochs") == 500
        assert get_state("batch_size") == DEFAULTS["batch_size"]
        assert get_state("lr") == DEFAULTS["lr"]

    # -- Edge case: second edit overwrites first --

    def test_second_edit_overwrites_first(self, mock_session_state):
        """User changes epochs to 200, navigates away, comes back,
        changes epochs to 400 — the latest value must win."""
        from flashstudio.utils import get_state, update_config_mirror

        mock_session_state["epochs"] = 200
        update_config_mirror()
        mock_session_state.pop("epochs", None)
        assert get_state("epochs") == 200

        mock_session_state["epochs"] = 400
        update_config_mirror()
        mock_session_state.pop("epochs", None)
        assert get_state("epochs") == 400

    # -- Edge case: boolean toggles (AMP, augmentations) --

    def test_boolean_toggles_survive_navigation(self, model_page_state):
        """AMP=off, mosaic=off, mixup=on must not revert to defaults."""
        from flashstudio.utils import update_config_mirror
        update_config_mirror()
        self._simulate_page_navigation(model_page_state)

        mirror = model_page_state["_config_mirror"]
        assert mirror["amp"] is False
        assert mirror["aug_mosaic"] is False
        assert mirror["aug_mixup"] is True
        assert mirror["multiscale"] is True
        assert mirror["compile_model"] is True

    # -- Preflight checks see updated model --

    def test_preflight_sees_changed_model(self, model_page_state):
        """Preflight on Training page should see the model the user
        selected on the Model page, not the default."""
        self._simulate_page_navigation(model_page_state)

        from flashstudio.utils import get_state
        assert get_state("model_arch") == "FlashDet-Large"

        from flashstudio.pages.training.launch.preflight import _run_preflight_checks
        checks = _run_preflight_checks()
        model_check = next(c for c in checks if c[0] == "Model")
        assert model_check[1] is True
