"""Tests for build_training_config and apply_training_config."""

from flashstudio.utils.config import (
    build_training_config, apply_training_config, CONFIG_SCHEMA,
    get_pipeline_steps,
)


class TestBuildTrainingConfig:
    def test_returns_dict_with_sections(self, mock_session_state):
        config = build_training_config()
        assert isinstance(config, dict)
        for section in CONFIG_SCHEMA:
            assert section in config

    def test_captures_session_values(self, mock_session_state):
        mock_session_state["epochs"] = 200
        mock_session_state["batch_size"] = 64
        mock_session_state["model_arch"] = "FlashDet-Large"
        config = build_training_config()
        assert config["training"]["epochs"] == 200
        assert config["training"]["batch_size"] == 64
        assert config["model"]["model_arch"] == "FlashDet-Large"

    def test_skips_none_values(self, mock_session_state):
        config = build_training_config()
        for section_data in config.values():
            for val in section_data.values():
                assert val is not None

    def test_includes_all_sections(self, mock_session_state):
        config = build_training_config()
        expected = ["project", "dataset", "model", "training",
                    "optimizer", "augmentation", "finetune", "advanced", "output"]
        for s in expected:
            assert s in config

    def test_dataset_section(self, mock_session_state):
        mock_session_state["dataset_name"] = "MyDataset"
        mock_session_state["train_img_path"] = "/data/train"
        mock_session_state["val_img_path"] = "/data/val"
        config = build_training_config()
        assert config["dataset"]["dataset_name"] == "MyDataset"
        assert config["dataset"]["train_img_path"] == "/data/train"

    def test_augmentation_section(self, mock_session_state):
        mock_session_state["aug_mosaic"] = True
        mock_session_state["aug_mixup"] = False
        config = build_training_config()
        assert config["augmentation"]["aug_mosaic"] is True
        assert config["augmentation"]["aug_mixup"] is False

    def test_finetune_section(self, mock_session_state):
        mock_session_state["finetune_strategy"] = "LoRA"
        mock_session_state["lora_rank"] = 16
        config = build_training_config()
        assert config["finetune"]["finetune_strategy"] == "LoRA"
        assert config["finetune"]["lora_rank"] == 16


class TestApplyTrainingConfig:
    def test_applies_values_to_session(self, mock_session_state):
        config = {
            "training": {"epochs": 300, "batch_size": 128},
            "model": {"model_arch": "FlashDet-Nano"},
        }
        apply_training_config(config)
        assert mock_session_state["epochs"] == 300
        assert mock_session_state["batch_size"] == 128
        assert mock_session_state["model_arch"] == "FlashDet-Nano"

    def test_skips_missing_sections(self, mock_session_state):
        config = {"training": {"epochs": 50}}
        apply_training_config(config)
        assert mock_session_state["epochs"] == 50
        assert "model_arch" not in mock_session_state

    def test_skips_none_values(self, mock_session_state):
        config = {"training": {"epochs": None, "batch_size": 32}}
        apply_training_config(config)
        assert "epochs" not in mock_session_state
        assert mock_session_state["batch_size"] == 32

    def test_roundtrip(self, mock_session_state):
        mock_session_state["epochs"] = 500
        mock_session_state["lr"] = 0.005
        mock_session_state["model_arch"] = "FlashDet-Small"
        original = build_training_config()

        for k in list(mock_session_state.keys()):
            del mock_session_state[k]

        apply_training_config(original)
        assert mock_session_state["epochs"] == 500
        assert mock_session_state["lr"] == 0.005
        assert mock_session_state["model_arch"] == "FlashDet-Small"

    def test_unknown_keys_in_section_ignored(self, mock_session_state):
        config = {"training": {"epochs": 100, "unknown_key_xyz": 999}}
        apply_training_config(config)
        assert mock_session_state["epochs"] == 100
        assert "unknown_key_xyz" not in mock_session_state


class TestGetPipelineSteps:
    def test_returns_list(self, mock_session_state):
        steps = get_pipeline_steps()
        assert isinstance(steps, list)
        assert len(steps) == 7

    def test_step_numbers_sequential(self, mock_session_state):
        steps = get_pipeline_steps()
        nums = [s["step"] for s in steps]
        assert nums == list(range(1, 8))

    def test_steps_include_model_name(self, mock_session_state):
        mock_session_state["_config_mirror"] = {"model_arch": "FlashDet-Large"}
        steps = get_pipeline_steps()
        descriptions = " ".join(s["description"] for s in steps)
        assert "FlashDet-Large" in descriptions

    def test_steps_include_epoch_count(self, mock_session_state):
        mock_session_state["_config_mirror"] = {"epochs": 999}
        steps = get_pipeline_steps()
        descriptions = " ".join(s["description"] for s in steps)
        assert "999" in descriptions

    def test_each_step_has_required_keys(self, mock_session_state):
        steps = get_pipeline_steps()
        for step in steps:
            assert "step" in step
            assert "action" in step
            assert "description" in step
