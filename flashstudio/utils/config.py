"""Unified training config build/apply — single source of truth for YAML configs."""

import os
import yaml
import streamlit as st


# Keys to persist in config grouped by section
CONFIG_SCHEMA = {
    "project": ["run_name"],
    "dataset": [
        "dataset_name", "train_img_path", "val_img_path",
        "ann_format", "upload_num_classes", "class_names", "num_classes",
    ],
    "model": [
        "arch_family", "model_arch", "pico_backbone",
        "pretrain_option", "custom_weights",
    ],
    "training": [
        "epochs", "batch_size", "lr", "img_size", "input_size",
        "warmup_epochs", "patience", "grad_accum", "num_workers",
        "val_interval", "lr_final_ratio", "grad_clip", "multiscale",
    ],
    "optimizer": ["optimizer", "weight_decay", "use_8bit_optimizer"],
    "augmentation": ["aug_mosaic", "aug_mixup", "aug_copypaste"],
    "finetune": [
        "finetune_strategy", "lora_variant", "lora_rank",
        "lora_alpha", "lora_dropout", "lora_targets",
        "qlora", "qlora_dtype",
    ],
    "advanced": [
        "amp", "compile_model", "ddp",
        "activation_checkpointing", "activation_offloading",
        "optimizer_in_bwd", "chunked_loss", "chunk_size",
        "save_best", "resume_training",
    ],
    "output": ["save_dir"],
}


def build_training_config() -> dict:
    """Build a complete training config dictionary from current session state.

    Uses mirror-aware reads so values survive page navigation.
    """
    from flashstudio.utils import get_state
    config = {}
    for section, keys in CONFIG_SCHEMA.items():
        config[section] = {}
        for key in keys:
            val = get_state(key)
            if val is not None:
                config[section][key] = val
    return config


def apply_training_config(config: dict):
    """Apply a loaded config dictionary to session state."""
    for section, keys in CONFIG_SCHEMA.items():
        if section not in config:
            continue
        section_data = config[section]
        for key in keys:
            if key in section_data:
                val = section_data[key]
                if val is not None:
                    st.session_state[key] = val


def save_config_yaml(config: dict, path: str):
    """Save config dict to a YAML file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config_yaml(path: str) -> dict:
    """Load config from a YAML file. Returns empty dict on failure."""
    try:
        with open(path) as f:
            return yaml.safe_load(f.read()) or {}
    except (OSError, yaml.YAMLError):
        return {}


def config_to_yaml_str(config: dict) -> str:
    """Convert config dict to YAML string for display/download."""
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def get_pipeline_steps() -> list:
    """Generate pipeline step descriptions based on current config."""
    from flashstudio.utils import get_state
    model = get_state("model_arch")
    epochs = get_state("epochs")
    return [
        {"step": 1, "action": "load_dataset", "description": "Load and validate dataset"},
        {"step": 2, "action": "convert_format", "description": "Convert to COCO JSON if needed"},
        {"step": 3, "action": "configure_model", "description": f"Initialize {model}"},
        {"step": 4, "action": "apply_augmentations", "description": "Apply data augmentation pipeline"},
        {"step": 5, "action": "train", "description": f"Train for {epochs} epochs"},
        {"step": 6, "action": "evaluate", "description": "Evaluate on validation set"},
        {"step": 7, "action": "export", "description": "Export best model (ONNX)"},
    ]
