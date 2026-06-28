"""FlashStudio — Model Selection & Finetune Configuration Page."""

import os
import yaml
import streamlit as st


FLASHDET_MODELS = {
    "FlashDet-Pico": {
        "size": "p", "params": "~298K", "speed": "Ultra-fast",
        "backbone": "LiteBackbone (0.5x)", "neck": "PicoNeck (64ch)",
        "head": "E2EDualHead", "best_for": "Edge / Mobile / MCU",
    },
    "FlashDet-Nano": {
        "size": "n", "params": "~790K", "speed": "Very fast",
        "backbone": "FlashBackbone (stem=32)", "neck": "PicoNeck (96ch)",
        "head": "E2EDualHead", "best_for": "Embedded / IoT",
    },
    "FlashDet-Small": {
        "size": "s", "params": "~1.8M", "speed": "Fast",
        "backbone": "FlashBackbone (stem=48)", "neck": "PicoNeck (128ch)",
        "head": "E2EDualHead", "best_for": "General purpose",
    },
    "FlashDet-Medium": {
        "size": "m", "params": "~3.6M", "speed": "Balanced",
        "backbone": "FlashBackbone (stem=64)", "neck": "PicoNeck (192ch)",
        "head": "E2EDualHead", "best_for": "High accuracy",
    },
    "FlashDet-Large": {
        "size": "l", "params": "~5.8M", "speed": "Accurate",
        "backbone": "FlashBackbone (stem=80)", "neck": "PicoNeck (256ch)",
        "head": "E2EDualHead", "best_for": "High accuracy",
    },
    "FlashDet-X": {
        "size": "x", "params": "~9.0M", "speed": "Max accuracy",
        "backbone": "FlashBackbone (stem=96)", "neck": "PicoNeck (320ch)",
        "head": "E2EDualHead", "best_for": "Max accuracy / Server",
    },
}

OTHER_ARCHITECTURES = {
    "YOLOv8": {"params": "Varies", "speed": "Fast", "best_for": "General YOLO"},
    "YOLOv9": {"params": "Varies", "speed": "Fast", "best_for": "PGI-based detection"},
    "YOLOv10": {"params": "Varies", "speed": "Fast", "best_for": "PSA-enhanced"},
    "YOLOv11": {"params": "Varies", "speed": "Fast", "best_for": "C2PSA-based"},
    "YOLOX": {"params": "Varies", "speed": "Fast", "best_for": "Anchor-free YOLO"},
}

OPTIMIZERS = ["AdamW", "SGD", "MuSGD"]
LORA_VARIANTS = ["standard", "dora", "lora_plus", "adalora", "ortho", "lora_fa"]


def render_model_page():
    """Render model selection and fine-tuning configuration."""
    from flashstudio.components.styles import render_page_header
    render_page_header("🧠", "Model Configuration",
                       "Choose your architecture and configure all training parameters.")

    tab_arch, tab_hyper, tab_aug, tab_advanced = st.tabs([
        "🏗️ Architecture", "⚙️ Hyperparameters", "🎨 Augmentation", "🔧 Advanced"
    ])

    with tab_arch:
        _render_architecture_tab()

    with tab_hyper:
        _render_hyperparameters_tab()

    with tab_aug:
        _render_augmentation_tab()

    with tab_advanced:
        _render_advanced_tab()

    st.divider()
    _render_config_summary()


def _render_architecture_tab():
    """Model architecture selection."""
    st.markdown("### Select Architecture")

    arch_family = st.radio(
        "Architecture Family",
        ["FlashDet (recommended)", "YOLOv8", "YOLOv9", "YOLOv10", "YOLOv11", "YOLOX"],
        key="arch_family",
        horizontal=True,
    )

    if arch_family == "FlashDet (recommended)":
        _render_flashdet_config()
    else:
        _render_yolo_config(arch_family.replace(" (recommended)", ""))


def _render_flashdet_config():
    """FlashDet-specific architecture configuration."""
    col_select, col_info = st.columns([1, 2])

    with col_select:
        model_size = st.radio(
            "Model Size",
            list(FLASHDET_MODELS.keys()),
            index=0,
            key="model_arch",
        )

        info = FLASHDET_MODELS[model_size]

        # Only Pico has backbone choice
        if model_size == "FlashDet-Pico":
            st.selectbox(
                "Backbone",
                ["LiteBackbone (ShuffleNetV2-0.5x)", "PicoBackbone (RepNeXt-style)"],
                key="pico_backbone",
                help="Only FlashDet-Pico allows backbone selection",
            )
        else:
            st.info(f"**Backbone:** {info['backbone']} (fixed for {model_size})")

        # Neck and Head are ALWAYS fixed — show as info, not selectable
        st.info(f"**Neck:** {info['neck']} (fixed)")
        st.info(f"**Head:** {info['head']} (fixed)")

    with col_info:
        st.markdown(f"### {model_size}")

        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Parameters", info["params"])
        with metric_cols[1]:
            st.metric("Speed", info["speed"])
        with metric_cols[2]:
            st.metric("Best For", info["best_for"])

        with st.container(border=True):
            st.markdown("**Architecture Details**")
            st.caption(f"Backbone: {info['backbone']}")
            st.caption(f"Neck: {info['neck']}")
            st.caption(f"Head: {info['head']} (o2o + o2m dual-head, DFL-free)")
            st.caption("Loss: CIoU + BCE + L1 with STAL assignment")
            st.caption("Training: ProgLoss (progressive o2m→o2o balancing)")

        _render_pretrain_and_finetune()


def _render_yolo_config(arch_name):
    """YOLO-family architecture configuration."""
    col_select, col_info = st.columns([1, 2])

    with col_select:
        st.markdown(f"#### {arch_name} Configuration")

        st.slider("Width Multiplier", 0.25, 1.5, 0.5, step=0.25, key="yolo_width_mult",
                  help="Controls channel width (0.25=nano, 0.5=small, 1.0=medium)")
        st.slider("Depth Multiplier", 0.33, 1.5, 0.33, step=0.33, key="yolo_depth_mult",
                  help="Controls network depth (0.33=small, 0.67=medium, 1.0=large)")

        if arch_name == "YOLOv9":
            st.checkbox("Enable PGI (Programmable Gradient Info)", value=True, key="yolo_use_pgi")
        elif arch_name == "YOLOv10":
            st.checkbox("Enable PSA (Partial Self-Attention)", value=True, key="yolo_use_psa")
        elif arch_name == "YOLOv11":
            st.checkbox("Enable C2PSA", value=True, key="yolo_use_c2psa")

    with col_info:
        yolo_info = OTHER_ARCHITECTURES[arch_name]
        st.markdown(f"### {arch_name}")
        st.caption(f"Best for: {yolo_info['best_for']}")
        st.caption("Backbone, Neck, Head are fixed per architecture — not user-selectable.")

        with st.container(border=True):
            st.markdown("**Note**")
            st.write(
                f"The {arch_name} architecture uses its own fixed backbone, neck, and head. "
                "You can control model capacity via width/depth multipliers."
            )

        _render_pretrain_and_finetune()


def _render_pretrain_and_finetune():
    """Shared pretrained weights and fine-tuning strategy UI."""
    with st.container(border=True):
        st.markdown("**Pretrained Weights**")
        pretrain_opt = st.radio(
            "Initialize from",
            ["COCO pretrained (recommended)", "ImageNet backbone only",
             "Random (train from scratch)", "Custom weights"],
            key="pretrain_option",
        )
        if pretrain_opt == "Custom weights":
            st.text_input("Path to weights", placeholder="/content/model.pth", key="custom_weights")

    with st.container(border=True):
        st.markdown("**Fine-tuning Strategy**")
        st.radio(
            "Training mode",
            [
                "Full fine-tune (all layers trainable)",
                "Freeze backbone (train neck + head only)",
                "Freeze backbone + neck (train head only)",
                "LoRA fine-tune (low-rank adapters)",
            ],
            key="finetune_strategy",
            help="Note: Layer freezing is not yet natively supported by FlashDet Trainer. "
                 "Currently only 'Full fine-tune' and 'LoRA' are functional.",
        )
        if st.session_state.get("finetune_strategy", "").startswith("LoRA"):
            st.selectbox("LoRA Variant", LORA_VARIANTS, key="lora_variant")
            st.slider("LoRA Rank", 4, 64, 16, key="lora_rank")
            st.slider("LoRA Alpha", 8, 128, 32, key="lora_alpha")
            st.slider("LoRA Dropout", 0.0, 0.5, 0.0, key="lora_dropout")
            st.multiselect("LoRA Targets", ["backbone", "fpn", "neck", "head"],
                           default=["backbone", "fpn"], key="lora_targets")
            st.divider()
            st.checkbox("Enable QLoRA (quantized LoRA — lower memory)", value=False, key="qlora")
            if st.session_state.get("qlora"):
                st.selectbox("QLoRA Quantization", ["int8", "int4"], key="qlora_dtype")
        elif "Freeze" in st.session_state.get("finetune_strategy", ""):
            st.warning("⚠️ Layer freezing is planned but not yet implemented in FlashDet's Trainer. "
                       "Training will proceed as full fine-tune.")


def _render_hyperparameters_tab():
    """Training hyperparameters — only options FlashDet actually supports."""
    st.markdown("### Training Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("**Core Parameters**")
            st.slider("Epochs", 1, 500, 100, key="epochs")
            st.select_slider("Batch Size", [2, 4, 8, 16, 32, 64, 128], value=16, key="batch_size")
            st.select_slider("Image Size", [320, 416, 640], value=320, key="img_size",
                             help="FlashDet default is 320; 416 and 640 also supported")
            st.slider("Learning Rate", 1e-5, 1e-1, 1e-3, format="%.5f", key="lr")
            st.slider("Weight Decay", 0.0, 0.1, 0.05, format="%.4f", key="weight_decay",
                      help="Advisory — FlashDet uses its internal weight_decay (default 0.05)")

        with st.container(border=True):
            st.markdown("**Optimizer**")
            st.selectbox("Optimizer", OPTIMIZERS, key="optimizer",
                         help="Advisory — FlashDet Trainer uses its internal optimizer factory (AdamW)")
            st.caption("FlashDet's Trainer uses AdamW internally. This setting is for reference.")
            st.checkbox("Use 8-bit Optimizer (memory saving)", value=False, key="use_8bit_optimizer")
            st.checkbox("Gradient Clipping (max_norm=10)", value=True, key="grad_clip",
                        help="Advisory — FlashDet uses grad clipping by default")

    with col2:
        with st.container(border=True):
            st.markdown("**LR Schedule**")
            st.caption("Cosine decay with linear warmup (fixed scheduler)")
            st.slider("Warmup Epochs", 0, 20, 3, key="warmup_epochs")
            st.slider("Final LR Ratio (lrf)", 0.01, 0.5, 0.1, format="%.2f", key="lr_final_ratio",
                      help="Final LR = initial_LR × lrf")

        with st.container(border=True):
            st.markdown("**Training Options**")
            st.checkbox("Mixed Precision (AMP FP16)", value=True, key="amp")
            st.caption("EMA (decay=0.9998) is always enabled")
            st.slider("DataLoader Workers", 0, 16, 4, key="num_workers")
            st.number_input("Gradient Accumulation Steps", 1, 16, 1, key="grad_accum")
            st.slider("Early Stopping Patience", 5, 100, 50, key="patience")
            st.slider("Validation Interval", 1, 20, 5, key="val_interval",
                      help="Advisory — FlashDet Trainer validates every epoch by default")

        with st.container(border=True):
            st.markdown("**Multi-Scale Training**")
            st.checkbox("Enable Multi-Scale", value=False, key="multiscale",
                        help="Randomly varies input size between 256-416 every 10 batches")


def _render_augmentation_tab():
    """Data augmentation settings — only what FlashDet actually uses."""
    st.markdown("### Data Augmentation")
    st.caption("These are the augmentation options supported by FlashDet's training pipeline.")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("**Multi-Image Augmentations** (CLI flags)")
            st.checkbox("Mosaic (4-image composite)", value=True, key="aug_mosaic")
            st.checkbox("MixUp (blend two images)", value=False, key="aug_mixup")
            st.checkbox("CopyPaste (instance-level)", value=False, key="aug_copypaste")

    with col2:
        with st.container(border=True):
            st.markdown("**Built-in Augmentations** (always applied)")
            st.caption("These run automatically during training and are not separately configurable via CLI:")
            st.markdown("""
            - Random scale jitter (0.5–1.5×)
            - Random horizontal flip
            - Brightness / Contrast / Saturation / Hue jitter
            - Letterbox resize (keep aspect ratio)
            - ImageNet normalization
            """)
            st.info("💡 Individual geometric/color params are hardcoded in FlashDet's "
                    "`TrainTransform`. The YAML `augment:` section exists but is not "
                    "wired to the dataloader.")


def _render_advanced_tab():
    """Advanced training options — verified against FlashDet source."""
    st.markdown("### Advanced Options")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("**Loss & Assignment** (fixed in FlashDet)")
            st.caption("These are NOT configurable — shown for reference only:")
            st.markdown("""
            - **Box Loss:** CIoU
            - **Cls Loss:** BCE
            - **Aux Loss:** L1 (on o2m head)
            - **Assignment:** STAL (Small-Target-Aware Label)
            - **ProgLoss:** Progressive o2m→o2o balancing
            """)
            st.checkbox("Chunked Loss (memory-efficient for large batches)", value=False,
                        key="chunked_loss")
            if st.session_state.get("chunked_loss"):
                st.number_input("Chunk Size", 256, 4096, 1024, key="chunk_size")

        with st.container(border=True):
            st.markdown("**Memory & Performance**")
            st.checkbox("Activation Checkpointing", value=False, key="activation_checkpointing",
                        help="Trade compute for memory — useful for large models")
            st.checkbox("Activation Offloading", value=False, key="activation_offloading")
            st.checkbox("Optimizer in Backward", value=False, key="optimizer_in_bwd",
                        help="Fuse optimizer step into backward pass")
            st.checkbox("Compile Model (torch.compile)", value=False, key="compile_model")

    with col2:
        with st.container(border=True):
            st.markdown("**Distributed Training**")
            st.checkbox("Multi-GPU (DDP)", value=False, key="ddp")
            if st.session_state.get("ddp"):
                st.caption("Launch via torchrun for DDP; DataParallel as fallback")

        with st.container(border=True):
            st.markdown("**Class Configuration**")
            st.text_input("Class File (.txt, one class per line)",
                          placeholder="/content/classes.txt",
                          key="class_file",
                          help="Path to a text file with class names (one per line). "
                               "If provided, overrides auto-detection from annotations.")

        with st.container(border=True):
            st.markdown("**Checkpointing**")
            _default_save_dir = os.path.join(os.getcwd(), "flashstudio_runs")
            st.text_input("Save Directory", value=_default_save_dir, key="save_dir")
            st.checkbox("Save Best Only", value=True, key="save_best")
            st.checkbox("Resume from Checkpoint", value=False, key="resume_training")
            if st.session_state.get("resume_training"):
                st.text_input("Checkpoint Path", placeholder="/content/runs/checkpoint_last.pth",
                              key="resume_path")

        with st.container(border=True):
            st.markdown("**Other Training Methods**")
            st.caption("FlashDet also supports (via separate scripts):")
            st.markdown("""
            - **SSL Pretraining** (BYOL / MoCo / SimCLR)
            - **Semi-supervised** (pseudo-labeling)
            - **Few-shot** (N-shot fine-tuning)
            - **Active Learning** (entropy/margin query)
            """)


def _render_config_summary():
    """Show configuration summary with option to export as workflow config file."""
    st.markdown("### 📋 Configuration Summary")

    arch_family = st.session_state.get("arch_family", "FlashDet (recommended)")
    model = st.session_state.get("model_arch", "FlashDet-Pico")

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"**Architecture:** {arch_family}")
            st.markdown(f"**Model:** {model}")
            st.markdown(f"**Strategy:** {st.session_state.get('finetune_strategy', 'Full fine-tune')[:25]}")
        with col2:
            st.markdown(f"**Epochs:** {st.session_state.get('epochs', 100)}")
            st.markdown(f"**Batch Size:** {st.session_state.get('batch_size', 16)}")
            st.markdown(f"**Image Size:** {st.session_state.get('img_size', 320)}")
        with col3:
            st.markdown(f"**LR:** {st.session_state.get('lr', 0.001)}")
            st.markdown(f"**Optimizer:** {st.session_state.get('optimizer', 'AdamW')}")
            st.markdown("**Scheduler:** Cosine + warmup")
        with col4:
            st.markdown(f"**AMP:** {'✅' if st.session_state.get('amp') else '❌'}")
            st.markdown("**EMA:** ✅ (always on)")
            st.markdown(f"**Mosaic:** {'✅' if st.session_state.get('aug_mosaic') else '❌'}")

    # ─── Export Workflow Config ───
    st.divider()
    st.markdown("### 📄 Export Workflow Config")
    st.caption("Save this configuration as a YAML file to define your training workflow. "
               "This file can be loaded later in the Training Dashboard to reproduce your setup.")

    col_export, col_upload = st.columns(2)

    with col_export:
        workflow_config = {
            "workflow": {
                "name": st.session_state.get("dataset_name", "untitled") + "_workflow",
                "version": "1.0",
                "description": f"Training workflow for {model}",
            },
            "model": {
                "family": arch_family,
                "variant": model,
                "pretrained": st.session_state.get("pretrain_option", "COCO pretrained"),
                "custom_weights": st.session_state.get("custom_weights", ""),
            },
            "training": {
                "epochs": st.session_state.get("epochs", 100),
                "batch_size": st.session_state.get("batch_size", 16),
                "learning_rate": st.session_state.get("lr", 0.001),
                "image_size": st.session_state.get("img_size", 320),
                "warmup_epochs": st.session_state.get("warmup_epochs", 3),
                "patience": st.session_state.get("patience", 50),
                "grad_accum": st.session_state.get("grad_accum", 1),
                "num_workers": st.session_state.get("num_workers", 4),
            },
            "optimizer": {
                "name": st.session_state.get("optimizer", "AdamW"),
                "weight_decay": st.session_state.get("weight_decay", 0.05),
                "use_8bit": st.session_state.get("use_8bit_optimizer", False),
            },
            "augmentation": {
                "mosaic": st.session_state.get("aug_mosaic", True),
                "mixup": st.session_state.get("aug_mixup", False),
                "copy_paste": st.session_state.get("aug_copypaste", False),
            },
            "finetune": {
                "strategy": st.session_state.get("finetune_strategy", "Full fine-tune"),
                "lora_variant": st.session_state.get("lora_variant", "standard"),
                "lora_rank": st.session_state.get("lora_rank", 16),
                "qlora": st.session_state.get("qlora", False),
            },
            "advanced": {
                "amp": st.session_state.get("amp", True),
                "compile": st.session_state.get("compile_model", False),
                "multi_gpu": st.session_state.get("ddp", False),
                "activation_checkpointing": st.session_state.get("activation_checkpointing", False),
            },
            "pipeline_steps": [
                {"step": 1, "action": "load_dataset", "description": "Load and validate dataset"},
                {"step": 2, "action": "convert_format", "description": "Convert to COCO JSON if needed"},
                {"step": 3, "action": "configure_model", "description": f"Initialize {model}"},
                {"step": 4, "action": "apply_augmentations", "description": "Apply data augmentation"},
                {"step": 5, "action": "train", "description": f"Train for {st.session_state.get('epochs', 100)} epochs"},
                {"step": 6, "action": "evaluate", "description": "Evaluate on validation set"},
                {"step": 7, "action": "export", "description": "Export best model (ONNX)"},
            ],
        }

        yaml_str = yaml.dump(workflow_config, default_flow_style=False, sort_keys=False)

        st.download_button(
            "📥 Download Workflow Config",
            yaml_str,
            file_name=f"workflow_{model.lower().replace('-', '_')}.yaml",
            mime="text/yaml",
            use_container_width=True,
            type="primary",
        )

    with col_upload:
        uploaded_config = st.file_uploader(
            "📂 Load Workflow Config",
            type=["yaml", "yml"],
            key="model_page_config_upload",
            help="Upload a previously saved workflow YAML to restore settings",
        )
        if uploaded_config:
            try:
                loaded = yaml.safe_load(uploaded_config.read().decode("utf-8"))
                if st.button("✅ Apply Loaded Config", key="apply_model_config", use_container_width=True):
                    _apply_workflow_config(loaded)
                    st.success("Workflow config applied!")
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to parse config: {e}")

    # Show YAML preview in expander
    with st.expander("👁️ Preview workflow config"):
        st.code(yaml_str, language="yaml")

    st.success("Configuration ready! Go to **Training Dashboard** to start.")


def _apply_workflow_config(config: dict):
    """Apply a workflow config loaded from YAML to session state."""
    section_map = {
        "training": {
            "epochs": "epochs",
            "batch_size": "batch_size",
            "learning_rate": "lr",
            "image_size": "img_size",
            "warmup_epochs": "warmup_epochs",
            "patience": "patience",
            "grad_accum": "grad_accum",
            "num_workers": "num_workers",
        },
        "optimizer": {
            "name": "optimizer",
            "weight_decay": "weight_decay",
            "use_8bit": "use_8bit_optimizer",
        },
        "augmentation": {
            "mosaic": "aug_mosaic",
            "mixup": "aug_mixup",
            "copy_paste": "aug_copypaste",
        },
        "finetune": {
            "strategy": "finetune_strategy",
            "lora_variant": "lora_variant",
            "lora_rank": "lora_rank",
            "qlora": "qlora",
        },
        "advanced": {
            "amp": "amp",
            "compile": "compile_model",
            "multi_gpu": "ddp",
            "activation_checkpointing": "activation_checkpointing",
        },
    }

    # Model section
    if "model" in config:
        m = config["model"]
        if "family" in m:
            st.session_state["arch_family"] = m["family"]
        if "variant" in m:
            st.session_state["model_arch"] = m["variant"]
        if "pretrained" in m:
            st.session_state["pretrain_option"] = m["pretrained"]
        if "custom_weights" in m:
            st.session_state["custom_weights"] = m["custom_weights"]

    for section, mapping in section_map.items():
        if section in config:
            for yaml_key, state_key in mapping.items():
                if yaml_key in config[section]:
                    val = config[section][yaml_key]
                    if val is not None:
                        st.session_state[state_key] = val
