"""FlashStudio — Model architecture tab (framework-specific flows)."""

import os

import streamlit as st

from flashstudio.constants import (
    FLASHDET_MODELS, LORA_VARIANTS, MODEL_FRAMEWORKS, PICO_BACKBONES,
    PICO_NECKS, YOLO_SERIES, YOLO_SIZES,
    LORA_RANK_DEFAULT, LORA_ALPHA_DEFAULT, LORA_DROPOUT_DEFAULT,
)
from flashstudio.utils import get_state


def _value(key, fallback):
    val = get_state(key)
    return fallback if val is None else val


def _index(options, key, fallback):
    options = list(options)
    val = _value(key, fallback)
    return options.index(val) if val in options else options.index(fallback)


YOLO_PARAMS = {
    "YOLOv8": {"n": "3.2M", "s": "11.2M", "m": "25.9M", "l": "43.7M", "x": "68.2M"},
    "YOLOv9": {"n": "~2.0M", "s": "~7.2M", "m": "~20M", "l": "~25M", "x": "~55M"},
    "YOLOv10": {"n": "2.3M", "s": "7.2M", "m": "15.4M", "l": "24.4M", "x": "29.5M"},
    "YOLOv11": {"n": "2.6M", "s": "9.4M", "m": "20.1M", "l": "25.3M", "x": "56.9M"},
}


def _set_shared(key: str, value) -> None:
    """Set a non-widget config key in session state and the mirror."""
    st.session_state[key] = value
    mirror = st.session_state.get("_config_mirror")
    if isinstance(mirror, dict):
        mirror[key] = value
        st.session_state["_config_mirror"] = mirror


def _sync_framework_from_family() -> None:
    if "model_framework" in st.session_state:
        return
    family = str(st.session_state.get("arch_family") or "")
    if family.startswith("YOLOv"):
        st.session_state["model_framework"] = "Ultralytics"
        st.session_state["yolo_series"] = family
    elif family == "YOLOX":
        st.session_state["model_framework"] = "YOLOX"
    else:
        st.session_state["model_framework"] = "FlashDet"


def _tab_arch():
    _sync_framework_from_family()

    st.markdown("#### Framework & architecture")
    framework = st.radio(
        "Framework",
        MODEL_FRAMEWORKS,
        horizontal=True,
        key="model_framework",
        help=(
            "FlashDet: Pico → X with Pico-Zoo backbones, LoRA/QLoRA, fusion and INT8. "
            "Ultralytics: native YOLO training/inference. YOLOX: inference/export via FlashDet."
        ),
    )
    from flashstudio.components.dependencies import render_dependency_hint

    render_dependency_hint(
        framework, "model configuration and training",
        key_suffix=f"model_{framework.lower()}",
    )

    # Head config — num_classes from dataset
    nc = st.session_state.get("num_classes", 0)
    cls_names = st.session_state.get("class_names", "")
    if isinstance(cls_names, list):
        cls_names = "\n".join(cls_names)
    if not nc and cls_names.strip():
        nc = len([c for c in cls_names.strip().split("\n") if c.strip()])
        st.session_state["num_classes"] = nc

    if framework == "FlashDet":
        _set_shared("arch_family", "FlashDet (recommended)")
        _flashdet_arch(nc)
    elif framework == "Ultralytics":
        _ultralytics_arch(nc)
    else:
        _set_shared("arch_family", "YOLOX")
        _yolox_arch(nc)


def _flashdet_arch(nc: int):
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        options = list(FLASHDET_MODELS.keys())
        model = st.selectbox(
            "Size", options,
            index=_index(options, "model_arch", options[0]),
            key="model_arch",
        )
        info = FLASHDET_MODELS[model]

        if model == "FlashDet-Pico":
            current = str(st.session_state.get("pico_backbone") or "LiteBackbone (default)")
            if current == "LiteBackbone":
                current = "LiteBackbone (default)"
            if current not in PICO_BACKBONES:
                current = "LiteBackbone (default)"
            st.session_state["pico_backbone"] = current
            st.selectbox(
                "Pico-Zoo backbone", PICO_BACKBONES,
                key="pico_backbone",
                help="Every Pico-Zoo family has a matching neck; pick LiteBackbone (default) unless you need a specific one.",
            )
            if st.session_state.get("pico_backbone") not in ("LiteBackbone (default)", "PicoBackbone"):
                st.selectbox(
                    "Neck family", PICO_NECKS,
                    index=_index(PICO_NECKS, "neck_type", PICO_NECKS[0]),
                    key="neck_type",
                )
        else:
            st.caption(f"Backbone: {info['backbone']}")

        st.caption(f"{info['neck']} → {info['head']} · CIoU+BCE+L1 · STAL · ProgLoss")

    with col_info:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Params", info["params"])
        with m2:
            st.metric("Speed", info["speed"])
        with m3:
            st.metric("Best For", info["for"])
        with m4:
            st.metric("Classes", nc if nc else "—")
        if nc:
            st.caption(f"Model head output → **{nc}** classes (auto-set from dataset)")
        else:
            st.info("No classes detected yet. Go to **Data** → **Upload** to load a dataset.")
        st.markdown(
            '<div class="ds-check-row">'
            '<span class="ds-check ds-ok">✓ Layer fusion</span>'
            '<span class="ds-check ds-ok">✓ INT8 ONNX</span>'
            '<span class="ds-check ds-ok">✓ LoRA / QLoRA</span>'
            '<span class="ds-check ds-ok">✓ Pico-Zoo</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        _pretrain_finetune("FlashDet")


def _ultralytics_arch(nc: int):
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        series = st.selectbox(
            "Series", YOLO_SERIES,
            index=_index(YOLO_SERIES, "yolo_series", YOLO_SERIES[0]),
            key="yolo_series",
        )
        _set_shared("arch_family", series)

        size = st.selectbox(
            "Size", YOLO_SIZES,
            index=_index(YOLO_SIZES, "yolo_size", YOLO_SIZES[0]),
            key="yolo_size",
        )
        last = st.session_state.get("_last_yolo_size")
        if size != last:
            from flashstudio.utils.weight_hub import YOLO_SIZE_MULT

            width, depth = YOLO_SIZE_MULT.get(size, (0.25, 0.33))
            st.session_state["yolo_width_mult"] = width
            st.session_state["yolo_depth_mult"] = depth
            st.session_state["_last_yolo_size"] = size

        st.session_state.setdefault("yolo_width_mult", 0.50)
        st.slider(
            "Width ×", 0.20, 1.50,
            step=0.05, key="yolo_width_mult",
        )
        st.session_state.setdefault("yolo_depth_mult", 0.33)
        st.slider(
            "Depth ×", 0.10, 1.50,
            step=0.01, key="yolo_depth_mult",
        )
        if series == "YOLOv9":
            st.checkbox("PGI", True, key="yolo_use_pgi")
        elif series == "YOLOv10":
            st.checkbox("PSA", True, key="yolo_use_psa")
        elif series == "YOLOv11":
            st.checkbox("C2PSA", True, key="yolo_use_c2psa")

    with col_info:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Approx params", YOLO_PARAMS.get(series, {}).get(size, "—"))
        with m2:
            st.metric("Classes", nc if nc else "—")
        with m3:
            st.metric("Backend", "Ultralytics")
        st.caption(
            f"{series} · fixed backbone/neck/head. Width/depth follow the size preset "
            "but can be overridden."
        )
        if nc:
            st.caption(f"Head → **{nc}** classes.")
        st.info(
            "For native `ultralytics` training, select **Ultralytics** on "
            "Training → Launch. FlashDet can also build this architecture."
        )
        _pretrain_finetune("Ultralytics", series=series, size=size)


def _yolox_arch(nc: int):
    col_sel, col_info = st.columns([2, 3])
    with col_sel:
        st.slider(
            "Width ×", 0.25, 1.5,
            float(st.session_state.get("yolo_width_mult", 0.5)),
            step=0.05, key="yolo_width_mult",
        )
        st.slider(
            "Depth ×", 0.33, 1.5,
            float(st.session_state.get("yolo_depth_mult", 0.33)),
            step=0.01, key="yolo_depth_mult",
        )

    with col_info:
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Classes", nc if nc else "—")
        with m2:
            st.metric("Backend", "FlashDet")
        st.warning(
            "YOLOX training is not implemented by the installed FlashDet build. "
            "Inference and export work; use FlashDet/YOLO families for training."
        )
        _pretrain_finetune("YOLOX")


def _pretrain_finetune(framework: str, series: str | None = None,
                       size: str | None = None):
    # A download action from a previous run is applied before the radio renders.
    pending = st.session_state.pop("_pending_pretrain_path", None)
    if pending:
        st.session_state["pretrain_option"] = "Custom"
        st.session_state["custom_weights"] = pending

    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Initialization")
            init_options = ["Random", "Custom", "Download"]
            if st.session_state.get("pretrain_option") not in init_options:
                st.session_state["pretrain_option"] = "Random"
            pt = st.radio(
                "Initialization", init_options,
                key="pretrain_option", label_visibility="collapsed",
                horizontal=True,
            )
            if pt == "Custom":
                st.text_input(
                    "Checkpoint path",
                    placeholder="/path/model.pth or /path/model.pt",
                    key="custom_weights",
                )
                path = str(st.session_state.get("custom_weights") or "")
                if path:
                    if os.path.isfile(os.path.abspath(path)):
                        st.success(f"`{os.path.basename(path)}`")
                    else:
                        st.caption(f"Not found: `{path}`")
            elif pt == "Download":
                _download_panel(framework, series=series, size=size)
        with c2:
            st.markdown("#### Fine-tune")
            ft = st.radio("Mode", ["Full fine-tune", "Freeze backbone", "Freeze backbone + neck", "LoRA"],
                          key="finetune_strategy", label_visibility="collapsed")
            st.caption({
                "Full fine-tune": "All layers are trainable.",
                "Freeze backbone": "Backbone frozen; neck and head are trainable.",
                "Freeze backbone + neck": "Backbone and neck frozen; only the head is trainable.",
                "LoRA": "Only low-rank adapters and the head are trainable.",
            }.get(ft, ""))
            if ft == "LoRA":
                lc1, lc2 = st.columns(2)
                with lc1:
                    st.selectbox("Variant", LORA_VARIANTS, key="lora_variant")
                    st.slider("Rank", 4, 64, LORA_RANK_DEFAULT, key="lora_rank")
                with lc2:
                    st.slider("Alpha", 8, 128, int(LORA_ALPHA_DEFAULT), key="lora_alpha")
                    st.slider("Dropout", 0.0, 0.5, LORA_DROPOUT_DEFAULT, key="lora_dropout")
                st.multiselect("Targets", ["backbone", "neck", "head"],
                               default=["backbone", "neck"], key="lora_targets")
                st.checkbox("QLoRA", False, key="qlora")
                if st.session_state.get("qlora"):
                    st.selectbox("Quant", ["int8", "nf4"], key="qlora_dtype")


def _download_panel(framework: str, series: str | None = None,
                    size: str | None = None):
    from flashstudio.utils.weight_hub import (
        download_flashdet, download_ultralytics,
        flashdet_weights, ultralytics_weight_name,
    )

    if framework == "YOLOX":
        st.caption("No pretrained YOLOX registry is bundled. Use Random or Custom.")
        return

    if framework == "FlashDet":
        entries = flashdet_weights()
        if not entries:
            st.caption("FlashDet has no registry checkpoints available in this build.")
            return
        names = [entry["name"] for entry in entries]
        labels = {entry["name"]: entry for entry in entries}
        selected = st.selectbox("Registry checkpoint", names, key="pretrain_fd_name")
        entry = labels[selected]
        st.caption(entry.get("description", ""))
        if entry.get("exists"):
            st.success("Already cached")
        if st.button("Download and use", key="pretrain_fd_download", use_container_width=True):
            try:
                path = download_flashdet(selected)
                st.session_state["_pending_pretrain_path"] = path
                st.rerun()
            except Exception as exc:
                st.error(f"Download failed: {exc}")
        return

    stem = ultralytics_weight_name(series or "YOLOv8", size or "n")
    st.caption(f"Ultralytics `{stem}.pt` (COCO-pretrained head is replaced by your classes).")
    if st.button("Download and use", key="pretrain_ul_download", use_container_width=True):
        try:
            path = download_ultralytics(stem)
            st.session_state["_pending_pretrain_path"] = path
            st.rerun()
        except Exception as exc:
            st.error(f"Download failed: {exc}")
