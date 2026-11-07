"""Inference — Model tab (framework-first)."""

import os

import streamlit as st

from flashstudio.constants import (
    INFER_CONF_THRESHOLD, INFER_NMS_THRESHOLD, INFER_IMG_SIZE,
    DEFAULT_SAVE_DIR,
)
from flashstudio.pages.inference._common import (
    INFER_RUNTIMES, INFER_UPLOAD_TYPES,
    _get_class_names, _get_device_options, latest_inference_weight,
)
from flashstudio.utils import get_state
from flashstudio.utils.model_loader import available_frameworks
from flashstudio.utils.weight_hub import (
    download_flashdet, download_ultralytics, flashdet_weights,
    ultralytics_weights,
)


def _apply_weight_selection():
    """Apply a pending download pick before the weight widgets are created."""
    picked = st.session_state.pop("_picked_weight_path", "")
    if picked:
        st.session_state["infer_weights_path"] = picked
        st.session_state["weight_source"] = "Path"
        st.session_state["infer_framework"] = st.session_state.pop(
            "_picked_weight_framework", "Auto"
        )


def _has_weights(family: str) -> bool:
    if st.session_state.get("infer_demo_mode"):
        return True
    source = st.session_state.get("weight_source")
    if source is None:
        source = (
            "Training output"
            if latest_inference_weight(
                family, get_state("save_dir") or DEFAULT_SAVE_DIR
            )
            else "Upload"
        )
    if source == "Upload":
        return st.session_state.get("infer_weights_file") is not None
    if source == "Path":
        path = os.path.abspath(st.session_state.get("infer_weights_path", "") or "")
        return bool(path) and os.path.isfile(path)
    sd = get_state("save_dir") or DEFAULT_SAVE_DIR
    return bool(latest_inference_weight(family, sd))


def _render_weight_downloader(family: str):
    with st.expander(f"Download {family} weights", expanded=not _has_weights(family)):
        if family == "FlashDet":
            entries = flashdet_weights()
            if not entries:
                st.caption("No FlashDet registry checkpoints are available in this build.")
                return
            labels = {
                entry["name"]: (
                    f"{entry['name']} · {entry.get('num_classes') or '?'} classes"
                    + (" · cached" if entry.get("exists") else "")
                )
                for entry in entries
            }
            name = st.selectbox("Checkpoint", list(labels), format_func=labels.get,
                                key="dl_fd_name")
            entry = next(item for item in entries if item["name"] == name)
            st.caption(entry.get("description", ""))
            if st.button(
                "Use cached" if entry.get("exists") else "Download",
                key="dl_fd_btn", type="primary", use_container_width=True,
            ):
                try:
                    path = download_flashdet(name)
                    st.session_state["_picked_weight_path"] = path
                    st.session_state["_picked_weight_framework"] = "FlashDet"
                    has_input = bool(
                        st.session_state.get("infer_images")
                        or st.session_state.get("infer_video")
                        or st.session_state.get("infer_stream_url")
                    )
                    st.session_state["_infer_tab_next"] = (
                        "Run" if has_input else "Data"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Download failed: {exc}")
            st.caption(
                "FlashDet checkpoints expect the class count they were trained with."
            )
            return

        if family == "Ultralytics":
            entries = ultralytics_weights()
            labels = {
                entry["name"]: (
                    f"{entry['label']} · {entry['name']}"
                    + (" · cached" if entry.get("exists") else "")
                )
                for entry in entries
            }
            name = st.selectbox("Checkpoint", list(labels), format_func=labels.get,
                                key="dl_ul_name")
            cached = next(entry["exists"] for entry in entries if entry["name"] == name)
            if st.button(
                "Use cached" if cached else "Download",
                key="dl_ul_btn", type="primary", use_container_width=True,
            ):
                try:
                    path = download_ultralytics(name)
                    st.session_state["_picked_weight_path"] = path
                    st.session_state["_picked_weight_framework"] = "Ultralytics"
                    has_input = bool(
                        st.session_state.get("infer_images")
                        or st.session_state.get("infer_video")
                        or st.session_state.get("infer_stream_url")
                    )
                    st.session_state["_infer_tab_next"] = (
                        "Run" if has_input else "Data"
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Download failed: {exc}")
            st.caption("YOLO checkpoints include COCO class names.")
            return

        st.caption(
            "No YOLOX registry checkpoint is bundled. Import a YOLOX `.pth` with "
            "Source → Path, or train one with a FlashDet build that supports it."
        )


def _tab_model(family: str = "FlashDet"):
    _apply_weight_selection()
    if not _has_weights(family):
        st.info(
            "No weights selected for this framework. Download a starter checkpoint "
            "below, upload a file, or point at a local checkpoint."
        )
    if family == "YOLOX":
        st.caption(
            "YOLOX uses the FlashDet runtime. Training is not implemented by the "
            "installed FlashDet build; inference works."
        )

    devices = _get_device_options()
    if st.session_state.get("infer_device") not in devices:
        st.session_state.pop("infer_device", None)
    for key, low, high in (
        ("infer_conf", 0.0, 1.0),
        ("infer_nms", 0.0, 1.0),
    ):
        try:
            value = float(st.session_state.get(key))
            if value < low or value > high:
                raise ValueError
        except (TypeError, ValueError):
            st.session_state.pop(key, None)
    try:
        size = int(st.session_state.get("infer_img_size"))
        if size < 320 or size > 1920:
            raise ValueError
    except (TypeError, ValueError):
        st.session_state.pop("infer_img_size", None)
    filter_classes = _get_class_names()
    selected_filter = st.session_state.get("infer_class_filter")
    if selected_filter and any(name not in filter_classes for name in selected_filter):
        st.session_state.pop("infer_class_filter", None)

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Parameters")
            pc1, pc2 = st.columns(2)
            with pc1:
                st.slider("Confidence", 0.0, 1.0, INFER_CONF_THRESHOLD, 0.05,
                          key="infer_conf")
                st.number_input("Image size", 320, 1920, INFER_IMG_SIZE, 32,
                                key="infer_img_size")
            with pc2:
                st.slider("NMS IoU", 0.0, 1.0, INFER_NMS_THRESHOLD, 0.05,
                          key="infer_nms")
                st.selectbox("Device", _get_device_options(), key="infer_device")
            st.multiselect("Class filter (empty = all)", filter_classes, default=[],
                           key="infer_class_filter")
            st.checkbox(
                "Demo mode (no model)", value=False, key="infer_demo_mode",
                help="Draw plausible fake detections for UI demos. Never enabled automatically.",
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### Weights")
            runtimes = INFER_RUNTIMES.get(family, ["Auto"])
            if st.session_state.get("infer_framework") not in runtimes:
                st.session_state["infer_framework"] = "Auto"
            st.selectbox(
                "Runtime", runtimes, key="infer_framework",
                help=f"Runtime options for {family} checkpoints.",
            )
            avail = available_frameworks()
            st.caption(
                " · ".join(
                    f"{'✓' if item['installed'] else '✗'} {item['label']}"
                    for item in avail
                )
            )
            runtime = st.session_state.get("infer_framework", "Auto")
            if runtime in ("ONNX Runtime", "TensorRT", "TorchScript"):
                from flashstudio.components.dependencies import render_dependency_hint

                render_dependency_hint(
                    runtime, "the selected inference runtime",
                    key_suffix=f"runtime_{runtime.lower()}_{family.lower()}",
                )
            source_index = 0
            if "weight_source" not in st.session_state:
                sd = get_state("save_dir") or DEFAULT_SAVE_DIR
                if latest_inference_weight(family, sd):
                    source_index = 2
            ws = st.radio(
                "Source", ["Upload", "Path", "Training output"],
                index=source_index, key="weight_source", horizontal=True,
            )
            if ws == "Upload":
                up = st.file_uploader(
                    "File", type=INFER_UPLOAD_TYPES.get(family, ["pt", "pth"]),
                    key="infer_weights_file",
                )
                if up:
                    st.success(f"{up.name} ({up.size / 1e6:.1f}MB)")
            elif ws == "Path":
                placeholder = (
                    "/path/best.pt" if family == "Ultralytics" else "/path/best.pth"
                )
                st.text_input("Path", placeholder=placeholder, key="infer_weights_path")
                path = os.path.abspath(st.session_state.get("infer_weights_path", "") or "")
                if path:
                    if os.path.isfile(path):
                        st.success(f"`{os.path.basename(path)}`")
                    else:
                        st.caption(f"Not found: `{path}`")
            else:
                sd = get_state("save_dir") or DEFAULT_SAVE_DIR
                best_path = latest_inference_weight(family, sd)
                st.session_state["infer_weights_path"] = best_path
                if best_path and os.path.isfile(best_path):
                    sz = os.path.getsize(best_path) / (1024 * 1024)
                    st.success(f"`{os.path.basename(best_path)}` ({sz:.1f}MB)")
                else:
                    st.caption(f"No {family} training output found in `{sd}`")

    _render_weight_downloader(family)
