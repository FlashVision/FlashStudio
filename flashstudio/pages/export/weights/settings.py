"""Export settings UI — one shared structure for every framework."""

import os

import streamlit as st

from flashstudio.constants import (
    DEFAULT_SAVE_DIR, EXPORT_OPSET_MIN, EXPORT_OPSET_MAX,
    EXPORT_OPSET_DEFAULT, EXPORT_FORMATS,
)
from flashstudio.pages.export._common import weights_with_mtime


def _get_export_save_dir():
    """Resolve save_dir: _config_mirror > session_state > DEFAULT_SAVE_DIR."""
    from flashstudio.utils import get_state

    sd = get_state("save_dir")
    return sd if sd else DEFAULT_SAVE_DIR


def _weight_files(root: str) -> list:
    """Compatibility helper: return ``(path, mtime)`` for supported weights."""
    found = []
    if not os.path.isdir(root):
        return found
    for dirpath, dirs, files in os.walk(root):
        if (
            os.path.basename(dirpath) == "weights"
            and os.path.basename(os.path.dirname(dirpath)) == "ultralytics"
        ):
            continue
        dirs[:] = [d for d in dirs if d not in ("yolo_dataset", "optimized")]
        for f in files:
            if f.endswith((".pth", ".pt")):
                path = os.path.join(dirpath, f)
                try:
                    found.append((path, os.path.getmtime(path)))
                except OSError:
                    pass
    found.sort(key=lambda item: item[1], reverse=True)
    return found


def _render_downloader(framework: str, key_prefix: str = "convert"):
    from flashstudio.utils.weight_hub import (
        download_flashdet, download_ultralytics, flashdet_weights,
        ultralytics_weight_name,
    )

    suffix = f"{key_prefix}_{framework.lower()}"
    st.caption("No matching checkpoint in this workspace. Download a starter weight.")
    if framework == "Ultralytics":
        from flashstudio.constants import YOLO_SERIES, YOLO_SIZES

        c1, c2 = st.columns(2)
        with c1:
            series = st.selectbox("Series", YOLO_SERIES, key=f"export_dl_series_{suffix}")
        with c2:
            size = st.selectbox("Size", YOLO_SIZES, key=f"export_dl_size_{suffix}")
        stem = ultralytics_weight_name(series, size)
        if st.button(f"Download {stem}.pt", key=f"export_dl_ul_{suffix}",
                     use_container_width=True):
            try:
                path = download_ultralytics(stem)
                st.session_state[f"_export_pending_weight_{suffix}"] = path
                st.rerun()
            except Exception as exc:
                st.error(f"Download failed: {exc}")
        return

    if framework == "YOLOX":
        st.caption("No YOLOX registry checkpoint is bundled. Import a `.pth` or train one.")
        return

    entries = flashdet_weights()
    if not entries:
        st.caption("No FlashDet registry checkpoint is available in this build.")
        return
    labels = {
        entry["name"]: (
            f"{entry['name']} · {entry.get('num_classes') or '?'} classes"
            + (" · cached" if entry.get("exists") else "")
        )
        for entry in entries
    }
    name = st.selectbox("Registry checkpoint", list(labels),
                        format_func=labels.get, key=f"export_dl_fd_name_{suffix}")
    if st.button("Download and use", key=f"export_dl_fd_{suffix}",
                 use_container_width=True):
        try:
            path = download_flashdet(name)
            st.session_state[f"_export_pending_weight_{suffix}"] = path
            st.rerun()
        except Exception as exc:
            st.error(f"Download failed: {exc}")


def render_checkpoint_card(framework: str, key_prefix: str = "convert"):
    """Render the shared checkpoint card. Returns ``(path, input_size)``."""
    pending_key = f"_export_pending_weight_{key_prefix}_{framework.lower()}"
    pending = st.session_state.pop(pending_key, None)
    if pending:
        custom_key = f"_export_custom_{key_prefix}_{framework.lower()}"
        widget_key = f"export_custom_path_{key_prefix}_{framework.lower()}"
        st.session_state[custom_key] = pending
        st.session_state[widget_key] = pending

    weights = [path for path, _mtime in weights_with_mtime(framework, _get_export_save_dir())]

    with st.container(border=True):
        st.markdown(f"#### {framework} checkpoint")
        if not weights:
            _render_downloader(framework, key_prefix)
            return "", 640 if framework == "Ultralytics" else 320

        from flashstudio.utils.filesystem import weight_label

        labels = {
            path: f"{weight_label(path)} · {os.path.basename(path)}"
            for path in weights
        }
        select_key = f"export_ckpt_{key_prefix}_{framework.lower()}"
        size_key = f"export_size_{key_prefix}_{framework.lower()}"
        c1, c2 = st.columns([3, 1])
        with c1:
            selected = st.selectbox("Checkpoint", weights, format_func=labels.get,
                                    key=select_key)
        with c2:
            default_size = 128 if "demo" in os.path.basename(selected).lower() else (
                640 if framework == "Ultralytics" else 320
            )
            input_size = st.number_input(
                "Input size", 64, 1920, default_size, 32, key=size_key,
            )
        try:
            sz = f"{os.path.getsize(selected) / (1024 * 1024):.1f} MB"
        except OSError:
            sz = "—"
        st.caption(f"`{selected}` · {sz}")

        custom = st.text_input(
            "Or custom path",
            value=str(st.session_state.get(f"_export_custom_{key_prefix}_{framework.lower()}", "")),
            key=f"export_custom_path_{key_prefix}_{framework.lower()}",
            placeholder="/path/to/model.pth",
        )
        if custom:
            custom_path = os.path.abspath(custom)
            if os.path.isfile(custom_path):
                selected = custom_path
                st.caption(f"Using custom checkpoint · `{os.path.basename(custom_path)}`")
            else:
                st.caption("Custom checkpoint not found.")
    return selected, int(input_size)


def render_export_settings(framework: str = "FlashDet", key_prefix: str = "convert"):
    """Render one checkpoint card + one options/action card.

    Returns ``(weights_path, img_size, export_fmt)``.
    """
    wp, img_sz = render_checkpoint_card(framework, key_prefix)

    formats = (
        ["ONNX", "TorchScript", "TensorRT engine"]
        if framework == "Ultralytics" else list(EXPORT_FORMATS)
    )
    fmt_key = f"export_format_{key_prefix}_{framework.lower()}"
    if st.session_state.get(fmt_key) not in formats:
        st.session_state[fmt_key] = formats[0]

    with st.container(border=True):
        st.markdown("#### Export options")
        c1, c2, c3 = st.columns(3)
        with c1:
            export_fmt = st.selectbox("Format", formats, key=fmt_key)
        with c2:
            current_opset = st.session_state.get("export_opset", EXPORT_OPSET_DEFAULT)
            if (
                not isinstance(current_opset, int)
                or current_opset < EXPORT_OPSET_MIN
                or current_opset > EXPORT_OPSET_MAX
            ):
                st.session_state["export_opset"] = EXPORT_OPSET_DEFAULT
            st.number_input("Opset", EXPORT_OPSET_MIN, EXPORT_OPSET_MAX,
                            EXPORT_OPSET_DEFAULT, key="export_opset")
        with c3:
            st.checkbox("Dynamic batch", True, key="export_dynamic")

        if framework == "Ultralytics":
            e1, e2, e3 = st.columns(3)
            with e1:
                st.checkbox("Half (FP16)", False, key="export_half")
            with e2:
                st.checkbox("INT8", False, key="export_int8",
                            help="Requires a calibration dataset YAML.")
            with e3:
                st.checkbox("Simplify", False, key="export_simplify")
            if st.session_state.get("export_int8"):
                st.text_input(
                    "Calibration data.yaml",
                    value=str(st.session_state.get("export_calib_data") or ""),
                    key="export_calib_data",
                    placeholder="path/to/data.yaml",
                )

        st.caption(
            "The checkpoint and input size come from the card above. "
            "Framework-specific options appear only when relevant."
        )
        if not wp:
            st.caption("Select or download a checkpoint first.")
        elif st.button(f"Export {export_fmt}", use_container_width=True,
                       type="primary",
                       key=f"export_run_{key_prefix}_{framework.lower()}"):
            from flashstudio.pages.export.weights.exporter import _run_export

            _run_export(
                wp, img_sz,
                st.session_state.get("export_opset", EXPORT_OPSET_DEFAULT),
                st.session_state.get("export_dynamic", True),
                export_fmt.lower(),
            )
    return wp, img_sz, export_fmt
