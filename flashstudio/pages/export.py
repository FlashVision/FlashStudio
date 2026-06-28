"""FlashStudio — Export (zero-scroll, split layout)."""

import os
import streamlit as st
from flashstudio.constants import (
    DEFAULT_SAVE_DIR, EXPORT_IMG_SIZES, EXPORT_OPSET_MIN,
    EXPORT_OPSET_MAX, EXPORT_OPSET_DEFAULT, EXPORT_FORMATS,
    EXPORT_WEIGHT_MAP, CKPT_BEST, INFER_NUM_CLASSES,
    MAX_WEIGHTS_DISPLAY, MODEL_SIZE_DEFAULT,
)


def render_export_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Export")
    show_flashes()

    # Two-panel: Config | Output
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("#### Weights")
            save_dir = st.session_state.get("save_dir", "")
            if save_dir and os.path.isdir(save_dir):
                wts = []
                for root, _dirs, files in os.walk(save_dir):
                    for f in files:
                        if f.endswith(".pth"):
                            wts.append(os.path.join(root, f))
                if wts:
                    cols = st.columns(min(len(wts), MAX_WEIGHTS_DISPLAY))
                    for i, wp_disp in enumerate(sorted(wts)[:MAX_WEIGHTS_DISPLAY]):
                        sz = os.path.getsize(wp_disp) / (1024 * 1024)
                        bn = os.path.basename(wp_disp)
                        lbl = "Best" if "best" in bn else ("Last" if "last" in bn else bn[:6])
                        with cols[i]:
                            st.metric(lbl, f"{sz:.1f}MB")
                else:
                    st.caption(f"No weights found in `{save_dir}`")
            else:
                st.caption("No save directory set. Train a model first or set `Save Dir` on Model page.")

            st.markdown("#### Config")
            src = st.radio("Source", ["Best (inference)", "Best (FP16)", "Last", "Custom"],
                           key="weights_source", horizontal=True)
            if src == "Custom":
                st.text_input("Path", placeholder="model.pth", key="export_weights_path")

            sd = st.session_state.get("save_dir", DEFAULT_SAVE_DIR)
            if src != "Custom":
                targets = EXPORT_WEIGHT_MAP.get(src, [CKPT_BEST])
                wp = ""
                if os.path.isdir(sd):
                    for root, _dirs, files in os.walk(sd):
                        for target in targets:
                            if target in files:
                                wp = os.path.join(root, target)
                                break
                        if wp:
                            break
                if not wp:
                    wp = os.path.join(sd, targets[0])
            else:
                wp = st.session_state.get("export_weights_path", "")

            oc = st.columns(4)
            with oc[0]:
                export_fmt = st.selectbox("Format", EXPORT_FORMATS, key="export_format")
            with oc[1]:
                img_sz = st.select_slider("Img", EXPORT_IMG_SIZES, value=EXPORT_IMG_SIZES[0], key="export_img_size")
            with oc[2]:
                st.number_input("Opset", EXPORT_OPSET_MIN, EXPORT_OPSET_MAX, EXPORT_OPSET_DEFAULT, key="export_opset")
            with oc[3]:
                st.checkbox("Dynamic", True, key="export_dynamic")

            if st.button(f"Export {export_fmt}", use_container_width=True, type="primary"):
                _run_export(wp, img_sz, st.session_state.get("export_opset", EXPORT_OPSET_DEFAULT),
                            st.session_state.get("export_dynamic", True),
                            export_fmt.lower())

    with right:
        with st.container(border=True):
            st.markdown("#### Output")
            exported = st.session_state.get("exported_files", [])
            if not exported:
                st.caption("Auto-saved weights from training:")
                for n, d in [("model_best_inference.pth", "Best, inference-ready"),
                              ("checkpoint_best.pth", "Best, full checkpoint"),
                              ("model_last_inference.pth", "Final, inference-ready"),
                              ("checkpoint_last.pth", "Final, full checkpoint")]:
                    st.markdown(f'<span style="font-size:0.82rem;color:#4B5563;">• `{n}` — {d}</span>', unsafe_allow_html=True)
                st.caption("Manual:")
                st.code("flashdet export --model model.pth --output model.onnx", language="bash")
            else:
                for exp in exported:
                    st.markdown(f"**{exp['format']}** · `{os.path.basename(exp.get('path', ''))}`")
                    if exp.get("size"):
                        st.caption(exp["size"])
                    st.success("OK") if exp.get("success") else st.error("Failed")
                    if exp.get("success") and os.path.isfile(exp.get("path", "")):
                        with open(exp["path"], "rb") as f:
                            st.download_button("Download", f, file_name=os.path.basename(exp["path"]),
                                               key=f"dl_{exp['format']}", use_container_width=True)
                with st.expander("Post-export"):
                    st.code("trtexec --onnx=model.onnx --saveEngine=model.engine --fp16", language="bash")


def _run_export(weights_path, img_size, opset=13, dynamic=True, fmt="onnx"):
    from flashstudio.utils import flash

    if not weights_path or not os.path.isfile(weights_path):
        flash(f"Weights not found: `{weights_path}`", "error")
        st.rerun()
        return

    import torch
    try:
        from flashdet import get_config, build_model
    except ImportError:
        flash("FlashDet not installed — run `pip install flashdet`", "error")
        st.rerun()
        return

    ext = ".onnx" if fmt == "onnx" else ".pt"
    out = weights_path.replace(".pth", ext)
    label = "ONNX" if fmt == "onnx" else "TorchScript"
    with st.spinner(f"Exporting to {label}..."):
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            nc = cfg.get("num_classes", st.session_state.get("num_classes", INFER_NUM_CLASSES))
            raw_input = cfg.get("input_size", img_size)
            inp_scalar = raw_input[0] if isinstance(raw_input, (tuple, list)) else raw_input
            arch = cfg.get("architecture", "flashdet")
            config = get_config(model_size=cfg.get("model_size", MODEL_SIZE_DEFAULT), input_size=inp_scalar, num_classes=nc)
            config.model.architecture = arch
            model = build_model(config, architecture=arch)
            sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt)))
            model.load_state_dict(sd, strict=False)
            model.eval()
            dummy = torch.randn(1, 3, inp_scalar, inp_scalar)

            if fmt == "onnx":
                dyn = {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
                torch.onnx.export(model, dummy, out, opset_version=opset,
                                  input_names=["images"], output_names=["output"],
                                  dynamic_axes=dyn)
            else:
                traced = torch.jit.trace(model, dummy)
                traced.save(out)

            sz = f"{os.path.getsize(out) / (1024*1024):.1f}MB" if os.path.isfile(out) else "—"
            st.session_state["exported_files"] = [{"format": label, "path": out, "size": sz, "success": True}]
            flash(f"Export successful: `{os.path.basename(out)}` ({sz})", "success")
        except Exception as e:
            flash(f"Export failed: {e}", "error")
            st.session_state["exported_files"] = [{"format": label, "path": out, "size": "—", "success": False}]
    st.rerun()
