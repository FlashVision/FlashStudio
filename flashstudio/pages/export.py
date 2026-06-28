"""FlashStudio — Export (zero-scroll, split layout)."""

import os
import streamlit as st


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
                    cols = st.columns(min(len(wts), 4))
                    for i, wp_disp in enumerate(sorted(wts)[:4]):
                        sz = os.path.getsize(wp_disp) / (1024 * 1024)
                        bn = os.path.basename(wp_disp)
                        lbl = "Best" if "best" in bn else ("Last" if "last" in bn else bn[:6])
                        with cols[i]:
                            st.metric(lbl, f"{sz:.1f}MB")

            st.markdown("#### Config")
            src = st.radio("Source", ["Best (inference)", "Best (FP16)", "Last", "Custom"],
                           key="weights_source", horizontal=True)
            if src == "Custom":
                st.text_input("Path", placeholder="model.pth", key="export_weights_path")

            sd = st.session_state.get("save_dir", os.path.join(os.getcwd(), "workspace"))
            wmap = {"Best (inference)": "checkpoint_best.pth", "Best (FP16)": "model_best_fp16.pth", "Last": "checkpoint_last.pth"}
            if src != "Custom":
                target = wmap.get(src, "")
                wp = ""
                if os.path.isdir(sd):
                    for root, _dirs, files in os.walk(sd):
                        if target in files:
                            wp = os.path.join(root, target)
                            break
                if not wp:
                    wp = os.path.join(sd, target)
            else:
                wp = st.session_state.get("export_weights_path", "")

            oc = st.columns(3)
            with oc[0]:
                img_sz = st.select_slider("Img", [320, 416, 640], value=320, key="export_img_size")
            with oc[1]:
                st.number_input("Opset", 11, 18, 13, key="export_opset")
            with oc[2]:
                st.checkbox("Dynamic", True, key="export_dynamic")

            if st.button("Export ONNX", use_container_width=True, type="primary"):
                _run_export(wp, img_sz, st.session_state.get("export_opset", 13), st.session_state.get("export_dynamic", True))

    with right:
        with st.container(border=True):
            st.markdown("#### Output")
            exported = st.session_state.get("exported_files", [])
            if not exported:
                st.caption("Auto-saved weights from training:")
                for n, d in [("checkpoint_best.pth", "Best, FP32"), ("model_best_fp16.pth", "Best, FP16"),
                              ("checkpoint_last.pth", "Final, FP32")]:
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


def _run_export(weights_path, img_size, opset=13, dynamic=True):
    from flashstudio.utils import flash

    if not weights_path or not os.path.isfile(weights_path):
        flash(f"Weights not found: `{weights_path}`", "error")
        st.rerun()
        return

    import torch
    from flashdet import get_config, build_model

    out = weights_path.replace(".pth", ".onnx")
    with st.spinner("Exporting..."):
        try:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {})
            config = get_config(model_size=cfg.get("model_size", "n"), input_size=cfg.get("input_size", img_size), num_classes=cfg.get("num_classes", 80))
            model = build_model(config, architecture=cfg.get("architecture", "flashdet"))
            sd = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
            model.load_state_dict(sd, strict=False)
            model.eval()
            dummy = torch.randn(1, 3, config.input_size, config.input_size)
            dyn = {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
            torch.onnx.export(model, dummy, out, opset_version=opset, input_names=["images"], output_names=["output"], dynamic_axes=dyn)
            sz = f"{os.path.getsize(out) / (1024*1024):.1f}MB" if os.path.isfile(out) else "—"
            st.session_state["exported_files"] = [{"format": "ONNX", "path": out, "size": sz, "success": True}]
            flash(f"Export successful: `{os.path.basename(out)}` ({sz})", "success")
        except Exception as e:
            flash(f"Export failed: {e}", "error")
            st.session_state["exported_files"] = [{"format": "ONNX", "path": out, "size": "—", "success": False}]
    st.rerun()
