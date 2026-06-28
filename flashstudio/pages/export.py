"""FlashStudio — Model Export Page (matches actual FlashDet capabilities)."""

import os
import streamlit as st


def render_export_page():
    """Render model export/conversion page."""
    from flashstudio.components.styles import render_page_header
    render_page_header("📤", "Export Model",
                       "Convert your trained model for production deployment.")

    _render_training_summary()
    st.divider()

    col_config, col_output = st.columns([1, 1])

    with col_config:
        _render_export_config()

    with col_output:
        _render_export_output()


def _render_training_summary():
    """Show what weights are already available from training."""
    save_dir = st.session_state.get("save_dir", "")

    if save_dir and os.path.isdir(save_dir):
        st.markdown("#### Available Weights from Training")
        weights_files = [f for f in os.listdir(save_dir) if f.endswith(".pth")]
        if weights_files:
            cols = st.columns(min(len(weights_files), 4))
            for i, wf in enumerate(sorted(weights_files)[:4]):
                fpath = os.path.join(save_dir, wf)
                size_mb = os.path.getsize(fpath) / (1024 * 1024)
                with cols[i]:
                    label = "Best" if "best" in wf else "Last" if "last" in wf else "Other"
                    precision = "FP16" if "fp16" in wf else "FP32"
                    st.metric(f"{label} ({precision})", f"{size_mb:.1f} MB")
            st.caption(f"Directory: `{save_dir}`")
        else:
            st.info("No weights found in save directory.")
    else:
        st.info("Complete training first, or provide a weights path below.")


def _render_export_config():
    """Export configuration — only options FlashDet actually supports."""
    st.markdown("### ⚙️ Export Settings")

    with st.container(border=True):
        st.markdown("**Model Weights**")
        weights_source = st.radio(
            "Source",
            ["Best model (inference weights)", "Best model (FP16)",
             "Last model (inference weights)", "Custom path"],
            key="weights_source",
        )
        if weights_source == "Custom path":
            st.text_input("Weights path (.pth)", placeholder="/path/to/model.pth",
                          key="export_weights_path")

        save_dir = st.session_state.get("save_dir", os.path.join(os.getcwd(), "flashstudio_runs"))
        weights_map = {
            "Best model (inference weights)": "model_best_inference.pth",
            "Best model (FP16)": "model_best_fp16.pth",
            "Last model (inference weights)": "model_last_inference.pth",
        }
        if weights_source in weights_map:
            weights_path = os.path.join(save_dir, weights_map[weights_source])
        else:
            weights_path = st.session_state.get("export_weights_path", "")

        st.caption(f"📁 `{weights_path}`")

    with st.container(border=True):
        st.markdown("**Export Format**")
        st.markdown("FlashDet supports **ONNX** export with dynamic batch size (opset 13).")

        st.checkbox("ONNX (recommended — universal format)", value=True, key="export_onnx",
                    disabled=True)
        st.caption("Works with: ONNX Runtime, OpenVINO, TensorRT (via onnx2trt), TFLite (via onnx2tf)")

        st.divider()
        st.caption("**Not natively supported** (convert from ONNX using external tools):")
        st.markdown("""
        - TensorRT: `trtexec --onnx=model.onnx --saveEngine=model.engine --fp16`
        - OpenVINO: `mo --input_model model.onnx --output_dir openvino/`
        - CoreML: `python -m coremltools.converters.onnx model.onnx`
        - NCNN: `onnx2ncnn model.onnx model.param model.bin`
        """)

    with st.container(border=True):
        st.markdown("**ONNX Export Options**")
        img_size = st.select_slider(
            "Input Size", [320, 416, 640], value=320, key="export_img_size",
            help="Should match your training input size"
        )
        st.checkbox("Dynamic batch size", value=True, key="export_dynamic",
                    help="Allows variable batch size at inference (default in FlashDet)")
        st.number_input("Opset Version", 11, 18, 13, key="export_opset",
                        help="FlashDet uses opset 13 by default")

    if st.button("🚀 Export to ONNX", use_container_width=True, type="primary"):
        opset = st.session_state.get("export_opset", 13)
        dynamic = st.session_state.get("export_dynamic", True)
        _run_export(weights_path, img_size, opset, dynamic)


def _render_export_output():
    """Show export results and commands."""
    st.markdown("### 📦 Export Output")

    exported = st.session_state.get("exported_files", [])

    if not exported:
        st.markdown("#### Pre-Generated Weights")
        st.info(
            "FlashDet **automatically saves** FP16 inference weights during training:\n\n"
            "- `model_best_inference.pth` — best mAP, FP32, o2m head stripped\n"
            "- `model_best_fp16.pth` — best mAP, FP16 (half memory)\n"
            "- `model_last_inference.pth` — final epoch, FP32\n"
            "- `model_last_fp16.pth` — final epoch, FP16\n\n"
            "These are ready to use with `flashdet.Predictor` — no export step needed for PyTorch inference."
        )

        st.divider()
        st.markdown("#### ONNX Export Command (Manual)")
        weights = st.session_state.get("export_weights_path", "model_best_inference.pth")
        st.code(
            f"flashdet export --model {weights} --output model.onnx\n"
            f"# Output: model.onnx (opset 13, dynamic batch)",
            language="bash"
        )
        return

    for exp in exported:
        with st.container(border=True):
            col_info, col_action = st.columns([3, 1])
            with col_info:
                st.markdown(f"**{exp['format']}** — `{exp['path']}`")
                if exp.get("size"):
                    st.caption(f"Size: {exp['size']}")
                if exp.get("success"):
                    st.success("Export successful")
                else:
                    st.error("Export failed")
            with col_action:
                if exp.get("success") and os.path.isfile(exp.get("path", "")):
                    with open(exp["path"], "rb") as f:
                        st.download_button("📥 Download", f, file_name=os.path.basename(exp["path"]),
                                           key=f"dl_{exp['format']}")

    st.divider()
    st.success("✅ Export complete! Click **Next** to test inference.")


def _run_export(weights_path: str, img_size: int, opset: int = 13, dynamic: bool = True):
    """Run ONNX export using flashdet."""
    if not weights_path:
        st.error("Please select or provide weights path.")
        return

    if not os.path.isfile(weights_path):
        st.warning(f"Weights file not found: `{weights_path}`. Cannot export.")
        return

    _flashdet_export(weights_path, img_size, opset, dynamic)


def _flashdet_export(weights_path: str, img_size: int, opset: int = 13, dynamic: bool = True):
    """ONNX export using flashdet's model loading + torch.onnx.export."""
    import torch
    from flashdet import get_config, build_model

    output_path = weights_path.replace(".pth", ".onnx")

    with st.spinner("Exporting to ONNX via flashdet..."):
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

            ckpt_config = checkpoint.get("config", {})
            architecture = ckpt_config.get("architecture", "flashdet")
            num_classes = ckpt_config.get("num_classes", 80)
            model_size = ckpt_config.get("model_size", "n")
            ckpt_input_size = ckpt_config.get("input_size", img_size)

            config = get_config(
                model_size=model_size,
                input_size=ckpt_input_size,
                num_classes=num_classes,
            )
            model = build_model(config, architecture=architecture)

            state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
            if isinstance(state_dict, dict) and "model_state_dict" not in checkpoint and "model" not in checkpoint:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            dummy_input = torch.randn(1, 3, ckpt_input_size, ckpt_input_size)

            dynamic_axes = None
            if dynamic:
                dynamic_axes = {"images": {0: "batch"}, "output": {0: "batch"}}

            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=opset,
                input_names=["images"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )

            size_str = "—"
            if os.path.isfile(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                size_str = f"{size_mb:.1f} MB"

            st.session_state["exported_files"] = [{
                "format": "ONNX", "path": output_path, "size": size_str, "success": True
            }]
            st.toast("ONNX export complete!", icon="✅")
        except Exception as e:
            st.error(f"Export failed: {e}")
            st.session_state["exported_files"] = [{
                "format": "ONNX", "path": output_path, "size": "—", "success": False
            }]

    st.rerun()
