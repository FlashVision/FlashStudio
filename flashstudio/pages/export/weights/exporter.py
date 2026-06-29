"""FlashStudio — Export engine (model export to ONNX / TorchScript)."""

import os
import streamlit as st
from flashstudio.constants import (
    INFER_NUM_CLASSES, MODEL_SIZE_DEFAULT,
)


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
