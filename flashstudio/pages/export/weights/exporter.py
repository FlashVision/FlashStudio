"""FlashStudio — Export engine (model export to ONNX / TorchScript)."""

import os
import json
import streamlit as st
from flashstudio.constants import (
    INFER_NUM_CLASSES, MODEL_SIZE_DEFAULT,
)


class _TorchScriptWrapper:
    """Wrap FlashDet's dict-returning forward so ``torch.jit.trace`` can work."""

    def __new__(cls, model):
        import torch.nn as nn

        class Wrapper(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                out = self.inner(x)
                if isinstance(out, dict):
                    if "preds" in out:
                        return out["preds"]
                    for value in out.values():
                        if hasattr(value, "shape"):
                            return value
                return out

        return Wrapper(model)


def _build_export_model(weights_path, img_size):
    """Load a checkpoint and rebuild the exact architecture that produced it."""
    import torch
    from flashdet.cfg import get_config
    from flashdet.models.detector import build_model

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg_data = ckpt.get("config", {}) or {}
    if not isinstance(cfg_data, dict):
        cfg_data = {}

    # Trainer checkpoints do not record YOLO width/depth multipliers.  The
    # manifest written next to them does, so use it as a fallback.
    manifest = {}
    try:
        manifest_path = os.path.join(os.path.dirname(weights_path), "train_manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if not isinstance(manifest, dict):
                manifest = {}
    except (OSError, ValueError):
        manifest = {}

    def _cfg(key, default=None):
        return cfg_data.get(key, manifest.get(key, default))

    nc = int(_cfg("num_classes", INFER_NUM_CLASSES) or INFER_NUM_CLASSES)
    raw_input = _cfg("input_size", img_size)
    inp_scalar = int(raw_input[0]) if isinstance(raw_input, (tuple, list)) else int(raw_input or img_size)
    arch = str(_cfg("architecture", "flashdet") or "flashdet").lower()
    model_size = str(_cfg("model_size", MODEL_SIZE_DEFAULT) or MODEL_SIZE_DEFAULT)

    config = get_config(model_size=model_size, input_size=inp_scalar, num_classes=nc)
    config.model.architecture = arch
    config.model.size = model_size
    config.model.input_size = (inp_scalar, inp_scalar)
    config.model.num_classes = nc

    if arch in ("yolov8", "yolov9", "yolov10", "yolov11", "yolox"):
        config.model.width_mult = float(_cfg("yolo_width_mult", _cfg("width_mult", 1.0)) or 1.0)
        config.model.depth_mult = float(_cfg("yolo_depth_mult", _cfg("depth_mult", 1.0)) or 1.0)
        config.model.use_pgi = bool(_cfg("yolo_use_pgi", _cfg("use_pgi", True)))
        config.model.use_psa = bool(_cfg("yolo_use_psa", _cfg("use_psa", True)))
        config.model.use_c2psa = bool(_cfg("yolo_use_c2psa", _cfg("use_c2psa", True)))
    else:
        backbone_type = _cfg("backbone_type") or "lite"
        config.model.backbone_type = backbone_type
        config.model.neck_type = _cfg("neck_type")

    model = build_model(config, architecture=arch)
    state_dict = ckpt.get(
        "model_state_dict", ckpt.get("state_dict", ckpt.get("model", ckpt))
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, inp_scalar, arch


def _run_ultralytics_export(weights_path, img_size, opset, dynamic, fmt,
                            half=None, int8=None, simplify=None,
                            calib_data=None):
    from flashstudio.utils import flash

    half = bool(st.session_state.get("export_half")) if half is None else bool(half)
    int8 = bool(st.session_state.get("export_int8")) if int8 is None else bool(int8)
    simplify = (
        bool(st.session_state.get("export_simplify"))
        if simplify is None else bool(simplify)
    )
    if calib_data is None:
        calib_data = st.session_state.get("export_calib_data")

    fmt_key = str(fmt or "onnx").lower()
    target = {
        "onnx": "onnx",
        "torchscript": "torchscript",
        "tensorrt engine": "engine",
        "engine": "engine",
    }.get(fmt_key, "onnx")
    label = {
        "onnx": "ONNX",
        "torchscript": "TorchScript",
        "engine": "TensorRT engine",
    }[target]
    try:
        from ultralytics import YOLO
    except ImportError:
        flash("Ultralytics is not installed — run `pip install ultralytics`.", "error")
        st.rerun()
        return

    with st.spinner(f"Exporting YOLO weights to {label}..."):
        try:
            model = YOLO(weights_path)
            kwargs = {"format": target, "imgsz": int(img_size)}
            if target == "onnx":
                kwargs.update(
                    opset=max(int(opset), 16),
                    dynamic=bool(dynamic),
                    simplify=bool(simplify),
                )
            if target == "engine":
                kwargs.update(half=bool(half))
            if int8:
                calib = str(calib_data or "").strip()
                if not calib or not os.path.isfile(calib):
                    raise RuntimeError(
                        "INT8 export needs a calibration dataset YAML "
                        "(calib_data)."
                    )
                kwargs.update(int8=True, data=calib)
            out = str(model.export(**kwargs))
            sz = f"{os.path.getsize(out) / (1024 * 1024):.1f} MB" if os.path.isfile(out) else "—"
            st.session_state["exported_files"] = [{
                "format": label, "path": out, "size": sz, "success": True,
                "architecture": target, "input_size": int(img_size),
                "opset": int(opset) if target == "onnx" else None,
            }]
            flash(f"Export successful: `{os.path.basename(out)}` ({sz})", "success")
        except Exception as exc:
            st.session_state["exported_files"] = [{
                "format": label, "path": "", "size": "—", "success": False,
            }]
            flash(f"Export failed: {exc}", "error")
    st.rerun()


def _run_export(weights_path, img_size, opset=13, dynamic=True, fmt="onnx"):
    from flashstudio.utils import flash

    if not weights_path or not os.path.isfile(weights_path):
        flash(f"Weights not found: `{weights_path}`", "error")
        st.rerun()
        return

    if str(weights_path).lower().endswith(".pt"):
        _run_ultralytics_export(weights_path, img_size, opset, dynamic, fmt)
        return

    if str(fmt).lower() not in ("onnx", "torchscript"):
        flash(
            "TensorRT engine export is only available for Ultralytics `.pt` models.",
            "error",
        )
        st.rerun()
        return

    import torch
    try:
        from flashdet.cfg import get_config  # noqa: F401
        from flashdet.models.detector import build_model  # noqa: F401
    except ImportError:
        flash("FlashDet not installed — run `pip install flashdet`", "error")
        st.rerun()
        return

    ext = ".onnx" if fmt == "onnx" else ".pt"
    out = os.path.splitext(weights_path)[0] + ext
    label = "ONNX" if fmt == "onnx" else "TorchScript"
    with st.spinner(f"Exporting to {label}..."):
        try:
            model, inp_scalar, arch = _build_export_model(weights_path, img_size)
            dummy = torch.randn(1, 3, inp_scalar, inp_scalar)
            effective_opset = None

            if fmt == "onnx":
                import warnings
                # DySample/GridSample needs opset >= 16. Older sessions may
                # still have 11-15 saved, so clamp instead of failing.
                effective_opset = max(int(opset), 16)
                if effective_opset != int(opset):
                    st.session_state["export_opset"] = effective_opset
                    flash(f"Raised opset to {effective_opset} (GridSample requires >= 16).", "info")
                dyn = {"images": {0: "batch"}, "output": {0: "batch"}} if dynamic else None
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    # ``dynamo=True`` is the default in PyTorch >= 2.9 and
                    # currently fails on FlashDet's output structure.  The
                    # legacy exporter handles this model correctly.
                    torch.onnx.export(
                        model, dummy, out, opset_version=effective_opset,
                        input_names=["images"], output_names=["output"],
                        dynamic_axes=dyn,
                        dynamo=False,
                    )
            else:
                traced = torch.jit.trace(_TorchScriptWrapper(model), dummy)
                traced.save(out)

            sz = f"{os.path.getsize(out) / (1024 * 1024):.1f} MB" if os.path.isfile(out) else "—"
            st.session_state["exported_files"] = [{
                "format": label, "path": out, "size": sz, "success": True,
                "architecture": arch, "input_size": inp_scalar,
                "opset": effective_opset,
            }]
            flash(f"Export successful: `{os.path.basename(out)}` ({sz})", "success")
        except Exception as e:
            st.session_state["exported_files"] = [{
                "format": label, "path": out, "size": "—", "success": False,
            }]
            flash(f"Export failed: {e}", "error")
    st.rerun()
