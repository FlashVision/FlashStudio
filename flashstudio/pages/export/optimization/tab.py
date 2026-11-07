"""Export — framework-aware optimization pipeline."""

import json
import os

import streamlit as st

from flashstudio.pages.export.weights.settings import _get_export_save_dir


def _artifact_dir() -> str:
    root = _get_export_save_dir()
    path = os.path.join(root, "optimized")
    os.makedirs(path, exist_ok=True)
    return path


def _size_mb(path: str) -> str:
    try:
        return f"{os.path.getsize(path) / (1024 * 1024):.2f} MB"
    except OSError:
        return "—"


def _select_weight(framework: str = "FlashDet") -> tuple[str, int]:
    from flashstudio.pages.export.weights.settings import render_checkpoint_card

    return render_checkpoint_card(framework, "optimize")


def _render_fusion(weights: str, input_size: int):
    st.markdown("#### Layer & kernel fusion")
    st.caption(
        "Folds BatchNorm into convs, collapses reparameterizable blocks, and "
        "merges conv→BN→activation triples. Training should use the original checkpoint."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        skip_rep = st.checkbox("Skip Rep fusion", False, key="opt_skip_rep")
    with c2:
        half = st.checkbox("Save FP16 fused", False, key="opt_fuse_half")
    with c3:
        verify = st.checkbox("Parity check", True, key="opt_fuse_verify")

    if st.button("Run fusion", type="primary", key="opt_fuse_btn",
                 use_container_width=True):
        if not weights:
            st.error("Select a checkpoint first.")
            return
        with st.spinner("Fusing layers…"):
            try:
                import torch
                from flashdet.deployment import (
                    build_model_from_checkpoint, input_size_from_checkpoint,
                )
                from flashdet.fusing import default_passes, run_fusions
                from flashdet.fusing.checkpoint import save_fused_checkpoint
                from flashdet.utils.checkpoint import safe_torch_load

                size = input_size or input_size_from_checkpoint(weights)
                model = build_model_from_checkpoint(weights, fuse=False)
                probe = torch.randn(1, 3, size, size)
                passes = default_passes(example_input=probe)
                if skip_rep:
                    passes = [p for p in passes if p.name != "rep_fusion"]
                fused, report = run_fusions(
                    model, passes, example_input=probe, verify=verify,
                )
                out = os.path.join(
                    _artifact_dir(),
                    os.path.splitext(os.path.basename(weights))[0]
                    + ("_fused_fp16.pth" if half else "_fused.pth"),
                )
                raw = safe_torch_load(weights, map_location="cpu")
                config = raw.get("config", {}) if isinstance(raw, dict) else {}
                saved = fused.half() if half else fused
                save_fused_checkpoint(
                    saved, out, config=config or {},
                    input_size=size, report=report,
                )
                st.session_state["opt_fusion"] = {
                    "path": out,
                    "report": report.to_dict(),
                    "text": str(report),
                }
                st.success(f"Fused checkpoint saved to `{out}`")
            except Exception as exc:
                import traceback
                st.error(f"Fusion failed: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    result = st.session_state.get("opt_fusion")
    if result:
        report = result["report"]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Changes", report.get("total_changes", 0))
        with m2:
            st.metric(
                "Params",
                f"{report.get('params_before', 0):,} → {report.get('params_after', 0):,}",
            )
        with m3:
            parity = report.get("parity_ok")
            st.metric("Parity", "OK" if parity else ("MISMATCH" if parity is False else "—"))
        st.caption(f"`{result['path']}` · {_size_mb(result['path'])}")
        st.code(result.get("text", ""), language="text")
        with open(result["path"], "rb") as f:
            st.download_button(
                "Download fused checkpoint", f,
                file_name=os.path.basename(result["path"]),
                key="opt_dl_fused", use_container_width=True,
            )


def _render_onnx(weights: str, input_size: int):
    st.markdown("#### ONNX export + parity")
    st.caption(
        "Exports a portable ONNX graph, optionally fusing Rep blocks and "
        "simplifying the graph. Parity compares PyTorch and ONNX outputs."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        fuse = st.checkbox("Fuse before export", True, key="opt_onnx_fuse")
    with c2:
        simplify = st.checkbox("Simplify graph", False, key="opt_onnx_simplify")
    with c3:
        opset = st.number_input("Opset", 11, 20, 18, key="opt_onnx_opset")

    if st.button("Export ONNX", type="primary", key="opt_onnx_btn",
                 use_container_width=True):
        if not weights:
            st.error("Select a checkpoint first.")
            return
        with st.spinner("Exporting ONNX…"):
            try:
                from flashdet.deployment import check_parity, export_onnx

                out = os.path.join(
                    _artifact_dir(),
                    os.path.splitext(os.path.basename(weights))[0] + "_fused.onnx",
                )
                path = export_onnx(
                    weights, out, fuse=fuse, simplify=simplify,
                    opset=int(opset),
                )
                try:
                    parity = check_parity(weights, path)
                except Exception as exc:
                    parity = {"ok": None, "error": str(exc)}
                st.session_state["opt_onnx"] = {"path": path, "parity": parity}
                st.success(f"ONNX exported to `{path}`")
            except Exception as exc:
                import traceback
                st.error(f"ONNX export failed: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    result = st.session_state.get("opt_onnx")
    if result:
        parity = result["parity"]
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Size", _size_mb(result["path"]))
        with m2:
            st.metric("Parity", "OK" if parity.get("ok") else ("—" if parity.get("ok") is None else "FAIL"))
        with m3:
            st.metric("Max |Δ|", f"{parity.get('max_abs_diff', 0):.2e}"
                      if parity.get("max_abs_diff") is not None else "—")
        st.caption(f"`{result['path']}`")
        if parity.get("error"):
            st.warning(parity["error"])
        with open(result["path"], "rb") as f:
            st.download_button(
                "Download ONNX", f, file_name=os.path.basename(result["path"]),
                key="opt_dl_onnx", use_container_width=True,
            )


def _render_int8(weights: str):
    st.markdown("#### INT8 quantization")
    st.caption(
        "Statically calibrates an ONNX graph to INT8 QDQ. Calibration images "
        "should look like the deployment data."
    )
    onnx_result = st.session_state.get("opt_onnx") or {}
    default_onnx = onnx_result.get("path", "")
    if default_onnx and not st.session_state.get("opt_int8_onnx"):
        st.session_state["opt_int8_onnx"] = default_onnx
    calib_default = str(st.session_state.get("val_img_path") or "")
    if calib_default and not st.session_state.get("opt_int8_calib"):
        st.session_state["opt_int8_calib"] = calib_default
    onnx_path = st.text_input("ONNX model", key="opt_int8_onnx")
    calib = st.text_input("Calibration images", key="opt_int8_calib")
    c1, c2 = st.columns(2)
    with c1:
        batches = st.number_input("Calibration batches", 1, 200, 16, key="opt_int8_batches")
    with c2:
        min_images = st.number_input("Minimum images", 1, 1000, 16, key="opt_int8_min")

    if st.button("Quantize INT8", type="primary", key="opt_int8_btn",
                 use_container_width=True):
        if not onnx_path or not os.path.isfile(onnx_path):
            st.error("Export or select an ONNX model first.")
            return
        if not calib or not os.path.isdir(calib):
            st.error("Choose a calibration image folder.")
            return
        with st.spinner("Quantizing to INT8…"):
            try:
                from flashdet.deployment import detection_parity, onnx_quantize_int8

                out = os.path.join(
                    _artifact_dir(),
                    os.path.splitext(os.path.basename(onnx_path))[0] + "_int8.onnx",
                )
                path = onnx_quantize_int8(
                    onnx_path, out, calib_images=calib,
                    min_calib_images=int(min_images),
                    max_batches=int(batches),
                )
                det = None
                if weights and os.path.isfile(weights):
                    try:
                        det = detection_parity(weights, path)
                    except Exception:
                        det = None
                st.session_state["opt_int8"] = {"path": path, "det": det}
                st.success(f"INT8 model saved to `{path}`")
            except Exception as exc:
                import traceback
                st.error(f"INT8 quantization failed: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    result = st.session_state.get("opt_int8")
    if result:
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Size", _size_mb(result["path"]))
        with m2:
            det = result.get("det") or {}
            iou = det.get("mean_box_iou")
            st.metric("Mean box IoU", f"{iou:.3f}" if iou is not None else "—")
        with m3:
            st.metric("Boxes", det.get("num_onnx_boxes", "—"))
        st.caption(f"`{result['path']}`")
        with open(result["path"], "rb") as f:
            st.download_button(
                "Download INT8 ONNX", f,
                file_name=os.path.basename(result["path"]),
                key="opt_dl_int8", use_container_width=True,
            )


def _render_benchmark(weights: str, input_size: int):
    st.markdown("#### Benchmark")
    st.caption("Measures FPS, latency, parameters, and file size for PyTorch/ONNX artifacts.")
    candidates = []
    if weights:
        candidates.append(weights)
    for key in ("opt_fusion", "opt_onnx", "opt_int8"):
        result = st.session_state.get(key) or {}
        path = result.get("path")
        if path and os.path.isfile(path) and path not in candidates:
            candidates.append(path)

    selected = st.multiselect(
        "Artifacts", candidates, default=candidates[:2],
        format_func=os.path.basename, key="opt_bench_models",
    )
    c1, c2 = st.columns(2)
    with c1:
        warmup = st.number_input("Warmup", 0, 100, 10, key="opt_bench_warmup")
    with c2:
        iterations = st.number_input("Iterations", 10, 1000, 100, key="opt_bench_iter")

    if st.button("Run benchmark", type="primary", key="opt_bench_btn",
                 use_container_width=True):
        if not selected:
            st.error("Select at least one artifact.")
            return
        with st.spinner("Benchmarking…"):
            try:
                from flashdet.analytics import Benchmark

                device = "cuda" if _has_cuda() else "cpu"
                rows = []
                for path in selected:
                    result = Benchmark(
                        path, device=device, input_size=input_size,
                    ).run(warmup=int(warmup), iterations=int(iterations))
                    result["model"] = os.path.basename(path)
                    result["size"] = _size_mb(path)
                    rows.append(result)
                st.session_state["opt_benchmark"] = rows
            except Exception as exc:
                import traceback
                st.error(f"Benchmark failed: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())

    rows = st.session_state.get("opt_benchmark")
    if rows:
        import pandas as pd

        frame = pd.DataFrame(rows)[
            ["model", "fps", "latency_ms", "params", "size"]
        ]
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download report", json.dumps(rows, indent=2),
            "benchmark.json", "application/json", use_container_width=True,
        )


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _ultralytics_candidates(weights: str) -> list[str]:
    candidates = []
    if weights:
        candidates.append(weights)
    for exp in st.session_state.get("exported_files", []) or []:
        path = exp.get("path")
        if path and os.path.isfile(path) and path not in candidates:
            candidates.append(path)
    root = _get_export_save_dir()
    if os.path.isdir(root):
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("yolo_dataset", "ultralytics")]
            for name in files:
                if name.endswith((".pt", ".onnx", ".engine", ".torchscript")):
                    path = os.path.join(dirpath, name)
                    if path not in candidates:
                        candidates.append(path)
    return candidates


def _benchmark_ultralytics(path: str, imgsz: int, device: str,
                           warmup: int, iterations: int) -> dict:
    import time

    import numpy as np
    from ultralytics import YOLO

    from flashstudio.utils.model_loader import _patch_torchvision_nms

    _patch_torchvision_nms()
    model = YOLO(path)
    image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(max(0, int(warmup))):
        model.predict(image, imgsz=int(imgsz), device=device, verbose=False)
    started = time.perf_counter()
    for _ in range(max(1, int(iterations))):
        model.predict(image, imgsz=int(imgsz), device=device, verbose=False)
    elapsed = max(time.perf_counter() - started, 1e-6)
    params = 0
    try:
        params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        params = 0
    return {
        "model": os.path.basename(path),
        "fps": round(int(iterations) / elapsed, 2),
        "latency_ms": round(elapsed / max(1, int(iterations)) * 1000, 3),
        "params": params,
        "size": _size_mb(path),
    }


def _render_ultralytics_export(selected: str, input_size: int,
                               int8_mode: bool = False):
    from flashstudio.pages.export.weights.exporter import _run_ultralytics_export

    if int8_mode:
        st.markdown("#### INT8 TensorRT engine")
        st.caption(
            "Ultralytics INT8 export targets TensorRT engines and requires the "
            "TensorRT runtime plus a calibration dataset YAML."
        )
        from flashstudio.components.dependencies import render_dependency_hint

        render_dependency_hint(
            "TensorRT", "TensorRT INT8 export",
            key_suffix="export_ul_int8",
        )
        st.caption(f"Input size · {input_size}×{input_size}")
        st.checkbox("Half (FP16)", True, key="ul_int8_half")
        calib = st.text_input(
            "Calibration data.yaml", key="ul_int8_calib",
            placeholder="path/to/data.yaml",
        )
        if st.button("Export INT8 engine", type="primary", key="ul_int8_export",
                     use_container_width=True):
            _run_ultralytics_export(
                selected, input_size, 17, True, "TensorRT engine",
                half=st.session_state.get("ul_int8_half", True),
                int8=True, simplify=False, calib_data=calib,
            )
        return

    st.markdown("#### Export")
    f1, f2 = st.columns(2)
    with f1:
        fmt = st.selectbox(
            "Format", ["ONNX", "TorchScript", "TensorRT engine"],
            key="ul_opt_fmt",
        )
    with f2:
        opset = st.number_input("Opset", 11, 20, 17, key="ul_opt_opset")
    st.caption(f"Input size · {input_size}×{input_size}")
    o1, o2, o3 = st.columns(3)
    with o1:
        dynamic = st.checkbox("Dynamic batch", True, key="ul_opt_dynamic")
    with o2:
        st.checkbox("Half (FP16)", False, key="ul_opt_half")
    with o3:
        st.checkbox("Simplify ONNX", False, key="ul_opt_simplify")
    st.caption(
        "TensorRT engine export requires the TensorRT runtime. Half is applied "
        "for TensorRT. ONNX simplify uses onnxslim when available."
    )
    if st.button(f"Export {fmt}", type="primary", key="ul_opt_export",
                 use_container_width=True):
        _run_ultralytics_export(
            selected, input_size, opset, dynamic, fmt,
            half=st.session_state.get("ul_opt_half"),
            int8=False,
            simplify=st.session_state.get("ul_opt_simplify"),
        )


def _render_ultralytics_benchmark(selected: str, input_size: int):
    st.markdown("#### Benchmark")
    candidates = _ultralytics_candidates(selected)
    chosen = st.multiselect(
        "Artifacts", candidates, default=candidates[:1],
        format_func=os.path.basename, key="ul_opt_bench_models",
    )
    b1, b2 = st.columns(2)
    with b1:
        warmup = st.number_input("Warmup", 0, 100, 5, key="ul_opt_warmup")
    with b2:
        iterations = st.number_input("Iterations", 5, 500, 30, key="ul_opt_iter")
    st.caption(f"Input size · {input_size}×{input_size}")
    if st.button("Run benchmark", type="primary", key="ul_opt_bench",
                 use_container_width=True):
        if not chosen:
            st.error("Select at least one artifact.")
        else:
            rows = []
            device = "cuda" if _has_cuda() else "cpu"
            with st.spinner("Benchmarking Ultralytics artifacts…"):
                for path in chosen:
                    try:
                        rows.append(_benchmark_ultralytics(
                            path, int(input_size), device, int(warmup), int(iterations),
                        ))
                    except Exception as exc:
                        st.warning(f"{os.path.basename(path)}: {exc}")
            st.session_state["ul_opt_benchmark"] = rows

    rows = st.session_state.get("ul_opt_benchmark")
    if rows:
        import pandas as pd

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.download_button(
            "Download report", json.dumps(rows, indent=2),
            "ultralytics_benchmark.json", "application/json",
            use_container_width=True,
        )


def _empty_step(message: str):
    st.caption(message)


def _render_output(prefix: str = "optimize"):
    with st.container(border=True):
        st.markdown("#### Output")
        exported = st.session_state.get("exported_files") or []
        if not exported:
            st.caption("Exported artifacts appear here after running an export.")
            return
        for i, exp in enumerate(exported):
            path = exp.get("path", "")
            if not path:
                continue
            st.markdown(
                f"**{exp.get('format', 'Artifact')}** · `{os.path.basename(path)}`"
            )
            st.caption(f"{exp.get('size', '—')} · `{path}`")
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    st.download_button(
                        "Download", f, file_name=os.path.basename(path),
                        key=f"{prefix}_dl_{i}", use_container_width=True,
                    )


def _render_ultralytics_optimization():
    from flashstudio.pages.export.weights.settings import render_checkpoint_card

    selected, input_size = render_checkpoint_card("Ultralytics", "optimize")

    tab_fusion, tab_export, tab_int8, tab_bench = st.tabs(
        ["Fusion", "Export", "INT8", "Benchmark"]
    )
    with tab_fusion:
        with st.container(border=True):
            st.markdown("#### Fusion")
            st.caption(
                "Ultralytics applies layer fusion automatically during ONNX and "
                "TensorRT export. There is no separate fused PyTorch checkpoint."
            )
            st.info(
                "Use Export → ONNX or TensorRT engine to get the fused deployment graph."
            )
    with tab_export:
        with st.container(border=True):
            if selected:
                _render_ultralytics_export(selected, input_size)
            else:
                _empty_step("Select or download an Ultralytics checkpoint to export.")
    with tab_int8:
        with st.container(border=True):
            if selected:
                _render_ultralytics_export(selected, input_size, int8_mode=True)
            else:
                _empty_step("Select or download an Ultralytics checkpoint to quantize.")
    with tab_bench:
        with st.container(border=True):
            if selected:
                _render_ultralytics_benchmark(selected, input_size)
            else:
                _empty_step("Select or download an Ultralytics checkpoint to benchmark.")
    _render_output("ul_opt")


def _render_optimization_tab(framework: str = "FlashDet"):
    if framework == "Ultralytics":
        _render_ultralytics_optimization()
        return

    if framework == "YOLOX":
        st.info(
            "YOLOX training is not implemented by the installed FlashDet build, "
            "but fusion, ONNX export, INT8, and benchmarking work on YOLOX checkpoints."
        )

    weights, input_size = _select_weight(framework)

    tab_fuse, tab_export, tab_int8, tab_bench = st.tabs(
        ["Fusion", "Export", "INT8", "Benchmark"]
    )
    with tab_fuse:
        with st.container(border=True):
            if weights:
                _render_fusion(weights, input_size)
            else:
                st.markdown("#### Fusion")
                _empty_step(f"Select or download a {framework} checkpoint to fuse.")
    with tab_export:
        with st.container(border=True):
            if weights:
                _render_onnx(weights, input_size)
            else:
                st.markdown("#### Export")
                _empty_step(f"Select or download a {framework} checkpoint to export.")
    with tab_int8:
        with st.container(border=True):
            if weights:
                _render_int8(weights)
            else:
                st.markdown("#### INT8")
                _empty_step(f"Select or download a {framework} checkpoint to quantize.")
    with tab_bench:
        with st.container(border=True):
            if weights:
                _render_benchmark(weights, input_size)
            else:
                st.markdown("#### Benchmark")
                _empty_step(f"Select or download a {framework} checkpoint to benchmark.")

    _render_output(f"{framework.lower()}_opt")
