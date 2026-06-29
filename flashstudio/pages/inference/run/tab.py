"""Inference — Run tab."""

import io
import time
import json
import streamlit as st
from PIL import Image
from flashstudio.constants import DEFAULT_MODEL_ARCH
from flashstudio.pages.inference.run.detection import (
    _detect, _draw_boxes, _to_coco_format,
)
from flashstudio.pages.inference.run.video import _run_video


def _tab_run():
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Model", st.session_state.get("infer_model_arch", DEFAULT_MODEL_ARCH).replace("FlashDet-", ""))
    with m2:
        st.metric("Input", st.session_state.get("infer_input_type", "Images"))
    with m3:
        st.metric("Solution", st.session_state.get("selected_solution", "Detection")[:12])
    with m4:
        st.metric("Device", st.session_state.get("infer_device", "cpu").split(" ")[0])

    input_type = st.session_state.get("infer_input_type", "Images")
    has_data = st.session_state.get("infer_images") or st.session_state.get("infer_video") or st.session_state.get("infer_stream_url")
    if not has_data:
        st.warning("No data. Upload in Data tab.")
        return

    rc1, rc2 = st.columns([1, 3])
    with rc1:
        run = st.button("Run Inference", type="primary", key="run_infer_btn", use_container_width=True)
    with rc2:
        st.checkbox("Per-frame details", False, key="show_per_frame")

    if run:
        if input_type == "Images":
            _run_images()
        elif input_type == "Video":
            _run_video()
        else:
            st.info("Stream — coming soon")

    if input_type == "Images":
        _show_image_results()
    else:
        _show_video_results()


def _run_images():
    images = st.session_state.get("infer_images", [])
    if not images:
        return
    results = []
    total_time = 0
    progress = st.progress(0)
    for i, f in enumerate(images):
        img = Image.open(f).convert("RGB")
        t0 = time.perf_counter()
        dets = _detect(img)
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        results.append({"name": f.name, "dets": dets, "annotated": _draw_boxes(img.copy(), dets),
                         "inference_time_ms": elapsed * 1000, "width": img.width, "height": img.height})
        progress.progress((i + 1) / len(images))
    progress.empty()
    st.session_state["infer_img_results"] = results
    st.session_state["infer_total_time"] = total_time
    st.rerun()


def _show_image_results():
    if "infer_img_results" not in st.session_state:
        return
    results = st.session_state["infer_img_results"]
    total_dets = sum(len(r["dets"]) for r in results)
    total_time = st.session_state.get("infer_total_time", 1)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("FPS", f"{len(results) / max(total_time, 0.001):.1f}")
    with c2:
        st.metric("Latency", f"{total_time / max(len(results), 1) * 1000:.1f}ms")
    with c3:
        st.metric("Detections", total_dets)
    with c4:
        st.metric("Images", len(results))

    for r in results:
        ci, ct = st.columns([1, 1])
        with ci:
            st.image(r["annotated"], caption=f"{r['name'][:15]} ({r['inference_time_ms']:.0f}ms)", use_container_width=True)
        with ct:
            if r["dets"]:
                import pandas as pd
                st.dataframe(pd.DataFrame(r["dets"], columns=["Class", "Conf", "BBox"]),
                             use_container_width=True, hide_index=True, height=180)

    with st.expander("Export"):
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            import pandas as pd
            all_d = [[r["name"]] + d for r in results for d in r["dets"]]
            if all_d:
                st.download_button("CSV", pd.DataFrame(all_d, columns=["File", "Class", "Conf", "BBox"]).to_csv(index=False),
                                   "detections.csv", "text/csv", use_container_width=True)
        with ec2:
            st.download_button("JSON", json.dumps(_to_coco_format(results), indent=2),
                               "coco.json", "application/json", use_container_width=True)
        with ec3:
            if results:
                buf = io.BytesIO()
                results[0]["annotated"].save(buf, format="PNG")
                st.download_button("Image", buf.getvalue(), "annotated.png", "image/png", use_container_width=True)


def _show_video_results():
    if "video_results" not in st.session_state:
        return
    res = st.session_state["video_results"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Frames", res.get("frames", 0))
    with c2:
        st.metric("Detections", res.get("detections", 0))
    with c3:
        st.metric("FPS", f"{res.get('processing_fps', 0):.1f}")
    with c4:
        st.metric("Time", f"{res.get('elapsed_time', 0):.1f}s")

    sol = res.get("solution_output", {})
    in_count = sol.get("in_count") or sol.get("in", 0)
    out_count = sol.get("out_count") or sol.get("out", 0)
    if in_count or out_count:
        sc1, sc2, sc3 = st.columns(3)
        with sc1: st.metric("In", in_count)
        with sc2: st.metric("Out", out_count)
        with sc3: st.metric("Net", in_count - out_count)
    region_data = sol.get("region_counts") or {k: v for k, v in sol.items()
                                                if k not in ("in", "out", "in_count", "out_count", "total", "speeds") and isinstance(v, (int, float))}
    if region_data:
        cols = st.columns(min(len(region_data), 6))
        for i, (r, c) in enumerate(list(region_data.items())[:6]):
            with cols[i]:
                st.metric(r, c)
    if "speeds" in sol:
        import pandas as pd
        st.dataframe(pd.DataFrame(sol["speeds"], columns=["ID", "km/h"]), hide_index=True, height=150)

    if res.get("frames_preview"):
        cols = st.columns(min(len(res["frames_preview"]), 4))
        for i, frame in enumerate(res["frames_preview"][:4]):
            with cols[i]:
                st.image(frame, use_container_width=True)

    if res.get("output_path"):
        st.caption(f"Output: `{res['output_path']}`")
