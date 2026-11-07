"""Inference — Run tab."""

import io
import os
import time
import json
import zipfile
import streamlit as st
from PIL import Image
from flashstudio.pages.inference.run.detection import (
    _detect, _draw_boxes, _to_coco_format,
)
from flashstudio.pages.inference._common import SOLUTIONS, _get_class_names
from flashstudio.pages.inference.run.video import _run_video


def _model_label() -> str:
    from flashstudio.utils.filesystem import weight_label
    if st.session_state.get("weight_source") == "Upload":
        upload = st.session_state.get("infer_weights_file")
        if upload is not None:
            return weight_label(getattr(upload, "name", "uploaded"))
    path = str(st.session_state.get("infer_weights_path", "") or "")
    if path:
        return weight_label(path)
    upload = st.session_state.get("infer_weights_file")
    if upload is not None:
        return weight_label(getattr(upload, "name", "uploaded"))
    return "—"


def _tab_run():
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Model", _model_label())
    with m2:
        st.metric("Input", st.session_state.get("infer_input_type", "Images"))
    with m3:
        sol_name = st.session_state.get("selected_solution", "Detection")
        sol_display = "Detection" if "Detection Only" in sol_name else sol_name.split(" (")[0]
        st.metric("Solution", sol_display)
    with m4:
        device_label = st.session_state.get("infer_device", "cpu").split(" ")[0]
        st.metric("Device", str(device_label).upper())

    input_type = st.session_state.get("infer_input_type", "Images")
    if input_type == "Images":
        has_data = bool(st.session_state.get("infer_images"))
        empty_hint = "No images yet. Add images in the Data tab."
    elif input_type == "Video":
        has_data = bool(st.session_state.get("infer_video"))
        empty_hint = "No video yet. Upload a video in the Data tab."
    else:
        has_data = bool(st.session_state.get("infer_stream_url"))
        empty_hint = "No stream URL yet. Enter an RTSP URL in the Data tab."

    with st.container(border=True):
        st.markdown("#### Run")
        if st.session_state.get("infer_demo_mode"):
            st.warning("Demo mode is ON — detections are fabricated for UI previews.")
        if not has_data:
            st.info(empty_hint)
            if st.button("Add input in Data", key="run_add_input"):
                st.session_state["_infer_tab_next"] = "Data"
                st.rerun()
            return

        solution_name = st.session_state.get(
            "selected_solution", "None (Detection Only)"
        )
        solution = SOLUTIONS.get(solution_name, {})
        needs_zone = bool(solution.get("needs_zone"))
        has_zone = bool(
            st.session_state.get("zone_line_points")
            or st.session_state.get("zone_polygons")
        )
        if needs_zone and not has_zone:
            st.warning("This solution needs a zone — draw one in the Solution tab first.")
            if st.button("Open Solution", key="run_open_solution"):
                st.session_state["_infer_tab_next"] = "Solution"
                st.rerun()

        rc1, rc2 = st.columns([1, 3])
        with rc1:
            run = st.button(
                "Run inference", type="primary", key="run_infer_btn",
                use_container_width=True, disabled=needs_zone and not has_zone,
            )
        with rc2:
            st.checkbox("Per-frame details", False, key="show_per_frame")

        if run:
            if input_type == "Images":
                _run_images()
            elif input_type == "Video":
                _run_video()
            else:
                st.info("Switch **Run mode** to **Live** for continuous RTSP inference.")

    with st.container(border=True):
        st.markdown("#### Results")
        if input_type == "Images":
            _show_image_results()
        else:
            _show_video_results()


def _run_images():
    images = st.session_state.get("infer_images", [])
    if not images:
        return
    from flashstudio.pages.inference.run.detection import _resolve_weights_path
    if not _resolve_weights_path() and not st.session_state.get("infer_demo_mode"):
        st.error("No model selected. Upload a checkpoint or enable Demo mode in Inference → Model.")
        return
    results = []
    total_time = 0
    progress = st.progress(0)
    skipped = []
    for i, f in enumerate(images):
        try:
            with Image.open(f) as source:
                img = source.convert("RGB")
        except Exception as exc:
            skipped.append(f"{getattr(f, 'name', 'image')}: {exc}")
            progress.progress((i + 1) / len(images))
            continue
        t0 = time.perf_counter()
        dets = _detect(img)
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        results.append({"name": f.name, "dets": dets, "annotated": _draw_boxes(img.copy(), dets),
                         "inference_time_ms": elapsed * 1000, "width": img.width, "height": img.height})
        progress.progress((i + 1) / len(images))
    progress.empty()
    if skipped:
        from flashstudio.utils import flash

        flash("Skipped unreadable image(s): " + "; ".join(skipped[:10]), "warning")
    st.session_state["infer_img_results"] = results
    st.session_state["infer_total_time"] = total_time
    st.rerun()


def _show_image_results():
    if "infer_img_results" not in st.session_state:
        st.caption("Run inference to see detections here.")
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
            st.download_button("JSON", json.dumps(_to_coco_format(results, _get_class_names()), indent=2),
                               "coco.json", "application/json", use_container_width=True)
        with ec3:
            if results:
                buf = io.BytesIO()
                results[0]["annotated"].save(buf, format="PNG")
                st.download_button("Image", buf.getvalue(), "annotated.png", "image/png", use_container_width=True)

    _render_active_learning(results)


def _confidence(row) -> float:
    try:
        return float(row[1])
    except (TypeError, ValueError, IndexError):
        return 0.0


def _render_active_learning(results):
    """Export uncertain detections for annotation review."""
    with st.expander("Active learning — review low-confidence detections"):
        st.caption("Exports detections in this band with their images.")
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            low = st.slider("Low confidence", 0.0, 1.0, 0.15, 0.01, key="al_low")
        with c2:
            high = st.slider("High confidence", 0.0, 1.0, 0.45, 0.01, key="al_high")
        with c3:
            include_images = st.checkbox("Include images", True, key="al_images")

        if low > high:
            st.warning("Low confidence must be below high confidence.")
            return

        selected = []
        uncertain = 0
        for r in results:
            kept = [d for d in r["dets"] if low <= _confidence(d) <= high]
            uncertain += len(kept)
            selected.append({**r, "dets": kept})

        st.caption(f"Uncertain detections: **{uncertain}** across {len(results)} image(s).")
        if not uncertain:
            return

        coco = _to_coco_format(
            [{"name": r["name"], "dets": r["dets"], "width": r["width"], "height": r["height"]}
             for r in selected],
            _get_class_names(),
        )
        import pandas as pd

        rows = [[r["name"]] + d for r in selected for d in r["dets"]]
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button(
                "Uncertain CSV",
                pd.DataFrame(rows, columns=["File", "Class", "Conf", "BBox"]).to_csv(index=False),
                "uncertain_detections.csv", "text/csv", use_container_width=True,
            )
        with e2:
            st.download_button(
                "Review COCO JSON",
                json.dumps(coco, indent=2),
                "review_annotations.json", "application/json", use_container_width=True,
            )
        with e3:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("annotations.json", json.dumps(coco, indent=2))
                zf.writestr(
                    "detections.csv",
                    pd.DataFrame(rows, columns=["File", "Class", "Conf", "BBox"]).to_csv(index=False),
                )
                if include_images:
                    for r in selected:
                        if not r["dets"]:
                            continue
                        img_buf = io.BytesIO()
                        r["annotated"].save(img_buf, format="PNG")
                        zf.writestr(f"images/{os.path.basename(r['name'])}.png", img_buf.getvalue())
            st.download_button(
                "Review bundle (ZIP)",
                buf.getvalue(),
                "active_learning_review.zip",
                "application/zip",
                use_container_width=True,
            )


def _show_video_results():
    if "video_results" not in st.session_state:
        st.caption("Run inference to see video metrics and previews here.")
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
