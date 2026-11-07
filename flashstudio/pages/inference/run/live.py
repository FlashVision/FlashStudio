"""Inference — Live stream (RTSP/webcam) tab."""

import os
import time

import numpy as np
import streamlit as st

from flashstudio.constants import (
    INFER_CONF_THRESHOLD, INFER_NMS_THRESHOLD, INFER_IMG_SIZE,
    INFER_FRAME_SKIP, INFER_MAX_FRAMES,
)
from flashstudio.pages.inference.run.detection import (
    _detect_demo, _draw_boxes_cv2, _get_adapter, _resolve_weights_path,
)
from flashstudio.pages.inference.run.video import _create_solution


def _live_key(name):
    return f"_live_{name}"


def _cleanup_live():
    cap = st.session_state.get(_live_key("cap"))
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    writer = st.session_state.get(_live_key("writer"))
    if writer is not None:
        try:
            writer.release()
        except Exception:
            pass
    for key in ("cap", "writer", "solution", "adapter", "weights", "started", "frames", "dets", "last", "output"):
        st.session_state.pop(_live_key(key), None)
    st.session_state["live_running"] = False


def _render_live_tab():
    with st.container(border=True):
        st.markdown("#### Live stream")

        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            if (
                "live_source" not in st.session_state
                and st.session_state.get("infer_stream_url")
            ):
                st.session_state["live_source"] = st.session_state["infer_stream_url"]
            source = st.text_input(
                "Stream source",
                placeholder="rtsp://user:pass@host:554/stream  or  0 for webcam",
                key="live_source",
            )
        with c2:
            max_frames = st.number_input("Max frames", 0, 100000, 300, 10,
                                         key="live_max_frames")
        with c3:
            skip = st.number_input("Frame skip", 1, 30, max(1, INFER_FRAME_SKIP),
                                   key="live_skip")

        running = bool(st.session_state.get("live_running"))
        b1, b2, b3 = st.columns([1, 1, 4])
        with b1:
            start = st.button("Start live", type="primary", key="live_start_btn",
                              disabled=running or not source)
        with b2:
            stop = st.button("Stop", key="live_stop_btn", disabled=not running)
        with b3:
            st.checkbox("Save output", value=False, key="live_save_output")

    if stop:
        _cleanup_live()
        st.rerun()

    if start:
        cap = None
        try:
            import cv2
            cap = cv2.VideoCapture(source)
            if isinstance(source, str) and source.strip().isdigit():
                cap = cv2.VideoCapture(int(source.strip()))
            if not cap.isOpened():
                cap.release()
                st.error("Could not open the stream. Check the URL and the camera/RTSP server.")
                return
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        except Exception as exc:
            st.error(f"Could not start the stream: {exc}")
            if cap is not None:
                cap.release()
            return

        weights = _resolve_weights_path()
        if not weights and not st.session_state.get("infer_demo_mode"):
            st.error("No model selected. Choose a checkpoint or enable Demo mode.")
            cap.release()
            return

        adapter = None
        solution = None
        try:
            if weights:
                adapter = _get_adapter(weights)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30
                solution = _create_solution(
                    st.session_state.get("selected_solution", "None (Detection Only)"),
                    adapter,
                    st.session_state.get("_live_tracker"),
                    fps,
                )
        except Exception as exc:
            st.error(f"Could not load the model: {exc}")
            cap.release()
            return

        output_path = None
        writer = None
        if st.session_state.get("live_save_output"):
            try:
                from flashstudio.pages.inference.run.video import _get_output_path
                output_path = _get_output_path("live_output.mp4")
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                writer = cv2.VideoWriter(
                    output_path, cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0),
                    (width, height),
                )
                if not writer.isOpened():
                    writer = None
                    output_path = None
            except Exception:
                writer = None
                output_path = None

        st.session_state.update({
            "live_running": True,
            _live_key("cap"): cap,
            _live_key("adapter"): adapter,
            _live_key("weights"): weights,
            _live_key("solution"): solution,
            _live_key("writer"): writer,
            _live_key("output"): output_path,
            _live_key("frames"): 0,
            _live_key("dets"): 0,
            _live_key("started"): time.time(),
            _live_key("last"): None,
        })
        st.rerun()

    if not running:
        last = st.session_state.get(_live_key("last"))
        if last is not None:
            st.image(last, caption="Last live frame", use_container_width=True)
        _render_multi_camera()
        return

    cap = st.session_state.get(_live_key("cap"))
    if cap is None:
        _cleanup_live()
        return

    import cv2

    with st.container(border=True):
        st.markdown("#### Live preview")
        placeholder = st.empty()
        metrics = st.empty()
    adapter = st.session_state.get(_live_key("adapter"))
    solution = st.session_state.get(_live_key("solution"))
    writer = st.session_state.get(_live_key("writer"))
    weights = st.session_state.get(_live_key("weights"))
    frames_total = int(st.session_state.get(_live_key("frames")) or 0)
    dets_total = int(st.session_state.get(_live_key("dets")) or 0)
    started = float(st.session_state.get(_live_key("started")) or time.time())
    skip = max(1, int(skip))
    limit = int(max_frames or 0)

    last_frame = None
    t0 = time.perf_counter()
    processed_this_run = 0
    target = 12  # frames per rerun keeps the UI responsive

    for _ in range(target):
        if limit and frames_total >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frames_total % skip != 0:
            frames_total += 1
            continue
        try:
            if solution is not None:
                annotated, results = solution.process_frame(frame)
                frame_out = annotated if annotated is not None else frame
                count = len(results) if isinstance(results, (list, tuple)) else 0
            elif adapter is not None:
                results = adapter(frame)
                frame_out = _draw_boxes_cv2(
                    frame, results, getattr(adapter, "class_names", None)
                )
                count = len(results)
            elif st.session_state.get("infer_demo_mode"):
                from PIL import Image
                from flashstudio.pages.inference.run.detection import _draw_boxes
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                dets = _detect_demo(pil)
                pil = _draw_boxes(pil, dets)
                frame_out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                count = len(dets)
            else:
                break
            dets_total += count
            if writer is not None:
                writer.write(frame_out)
            last_frame = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            st.warning(f"Frame processing stopped: {exc}")
            break
        frames_total += 1
        processed_this_run += 1

    elapsed = max(time.perf_counter() - t0, 1e-6)
    if last_frame is not None:
        placeholder.image(last_frame, channels="RGB", use_container_width=True)
        st.session_state[_live_key("last")] = last_frame
    fps_live = processed_this_run / elapsed
    metrics.markdown(
        f'<div class="ds-card-stats">'
        f'<span>Frames: <b>{frames_total}</b></span>'
        f'<span>Detections: <b>{dets_total}</b></span>'
        f'<span>Live FPS: <b>{fps_live:.1f}</b></span>'
        f'<span>Uptime: <b>{time.time() - started:.0f}s</b></span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.session_state[_live_key("frames")] = frames_total
    st.session_state[_live_key("dets")] = dets_total

    if limit and frames_total >= limit:
        output = st.session_state.get(_live_key("output"))
        _cleanup_live()
        if output and os.path.isfile(output):
            st.success(f"Live capture saved to `{output}`")
        else:
            st.success("Live capture finished.")
        return

    # Keep the stream moving without blocking the Stop button for long.
    st.rerun()


def _render_multi_camera():
    """Run a short inference pass over several streams and show a grid."""
    with st.expander("Multi-camera grid — sample several streams at once"):
        st.caption("Webcam index, RTSP URL, or local video — one per line.")
        sources_text = st.text_area(
            "Sources",
            placeholder="0\nrtsp://user:pass@host:554/stream1\n/tmp/opencode/live_test.mp4",
            key="grid_sources",
            height=90,
        )
        c1, c2 = st.columns([1, 3])
        with c1:
            frames = st.number_input(
                "Frames per camera", 1, 500, 20, 5, key="grid_frames",
            )
        with c2:
            run = st.button("Run grid", type="primary", key="grid_run_btn")
        if not run:
            return

        sources = [s.strip() for s in sources_text.splitlines() if s.strip()]
        if not sources:
            st.warning("Add at least one source.")
            return
        weights = _resolve_weights_path()
        if not weights and not st.session_state.get("infer_demo_mode"):
            st.error("No model selected. Choose a checkpoint or enable Demo mode.")
            return

        try:
            import cv2

            from flashstudio.pages.inference.run.detection import _draw_boxes
            from PIL import Image

            adapter = _get_adapter(weights) if weights else None
        except Exception as exc:
            st.error(f"Could not prepare the grid: {exc}")
            return

        results = []
        progress = st.progress(0)
        for idx, source in enumerate(sources):
            cap = None
            try:
                cap = cv2.VideoCapture(
                    int(source) if source.isdigit() else source
                )
                if not cap.isOpened():
                    results.append((source, None, 0, "Could not open stream"))
                    continue
                last = None
                dets = 0
                read = 0
                while read < int(frames):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    read += 1
                    if adapter is not None:
                        raw = adapter(frame)
                        dets += len(raw)
                        frame = _draw_boxes_cv2(
                            frame, raw, getattr(adapter, "class_names", None)
                        )
                    elif st.session_state.get("infer_demo_mode"):
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(rgb)
                        from flashstudio.pages.inference.run.detection import _detect_demo
                        demo_dets = _detect_demo(pil)
                        dets += len(demo_dets)
                        pil = _draw_boxes(pil, demo_dets)
                        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results.append((source, last, dets, f"{read} frames"))
            except Exception as exc:
                results.append((source, None, 0, str(exc)))
            finally:
                if cap is not None:
                    cap.release()
            progress.progress((idx + 1) / len(sources))
        progress.empty()

        cols_per_row = 2
        for start in range(0, len(results), cols_per_row):
            cols = st.columns(cols_per_row)
            for col, item in zip(cols, results[start:start + cols_per_row]):
                source, frame, dets, note = item
                with col:
                    if frame is not None:
                        st.image(frame, channels="RGB", use_container_width=True)
                    st.caption(f"**{source}** · {note} · {dets} detections")
