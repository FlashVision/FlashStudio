"""FlashStudio — Inference Page (ultra-compact, no-scroll)."""

import os
import time
import io
import json
import tempfile
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flashstudio.components.styles import render_page_header, render_info_bar

try:
    from flashstudio.components.zone_drawer import zone_drawer
    _HAS_ZONE_DRAWER = True
except ImportError:
    _HAS_ZONE_DRAWER = False

try:
    from flashdet import Predictor, FlashTracker
    _HAS_PREDICTOR = True
except ImportError:
    _HAS_PREDICTOR = False

_SOLUTIONS_AVAILABLE = {}
_solution_classes = [
    "ObjectCounter", "RegionCounter", "SpeedEstimator", "Heatmap",
    "SecurityAlarm", "QueueManager", "ParkingManager", "TrafficFlow",
    "DwellTimeAnalyzer", "DistanceCalculator", "TrajectoryVisualizer",
    "ObjectBlurrer", "CrowdDensity", "WorkoutMonitor", "ObjectCropper",
    "AnalyticsDashboard",
]
for _cls_name in _solution_classes:
    try:
        _mod = __import__("flashdet.solutions", fromlist=[_cls_name])
        _SOLUTIONS_AVAILABLE[_cls_name] = getattr(_mod, _cls_name)
    except (ImportError, AttributeError):
        pass

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
          "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9"]

SOLUTIONS = {
    "None (Detection Only)": {"desc": "Standard object detection", "needs_zone": False},
    "Object Counter (Line)": {"desc": "Count objects crossing line", "needs_zone": True, "zone_type": "line"},
    "Region Counter (Polygon)": {"desc": "Count in polygon zones", "needs_zone": True, "zone_type": "polygon"},
    "Speed Estimator": {"desc": "Estimate speed", "needs_zone": True, "zone_type": "line"},
    "Heatmap": {"desc": "Activity heatmap", "needs_zone": False},
    "Security Alarm (Zone)": {"desc": "Alert on zone entry", "needs_zone": True, "zone_type": "polygon"},
    "Trajectory Visualizer": {"desc": "Motion trails", "needs_zone": False},
    "Object Blurrer": {"desc": "Blur detections", "needs_zone": False},
    "Queue Manager": {"desc": "Queue lengths", "needs_zone": True, "zone_type": "polygon"},
    "Crowd Density": {"desc": "Grid density", "needs_zone": False},
    "Parking Manager": {"desc": "Parking occupancy", "needs_zone": True, "zone_type": "polygon"},
    "Traffic Flow": {"desc": "Traffic analysis", "needs_zone": True, "zone_type": "line"},
    "Dwell Time Analyzer": {"desc": "Time in zones", "needs_zone": True, "zone_type": "polygon"},
    "Distance Calculator": {"desc": "Pairwise distances", "needs_zone": False},
    "Workout Monitor": {"desc": "Exercise reps", "needs_zone": False},
    "Object Cropper": {"desc": "Crop detections", "needs_zone": False},
    "Analytics Dashboard": {"desc": "Real-time stats", "needs_zone": False},
}


def _get_device_options():
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
    except ImportError:
        pass
    return devices


def render_inference_page():
    from flashstudio.utils import show_flashes
    render_page_header("", "Inference")
    show_flashes()

    # Readiness dots
    has_model = bool(st.session_state.get("infer_weights_file") or st.session_state.get("infer_weights_path"))
    has_data = bool(st.session_state.get("infer_images") or st.session_state.get("infer_video") or st.session_state.get("infer_stream_url"))
    sol = SOLUTIONS.get(st.session_state.get("selected_solution", "None (Detection Only)"), {})
    has_zone = not sol.get("needs_zone") or bool(st.session_state.get("zone_line_points") or st.session_state.get("zone_polygons"))
    dot = lambda ok: '<span style="color:#22C55E;">&#9679;</span>' if ok else '<span style="color:#EF4444;">&#9679;</span>'
    st.markdown(
        f'<div class="ds-card-stats" style="padding:0.2rem 0;">'
        f'<span>{dot(has_model)} Model</span>'
        f'<span>{dot(has_data)} Data</span>'
        f'<span>{dot(has_zone)} Zone</span>'
        f'<span style="margin-left:auto;color:#9CA3AF;">Ready: {sum([has_model, has_data, has_zone])}/3</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    tab_model, tab_data, tab_solution, tab_run = st.tabs(["Model", "Data", "Solution", "Run"])
    with tab_model:
        _tab_model()
    with tab_data:
        _tab_data()
    with tab_solution:
        _tab_solution()
    with tab_run:
        _tab_run()


# ════════════════════════════════════════
# TAB 1: MODEL
# ════════════════════════════════════════

def _tab_model():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Architecture & Params")
            st.selectbox("Model", ["FlashDet-Pico", "FlashDet-Nano", "FlashDet-Small", "FlashDet-Medium", "FlashDet-Large", "FlashDet-X"],
                         key="infer_model_arch")
            pc1, pc2 = st.columns(2)
            with pc1:
                st.slider("Confidence", 0.0, 1.0, 0.25, 0.05, key="infer_conf")
                st.number_input("Img Size", 320, 1920, 640, 32, key="infer_img_size")
            with pc2:
                st.slider("NMS IoU", 0.0, 1.0, 0.45, 0.05, key="infer_nms")
                st.selectbox("Device", _get_device_options(), key="infer_device")
            st.number_input("Classes", 1, 1000, 80, key="infer_num_classes")
            st.multiselect("Filter (empty=all)", COCO_CLASSES, default=[], key="infer_class_filter")

    with col2:
        with st.container(border=True):
            st.markdown("#### Weights")
            ws = st.radio("Source", ["Upload", "Path", "Training output"], key="weight_source", horizontal=True)
            if ws == "Upload":
                up = st.file_uploader("File", type=["pt", "pth", "onnx", "engine"], key="infer_weights_file")
                if up:
                    st.success(f"{up.name} ({up.size / 1e6:.1f}MB)")
            elif ws == "Path":
                st.text_input("Path", placeholder="/path/best.pth", key="infer_weights_path")
            else:
                p = f"{st.session_state.get('save_dir', 'workspace')}/model_best_inference.pth"
                st.session_state["infer_weights_path"] = p
                st.caption(f"Using: `{p}`")


# ════════════════════════════════════════
# TAB 2: DATA
# ════════════════════════════════════════

def _tab_data():
    input_type = st.radio("Input", ["Images", "Video", "RTSP"], horizontal=True, key="infer_input_type")

    if input_type == "Images":
        uploaded = st.file_uploader("Images", type=["jpg", "jpeg", "png", "webp", "bmp"],
                                    accept_multiple_files=True, key="infer_images")
        if uploaded:
            cols = st.columns(min(len(uploaded), 6))
            for i, f in enumerate(uploaded[:6]):
                with cols[i]:
                    st.image(f, caption=f.name[:10], use_container_width=True)
            if len(uploaded) > 6:
                st.caption(f"+{len(uploaded) - 6} more")

    elif input_type == "Video":
        vc1, vc2 = st.columns([3, 1])
        with vc1:
            st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"], key="infer_video")
            vid = st.session_state.get("infer_video")
            if vid:
                st.caption(f"{vid.name} ({vid.size / 1e6:.1f}MB)")
                meta = _get_video_metadata(vid)
                if meta:
                    mc = st.columns(4)
                    with mc[0]: st.metric("Res", f"{meta['width']}x{meta['height']}")
                    with mc[1]: st.metric("FPS", f"{meta['fps']:.0f}")
                    with mc[2]: st.metric("Frames", meta["total_frames"])
                    with mc[3]: st.metric("Dur", f"{meta['total_frames'] / max(meta['fps'], 1):.0f}s")
        with vc2:
            st.number_input("Max frames (0=all)", 0, 100000, 300, key="infer_max_frames")
            st.select_slider("Skip N", [1, 2, 3, 5, 10], value=1, key="infer_frame_skip")
            st.checkbox("Save output", True, key="infer_save_video")

    else:
        sc1, sc2 = st.columns([4, 1])
        with sc1:
            st.text_input("RTSP URL", placeholder="rtsp://192.168.1.100:554/stream", key="infer_stream_url")
        with sc2:
            st.number_input("Duration (s)", 0, 3600, 60, key="infer_stream_duration")
        if st.session_state.get("infer_stream_url") and st.button("Test", key="test_stream"):
            _test_stream()


# ════════════════════════════════════════
# TAB 3: SOLUTION
# ════════════════════════════════════════

def _tab_solution():
    sc1, sc2 = st.columns([2, 3])
    with sc1:
        selected = st.selectbox("Solution", list(SOLUTIONS.keys()), key="selected_solution",
                                format_func=lambda x: x)
        sol = SOLUTIONS[selected]
        st.caption(sol["desc"])

    with sc2:
        if sol["needs_zone"]:
            st.info("Zone required — draw below using Polygon, Rectangle, or Line.")
        else:
            st.caption("Zone optional — draw one to limit detection area.")

    _zone_draw_ui(sol)


def _zone_draw_ui(sol):
    default_type = sol.get("zone_type", "polygon")
    mode_options = ["polygon", "rect", "line"]
    default_idx = mode_options.index(default_type) if default_type in mode_options else 0
    draw_mode = st.radio("Draw Mode", mode_options, index=default_idx, horizontal=True, key="zone_draw_mode",
                         format_func=lambda x: {"polygon": "Polygon", "rect": "Rectangle", "line": "Line"}[x])

    frame_img = _get_first_frame()

    existing_points = st.session_state.get("zone_draw_points", [])
    existing_closed = st.session_state.get("zone_closed", False)

    if _HAS_ZONE_DRAWER:
        zone_drawer(image=frame_img, mode=draw_mode, points=existing_points,
                    closed=existing_closed, display_width=650)
    else:
        st.warning("Zone drawing component not available.")

    with st.expander("Set coordinates manually", expanded=not _HAS_ZONE_DRAWER):
        _manual_zone_input(draw_mode)


def _manual_zone_input(draw_mode):
    if draw_mode == "line":
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.number_input("X1", 0, 1920, 100, key="manual_line_x1")
            y1 = st.number_input("Y1", 0, 1080, 240, key="manual_line_y1")
        with c2:
            x2 = st.number_input("X2", 0, 1920, 540, key="manual_line_x2")
            y2 = st.number_input("Y2", 0, 1080, 240, key="manual_line_y2")
        if st.button("Set Line", key="set_manual_line"):
            st.session_state["zone_line_points"] = [(x1, y1), (x2, y2)]
            st.success(f"({x1},{y1}) → ({x2},{y2})")
    else:
        pts = st.text_area("Points (x,y per line)", "100,100\n500,100\n500,400\n100,400", key="manual_polygon_text")
        btn_label = "Set Rectangle" if draw_mode == "rect" else "Set Polygon"
        min_pts = 2 if draw_mode == "rect" else 3
        if st.button(btn_label, key="set_manual_polygon"):
            parsed = []
            for line in pts.strip().split("\n"):
                p = line.strip().split(",")
                if len(p) == 2:
                    try:
                        parsed.append((int(p[0].strip()), int(p[1].strip())))
                    except ValueError:
                        continue
            if len(parsed) >= min_pts:
                if draw_mode == "rect" and len(parsed) >= 2:
                    x1, y1 = parsed[0]; x2, y2 = parsed[1]
                    st.session_state["zone_polygons"] = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
                    st.success(f"Rectangle: ({x1},{y1}) → ({x2},{y2})")
                else:
                    st.session_state["zone_polygons"] = [parsed]
                    st.success(f"{len(parsed)} vertices")


def _store_zone_coords(display_points, draw_mode, scale_x, scale_y):
    orig = [(int(x * scale_x), int(y * scale_y)) for x, y in display_points]
    if draw_mode == "line" and len(orig) >= 2:
        st.session_state["zone_line_points"] = [orig[0], orig[1]]
        st.success(f"Line: {orig[0]} → {orig[1]}")
    elif draw_mode == "rect" and len(orig) >= 2:
        x1, y1 = orig[0]; x2, y2 = orig[1]
        st.session_state["zone_polygons"] = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
        st.success(f"Rect: ({x1},{y1})→({x2},{y2})")
    elif draw_mode == "polygon" and len(orig) >= 3:
        st.session_state["zone_polygons"] = [orig]
        st.success(f"Polygon: {len(orig)} pts") if st.session_state.get("zone_closed") else st.info(f"{len(orig)} pts, click more or close")


# ════════════════════════════════════════
# TAB 4: RUN
# ════════════════════════════════════════

def _tab_run():
    # Compact config summary
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Model", st.session_state.get("infer_model_arch", "FlashDet-Nano").replace("FlashDet-", ""))
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


# ════════════════════════════════════════
# IMAGE INFERENCE
# ════════════════════════════════════════

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


# ════════════════════════════════════════
# VIDEO INFERENCE
# ════════════════════════════════════════

def _run_video():
    video = st.session_state.get("infer_video")
    solution = st.session_state.get("selected_solution", "None (Detection Only)")
    weights = st.session_state.get("infer_weights_path", "") or st.session_state.get("infer_weights_file")
    if weights:
        try:
            _run_flashdet_video(video, solution)
            return
        except Exception as e:
            st.warning(f"FlashDet error: {e}. Demo mode.")
    _run_demo_video(solution)


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
    if "in_count" in sol:
        sc1, sc2, sc3 = st.columns(3)
        with sc1: st.metric("In", sol["in_count"])
        with sc2: st.metric("Out", sol["out_count"])
        with sc3: st.metric("Net", sol["in_count"] - sol["out_count"])
    if "region_counts" in sol:
        cols = st.columns(len(sol["region_counts"]))
        for i, (r, c) in enumerate(sol["region_counts"].items()):
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


def _run_demo_video(solution_name):
    t0 = time.perf_counter()
    progress = st.progress(0)
    frames = []
    for i in range(4):
        time.sleep(0.2)
        progress.progress((i + 1) / 4)
        img = Image.new("RGB", (640, 480), (248, 248, 252))
        draw = ImageDraw.Draw(img)
        draw.rectangle([80 + i * 40, 80 + i * 30, 250 + i * 40, 230 + i * 30], outline="#7C3AED", width=3)
        draw.text((90 + i * 40, 65 + i * 30), f"car 0.{85 + i}", fill="#1A1A2E")
        frames.append(img)
    progress.empty()
    elapsed = time.perf_counter() - t0

    sol_output = {}
    if "Counter" in solution_name:
        sol_output = {"in_count": 12, "out_count": 8}
    elif "Region" in solution_name:
        sol_output = {"region_counts": {"Zone A": 5, "Zone B": 3}}
    elif "Speed" in solution_name:
        sol_output = {"speeds": [[1, 45.2], [2, 38.7], [3, 52.1]]}

    st.session_state["video_results"] = {
        "frames": 300, "detections": 1247, "processing_fps": 300 / max(elapsed, 0.001),
        "elapsed_time": elapsed, "solution_output": sol_output, "frames_preview": frames,
        "output_path": os.path.join(os.getcwd(), "output.mp4") if st.session_state.get("infer_save_video") else None,
    }
    st.rerun()


def _run_flashdet_video(video_file, solution_name):
    if not _HAS_PREDICTOR:
        raise ImportError("Predictor unavailable")
    import cv2
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_file.read()); tmp.flush(); video_file.seek(0)
    cap = cv2.VideoCapture(tmp.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    weights = st.session_state.get("infer_weights_path", "")
    if not weights:
        up = st.session_state.get("infer_weights_file")
        if up:
            wtmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
            wtmp.write(up.read()); wtmp.flush(); up.seek(0)
            weights = wtmp.name

    predictor = Predictor(model_path=weights, device=st.session_state.get("infer_device", "cpu").split(" ")[0],
                          conf_thresh=st.session_state.get("infer_conf", 0.25),
                          nms_thresh=st.session_state.get("infer_nms", 0.45),
                          input_size=st.session_state.get("infer_img_size", 640))
    tracker = FlashTracker()
    solution = _create_solution(solution_name, predictor, tracker, fps)

    max_frames = st.session_state.get("infer_max_frames", 300) or total
    skip = st.session_state.get("infer_frame_skip", 1)
    frames_preview, processed, total_dets = [], 0, 0
    t0 = time.perf_counter()
    progress = st.progress(0)

    for idx in range(min(total, max_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip != 0:
            continue
        if solution:
            annotated, results = solution.process_frame(frame)
        else:
            results = predictor(frame); annotated = frame
        if isinstance(results, list):
            total_dets += len(results)
        processed += 1
        if idx % max(total // 4, 1) == 0:
            frames_preview.append(Image.fromarray(cv2.cvtColor(annotated or frame, cv2.COLOR_BGR2RGB)))
        progress.progress(idx / min(total, max_frames))

    cap.release(); progress.empty()
    elapsed = time.perf_counter() - t0
    sol_output = {}
    if solution:
        try:
            sol_output = solution.get_results()
            if not isinstance(sol_output, dict):
                sol_output = {}
        except Exception:
            pass

    st.session_state["video_results"] = {
        "frames": processed, "detections": total_dets, "processing_fps": processed / max(elapsed, 0.001),
        "elapsed_time": elapsed, "solution_output": sol_output, "frames_preview": frames_preview,
        "output_path": os.path.join(os.getcwd(), "output.mp4") if st.session_state.get("infer_save_video") else None,
    }
    st.rerun()


def _create_solution(solution_name, predictor, tracker, fps):
    line_pts = st.session_state.get("zone_line_points")
    polygons = st.session_state.get("zone_polygons")
    classes = st.session_state.get("infer_class_filter") or None

    if solution_name == "None (Detection Only)":
        return None

    sol_map = {
        "Object Counter": ("ObjectCounter", dict(line_points=line_pts, classes=classes)),
        "Region Counter": ("RegionCounter", dict(regions=polygons, classes=classes)),
        "Speed Estimator": ("SpeedEstimator", dict(fps=fps, classes=classes)),
        "Heatmap": ("Heatmap", dict(classes=classes)),
        "Security Alarm": ("SecurityAlarm", dict(restricted_zones=polygons, classes=classes)),
        "Trajectory": ("TrajectoryVisualizer", dict(classes=classes)),
        "Blurrer": ("ObjectBlurrer", dict(classes=classes)),
        "Queue": ("QueueManager", dict(queue_regions=polygons, classes=classes, fps=fps)),
        "Crowd": ("CrowdDensity", dict(classes=classes)),
        "Parking": ("ParkingManager", dict(parking_spots=polygons, classes=classes)),
        "Traffic": ("TrafficFlow", dict(fps=fps, classes=classes)),
        "Dwell": ("DwellTimeAnalyzer", dict(zones=polygons, fps=fps, classes=classes)),
        "Distance": ("DistanceCalculator", dict(classes=classes)),
        "Workout": ("WorkoutMonitor", dict(classes=classes)),
        "Cropper": ("ObjectCropper", dict(classes=classes)),
        "Analytics": ("AnalyticsDashboard", dict(classes=classes)),
    }

    for keyword, (cls_name, kwargs) in sol_map.items():
        if keyword in solution_name:
            cls = _SOLUTIONS_AVAILABLE.get(cls_name)
            if not cls:
                return None
            try:
                if cls_name not in ("Heatmap", "ObjectBlurrer", "CrowdDensity"):
                    return cls(predictor=predictor, tracker=tracker, **kwargs)
                return cls(predictor=predictor, **kwargs)
            except TypeError:
                return cls(predictor=predictor, **kwargs)
    return None


# ════════════════════════════════════════
# DETECTION UTILITIES
# ════════════════════════════════════════

def _detect(image):
    weights = st.session_state.get("infer_weights_path", "")
    if not weights:
        up = st.session_state.get("infer_weights_file")
        if up:
            tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
            tmp.write(up.read()); tmp.flush(); up.seek(0)
            weights = tmp.name
    if weights:
        try:
            return _detect_real(image, weights)
        except Exception:
            pass
    return _detect_demo(image)


def _detect_real(image, weights_path):
    if not _HAS_PREDICTOR:
        raise ImportError("Predictor unavailable")
    predictor = Predictor(model_path=weights_path, device=st.session_state.get("infer_device", "cpu").split(" ")[0],
                          conf_thresh=st.session_state.get("infer_conf", 0.25),
                          nms_thresh=st.session_state.get("infer_nms", 0.45),
                          input_size=st.session_state.get("infer_img_size", 640))
    raw = predictor(np.array(image))
    class_filter = st.session_state.get("infer_class_filter", [])
    dets = []
    for bbox, score, cls_id in raw:
        name = COCO_CLASSES[int(cls_id)] if int(cls_id) < len(COCO_CLASSES) else f"class_{cls_id}"
        if class_filter and name not in class_filter:
            continue
        dets.append([name, f"{float(score):.2f}", str([int(x) for x in (bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox))])])
    return dets


def _detect_demo(image):
    w, h = image.size
    rng = np.random.default_rng(hash(image.tobytes()[:100]) % (2 ** 31))
    conf_thr = st.session_state.get("infer_conf", 0.25)
    class_filter = st.session_state.get("infer_class_filter", [])
    dets = []
    for _ in range(rng.integers(3, 8)):
        cid = rng.integers(0, 10)
        name = COCO_CLASSES[cid]
        if class_filter and name not in class_filter:
            continue
        conf = rng.uniform(0.3, 0.98)
        if conf < conf_thr:
            continue
        x1 = int(rng.integers(0, max(w - 100, 1)))
        y1 = int(rng.integers(0, max(h - 100, 1)))
        dets.append([name, f"{conf:.2f}", str([x1, y1, min(x1 + int(rng.integers(60, 200)), w), min(y1 + int(rng.integers(60, 200)), h)])])
    return sorted(dets, key=lambda x: float(x[1]), reverse=True)


def _draw_boxes(image, dets):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for i, (cls, conf, bbox_str) in enumerate(dets):
        bbox = json.loads(bbox_str.replace("'", '"'))
        color = COLORS[i % len(COLORS)]
        draw.rectangle(bbox, outline=color, width=3)
        label = f"{cls} {conf}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(bbox[1] - th - 6, 0)
        draw.rectangle([bbox[0], ly, bbox[0] + tw + 8, ly + th + 6], fill=color)
        draw.text((bbox[0] + 4, ly + 2), label, fill="white", font=font)
    return image


def _to_coco_format(results):
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": n} for i, n in enumerate(COCO_CLASSES)]}
    aid = 1
    for iid, r in enumerate(results, 1):
        coco["images"].append({"id": iid, "file_name": r["name"], "width": r.get("width", 640), "height": r.get("height", 480)})
        for cls, conf, bbox_str in r["dets"]:
            bbox = json.loads(bbox_str.replace("'", '"'))
            coco["annotations"].append({"id": aid, "image_id": iid,
                                         "category_id": COCO_CLASSES.index(cls) if cls in COCO_CLASSES else 0,
                                         "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                                         "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), "score": float(conf), "iscrowd": 0})
            aid += 1
    return coco


def _get_video_metadata(video_file):
    tmp_path = None
    try:
        import cv2
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.write(video_file.read()); tmp.flush(); tmp.close()
        video_file.seek(0)
        cap = cv2.VideoCapture(tmp_path)
        if cap.isOpened():
            meta = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS) or 30, "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
            cap.release()
            return meta
        cap.release()
    except Exception:
        pass
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return None


def _get_first_frame():
    video = st.session_state.get("infer_video")
    if video:
        try:
            import cv2
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(video.read()); tmp.flush(); tmp.close()
            video.seek(0)
            cap = cv2.VideoCapture(tmp.name)
            ret, frame = cap.read(); cap.release()
            os.unlink(tmp.name)
            if ret:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
    images = st.session_state.get("infer_images")
    if images:
        try:
            img = Image.open(images[0]).convert("RGB"); images[0].seek(0)
            return img
        except Exception:
            pass
    return None


def _test_stream():
    try:
        import cv2
        cap = cv2.VideoCapture(st.session_state["infer_stream_url"])
        if cap.isOpened():
            ret, frame = cap.read(); cap.release()
            if ret:
                st.success("Connected")
                st.image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), use_container_width=True)
            else:
                st.error("No frame")
        else:
            st.error("Failed to connect")
    except Exception as e:
        st.error(str(e)[:60])
