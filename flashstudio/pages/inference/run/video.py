"""Inference — Video processing utilities."""

import os
import time
import tempfile
import streamlit as st
from PIL import Image, ImageDraw
from flashstudio.constants import (
    COCO_CLASSES, INFER_CONF_THRESHOLD, INFER_NMS_THRESHOLD,
    INFER_IMG_SIZE, INFER_MAX_FRAMES, INFER_FRAME_SKIP,
    INFER_DEFAULT_RESOLUTION, COLOR_PRIMARY,
)
from flashstudio.pages.inference._common import (
    _get_class_names, _HAS_PREDICTOR, _SOLUTIONS_AVAILABLE,
)
from flashstudio.pages.inference.run.detection import _draw_boxes_cv2

try:
    from flashdet import Predictor, FlashTracker
except ImportError:
    pass


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


def _run_demo_video(solution_name):
    t0 = time.perf_counter()
    progress = st.progress(0)
    frames = []
    for i in range(4):
        time.sleep(0.2)
        progress.progress((i + 1) / 4)
        img = Image.new("RGB", INFER_DEFAULT_RESOLUTION, (248, 248, 252))
        draw = ImageDraw.Draw(img)
        draw.rectangle([80 + i * 40, 80 + i * 30, 250 + i * 40, 230 + i * 30], outline=COLOR_PRIMARY, width=3)
        draw.text((90 + i * 40, 65 + i * 30), f"car 0.{85 + i}", fill=COLOR_PRIMARY)
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
        "output_path": _get_output_path("output.mp4") if st.session_state.get("infer_save_video") else None,
    }
    st.rerun()


def _get_output_path(filename):
    """Resolve output path: prefer save_dir, fall back to tempdir."""
    sd = st.session_state.get("save_dir", "")
    if sd and os.path.isdir(sd):
        return os.path.join(sd, filename)
    return os.path.join(tempfile.gettempdir(), filename)


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

    custom_cls = _get_class_names()
    predictor = Predictor(model_path=weights, device=st.session_state.get("infer_device", "cpu").split(" ")[0],
                          conf_thresh=st.session_state.get("infer_conf", INFER_CONF_THRESHOLD),
                          nms_thresh=st.session_state.get("infer_nms", INFER_NMS_THRESHOLD),
                          input_size=st.session_state.get("infer_img_size", INFER_IMG_SIZE),
                          class_names=custom_cls if custom_cls != COCO_CLASSES else None)
    tracker = FlashTracker()
    solution = _create_solution(solution_name, predictor, tracker, fps)

    max_frames = st.session_state.get("infer_max_frames", INFER_MAX_FRAMES) or total
    skip = st.session_state.get("infer_frame_skip", INFER_FRAME_SKIP)
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
            results = predictor(frame)
            annotated = _draw_boxes_cv2(frame, results)
        if isinstance(results, (list, tuple)):
            total_dets += len(results)
        processed += 1
        if idx % max(total // 4, 1) == 0:
            frames_preview.append(Image.fromarray(cv2.cvtColor(annotated or frame, cv2.COLOR_BGR2RGB)))
        progress.progress(idx / min(total, max_frames))

    cap.release(); progress.empty()
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
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
        "output_path": _get_output_path("output.mp4") if st.session_state.get("infer_save_video") else None,
    }
    st.rerun()


def _create_solution(solution_name, predictor, tracker, fps):
    line_pts = st.session_state.get("zone_line_points")
    polygons = st.session_state.get("zone_polygons")
    class_filter_names = st.session_state.get("infer_class_filter") or None
    classes = None
    if class_filter_names:
        all_cls = _get_class_names()
        classes = [all_cls.index(n) for n in class_filter_names if n in all_cls]
        if not classes:
            classes = None

    if solution_name == "None (Detection Only)":
        return None

    first_poly = polygons[0] if polygons and len(polygons) > 0 else []
    sol_map = {
        "Object Counter": ("ObjectCounter", dict(line_points=line_pts, classes=classes)),
        "Region Counter": ("RegionCounter", dict(regions={"zone_0": first_poly} if first_poly else {}, classes=classes)),
        "Speed Estimator": ("SpeedEstimator", dict(fps=fps, classes=classes)),
        "Heatmap": ("Heatmap", dict(classes=classes)),
        "Security Alarm": ("SecurityAlarm", dict(restricted_zones=polygons or [], classes=classes)),
        "Trajectory": ("TrajectoryVisualizer", dict(classes=classes)),
        "Blurrer": ("ObjectBlurrer", dict(classes=classes)),
        "Queue": ("QueueManager", dict(queue_regions=polygons or [], classes=classes, fps=fps)),
        "Crowd": ("CrowdDensity", dict(classes=classes)),
        "Parking": ("ParkingManager", dict(parking_spots=polygons or [], classes=classes)),
        "Traffic": ("TrafficFlow", dict(fps=fps, classes=classes)),
        "Dwell": ("DwellTimeAnalyzer", dict(zones={"zone_0": first_poly} if first_poly else {}, fps=fps, classes=classes)),
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
