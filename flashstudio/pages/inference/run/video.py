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
    _get_class_names, _SOLUTIONS_AVAILABLE,
)
from flashstudio.pages.inference.run.detection import (
    _draw_boxes_cv2, _resolve_weights_path, _get_adapter,
)

try:
    from flashdet import FlashTracker
except ImportError:
    FlashTracker = None


def _run_video():
    video = st.session_state.get("infer_video")
    solution = st.session_state.get("selected_solution", "None (Detection Only)")
    weights = _resolve_weights_path()

    if weights:
        try:
            _run_flashdet_video(video, solution, weights)
            return
        except Exception as e:
            st.error(f"Inference failed: {e}")
            if st.session_state.get("infer_demo_mode"):
                st.warning("Demo mode is enabled — showing fake results.")
                _run_demo_video(solution)
            return

    if st.session_state.get("infer_demo_mode"):
        st.warning("No model selected — showing demo results.")
        _run_demo_video(solution)
        return

    st.error(
        "No model selected. Upload a checkpoint or enable Demo mode in Inference → Model."
    )


def _run_demo_video(solution_name):
    import cv2

    t0 = time.perf_counter()
    progress = st.progress(0)
    frames = []
    for i in range(4):
        time.sleep(0.2)
        progress.progress((i + 1) / 4)
        img = Image.new("RGB", INFER_DEFAULT_RESOLUTION, (248, 248, 252))
        draw = ImageDraw.Draw(img)
        draw.rectangle([80 + i * 40, 80 + i * 30, 250 + i * 40, 230 + i * 30],
                       outline=COLOR_PRIMARY, width=3)
        draw.text((90 + i * 40, 65 + i * 30), f"car 0.{85 + i}", fill=COLOR_PRIMARY)
        frames.append(img)
    progress.empty()
    elapsed = time.perf_counter() - t0

    output_path = None
    if st.session_state.get("infer_save_video"):
        output_path = _write_frames_video(frames, fps=5)
        if output_path:
            st.caption(f"Saved demo output to `{output_path}`")

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
        "output_path": output_path, "demo": True,
    }
    st.rerun()


def _write_frames_video(frames, fps=30):
    """Write a list of PIL frames to an mp4 file. Returns the path or None."""
    if not frames:
        return None
    try:
        import cv2
        import numpy as np
        path = _get_output_path("output.mp4")
        w, h = frames[0].size
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0), (w, h)
        )
        if not writer.isOpened():
            return None
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame.convert("RGB")), cv2.COLOR_RGB2BGR))
        writer.release()
        return path if os.path.isfile(path) and os.path.getsize(path) > 0 else None
    except Exception:
        return None


def _get_output_path(filename):
    """Resolve output path: use the configured save dir, else tempdir."""
    sd = st.session_state.get("save_dir", "")
    if sd and os.path.isdir(sd):
        return os.path.join(sd, filename)
    return os.path.join(tempfile.gettempdir(), filename)


def _run_flashdet_video(video_file, solution_name, weights):
    import cv2

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_file.read())
    tmp.flush()
    tmp.close()
    video_file.seek(0)
    cap = cv2.VideoCapture(tmp.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or INFER_DEFAULT_RESOLUTION[0]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or INFER_DEFAULT_RESOLUTION[1]

    predictor = _get_adapter(weights)
    tracker = FlashTracker() if FlashTracker is not None else None
    solution = _create_solution(solution_name, predictor, tracker, fps)

    requested_max = st.session_state.get("infer_max_frames", INFER_MAX_FRAMES)
    try:
        requested_max = int(requested_max or 0)
    except (TypeError, ValueError):
        requested_max = 0
    limit = total if total > 0 else (requested_max or INFER_MAX_FRAMES)
    if requested_max > 0:
        limit = min(limit, requested_max)

    skip = max(1, int(st.session_state.get("infer_frame_skip", INFER_FRAME_SKIP) or 1))
    save_output = bool(st.session_state.get("infer_save_video"))
    output_path = None
    writer = None
    if save_output:
        output_path = _get_output_path("output.mp4")
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), max(float(fps), 1.0),
            (width, height),
        )
        if not writer.isOpened():
            writer = None
            output_path = None

    frames_preview, processed, total_dets = [], 0, 0
    t0 = time.perf_counter()
    progress = st.progress(0)
    idx = 0

    try:
        while idx < limit and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % skip == 0:
                if solution:
                    annotated, results = solution.process_frame(frame)
                else:
                    results = predictor(frame)
                    annotated = _draw_boxes_cv2(frame, results)
                frame_out = annotated if annotated is not None else frame
                if isinstance(results, (list, tuple)):
                    total_dets += len(results)
                processed += 1
                if writer is not None:
                    writer.write(frame_out)
                if limit > 0 and idx % max(limit // 4, 1) == 0:
                    frames_preview.append(
                        Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
                    )
                progress.progress(min(processed / max(limit // skip, 1), 1.0))
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        progress.empty()
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

    if writer is not None and output_path and not os.path.isfile(output_path):
        output_path = None

    st.session_state["video_results"] = {
        "frames": processed, "detections": total_dets,
        "processing_fps": processed / max(elapsed, 0.001),
        "elapsed_time": elapsed, "solution_output": sol_output,
        "frames_preview": frames_preview,
        "output_path": output_path,
        "demo": False,
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

    sol_map = {
        "Object Counter": ("ObjectCounter", dict(line_points=line_pts, classes=classes)),
        "Region Counter": ("RegionCounter", dict(regions={f"zone_{i}": p for i, p in enumerate(polygons)}, classes=classes)),
        "Speed Estimator": ("SpeedEstimator", dict(fps=fps, classes=classes)),
        "Heatmap": ("Heatmap", dict(classes=classes)),
        "Security Alarm": ("SecurityAlarm", dict(restricted_zones=polygons or [], classes=classes)),
        "Trajectory": ("TrajectoryVisualizer", dict(classes=classes)),
        "Blurrer": ("ObjectBlurrer", dict(classes=classes)),
        "Queue": ("QueueManager", dict(queue_regions=polygons or [], classes=classes, fps=fps)),
        "Crowd": ("CrowdDensity", dict(classes=classes)),
        "Parking": ("ParkingManager", dict(parking_spots=polygons or [], classes=classes)),
        "Traffic": ("TrafficFlow", dict(fps=fps, classes=classes)),
        "Dwell": ("DwellTimeAnalyzer", dict(zones={f"zone_{i}": p for i, p in enumerate(polygons)}, fps=fps, classes=classes)),
        "Distance": ("DistanceCalculator", dict(classes=classes)),
        "Workout": ("WorkoutMonitor", dict(classes=classes)),
        "Cropper": ("ObjectCropper", dict(classes=classes)),
        "Analytics": ("AnalyticsDashboard", dict(class_names=_get_class_names())),
    }

    for keyword, (cls_name, kwargs) in sol_map.items():
        if keyword not in solution_name:
            continue
        cls = _SOLUTIONS_AVAILABLE.get(cls_name)
        if not cls:
            return None
        try:
            return cls(predictor=predictor, tracker=tracker, **kwargs)
        except TypeError:
            try:
                return cls(predictor=predictor, **kwargs)
            except TypeError as exc:
                st.warning(f"{cls_name} is incompatible with the installed FlashDet: {exc}")
                return None
    return None
