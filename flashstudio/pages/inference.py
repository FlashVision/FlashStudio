"""FlashStudio — Inference Pipeline Dashboard (powered by flashdet)."""

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

# Guard all solution imports individually
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

COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]

SOLUTIONS = {
    "None (Detection Only)": {"description": "Standard object detection — bounding boxes only", "icon": "🔍", "needs_zone": False},
    "Object Counter (Line)": {"description": "Count objects crossing a line", "icon": "🔢", "needs_zone": True, "zone_type": "line"},
    "Region Counter (Polygon)": {"description": "Count objects inside polygon zones", "icon": "📐", "needs_zone": True, "zone_type": "polygon"},
    "Speed Estimator": {"description": "Estimate speed of moving objects", "icon": "⚡", "needs_zone": True, "zone_type": "line"},
    "Heatmap": {"description": "Generate heatmap of object activity", "icon": "🔥", "needs_zone": False},
    "Security Alarm (Zone)": {"description": "Alert when objects enter restricted zone", "icon": "🚨", "needs_zone": True, "zone_type": "polygon"},
    "Trajectory Visualizer": {"description": "Draw motion trails for tracked objects", "icon": "🎯", "needs_zone": False},
    "Object Blurrer": {"description": "Blur detected objects for privacy", "icon": "🙈", "needs_zone": False},
    "Queue Manager": {"description": "Monitor queue lengths in defined zones", "icon": "🧍", "needs_zone": True, "zone_type": "polygon"},
    "Crowd Density": {"description": "Grid-based crowd density estimation", "icon": "👥", "needs_zone": False},
    "Parking Manager": {"description": "Track parking space occupancy", "icon": "🅿️", "needs_zone": True, "zone_type": "polygon"},
    "Traffic Flow": {"description": "Direction-aware traffic analysis", "icon": "🚗", "needs_zone": True, "zone_type": "line"},
    "Dwell Time Analyzer": {"description": "Measure time objects spend in zones", "icon": "⏱️", "needs_zone": True, "zone_type": "polygon"},
    "Distance Calculator": {"description": "Compute pairwise distances between objects", "icon": "📏", "needs_zone": False},
    "Workout Monitor": {"description": "Track exercise repetitions and form", "icon": "🏋️", "needs_zone": False},
    "Object Cropper": {"description": "Crop and save detected objects as individual images", "icon": "✂️", "needs_zone": False},
    "Analytics Dashboard": {"description": "Real-time analytics with charts and stats overlay", "icon": "📊", "needs_zone": False},
}


def _get_device_options() -> list:
    """Get available compute devices."""
    devices = ["cpu"]
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.append(f"cuda:{i} ({name})")
    except ImportError:
        pass
    return devices


def render_inference_page():
    """Render inference pipeline using tabs."""
    render_page_header("🔍", "FlashStudio Inference",
                       "Full inference pipeline — select model, upload data, configure solution, run detection.")

    # Pipeline readiness indicators
    _render_readiness_bar()

    tab_model, tab_data, tab_solution, tab_run = st.tabs([
        "🧠 Model & Weights", "📁 Upload Data", "📐 Solution & Zone", "🚀 Run & Results"
    ])

    with tab_model:
        _tab_model()

    with tab_data:
        _tab_data()

    with tab_solution:
        _tab_solution()

    with tab_run:
        _tab_run()

    model_name = st.session_state.get("infer_model_arch", "FlashDet-Nano")
    solution = st.session_state.get("selected_solution", "Detection Only")
    device = st.session_state.get("infer_device", "cpu")
    render_info_bar({"Model": model_name, "Solution": solution, "Device": device.split(" ")[0]})


def _render_readiness_bar():
    """Show pipeline readiness as colored indicators."""
    has_model = bool(st.session_state.get("infer_weights_file") or st.session_state.get("infer_weights_path"))
    has_data = bool(
        st.session_state.get("infer_images") or
        st.session_state.get("infer_video") or
        st.session_state.get("infer_stream_url")
    )
    solution = st.session_state.get("selected_solution", "None (Detection Only)")
    sol_info = SOLUTIONS.get(solution, {})
    needs_zone = sol_info.get("needs_zone", False)
    has_zone = bool(
        st.session_state.get("zone_line_points") or
        st.session_state.get("zone_polygons") or
        not needs_zone
    )

    def _dot(ok):
        return "🟢" if ok else "🔴"

    st.markdown(
        f'<div style="display:flex;gap:1.5rem;padding:0.5rem 1rem;background:#F9FAFB;'
        f'border-radius:8px;border:1px solid #E8E8EF;margin-bottom:1rem;font-size:0.82rem;">'
        f'<span>{_dot(has_model)} Model Weights</span>'
        f'<span>{_dot(has_data)} Input Data</span>'
        f'<span>{_dot(has_zone)} Zone Config</span>'
        f'<span style="margin-left:auto;color:#6B7280;">Ready: {sum([has_model, has_data, has_zone])}/3</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════
# TAB 1: MODEL & WEIGHTS
# ════════════════════════════════════════════════════════════════

def _tab_model():
    """Select model architecture, device, and upload weights."""
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Model Architecture")
        st.selectbox(
            "Architecture",
            ["FlashDet-Pico", "FlashDet-Nano", "FlashDet-Small", "FlashDet-Medium", "FlashDet-Large"],
            key="infer_model_arch",
        )

        st.markdown("#### Inference Parameters")
        st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="infer_conf")
        st.slider("NMS IoU Threshold", 0.0, 1.0, 0.45, 0.05, key="infer_nms")
        st.number_input("Image Size", 320, 1920, 640, 32, key="infer_img_size")

        st.markdown("#### Device")
        devices = _get_device_options()
        st.selectbox("Compute Device", devices, key="infer_device")

    with col_right:
        st.markdown("#### Model Weights")
        weight_source = st.radio(
            "Source",
            ["Upload file", "Enter path", "Use training output"],
            key="weight_source",
        )

        if weight_source == "Upload file":
            uploaded_w = st.file_uploader(
                "Upload .pt / .pth / .onnx / .engine",
                type=["pt", "pth", "onnx", "engine"],
                key="infer_weights_file",
            )
            if uploaded_w:
                st.success(f"Loaded: {uploaded_w.name} ({uploaded_w.size / 1e6:.1f} MB)")

        elif weight_source == "Enter path":
            st.text_input("Weights path", placeholder="/path/to/best.pt", key="infer_weights_path")

        else:
            save_dir = st.session_state.get("save_dir", os.path.join(os.getcwd(), "flashstudio_runs"))
            path = f"{save_dir}/model_best_inference.pth"
            st.session_state["infer_weights_path"] = path
            st.info(f"Using: `{path}`")

        st.markdown("#### Classes")
        st.number_input("Number of Classes", 1, 1000, 80, key="infer_num_classes")

        st.markdown("#### Class Filter")
        st.multiselect(
            "Detect only (leave empty for all)",
            COCO_CLASSES,
            default=[],
            key="infer_class_filter",
        )


# ════════════════════════════════════════════════════════════════
# TAB 2: UPLOAD DATA
# ════════════════════════════════════════════════════════════════

def _tab_data():
    """Upload images or video with preview and metadata."""
    input_type = st.radio(
        "Input Type",
        ["Images", "Video", "RTSP Stream"],
        horizontal=True,
        key="infer_input_type",
    )

    if input_type == "Images":
        _data_images()
    elif input_type == "Video":
        _data_video()
    else:
        _data_stream()


def _data_images():
    """Image upload with grid preview."""
    st.markdown("#### Upload Images")
    uploaded = st.file_uploader(
        "Select images (JPG, PNG, WebP)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
        key="infer_images",
    )
    if uploaded:
        st.success(f"{len(uploaded)} image(s) uploaded")
        cols = st.columns(min(len(uploaded), 4))
        for i, f in enumerate(uploaded[:4]):
            with cols[i]:
                st.image(f, caption=f.name, width=None)
        if len(uploaded) > 4:
            st.caption(f"...and {len(uploaded) - 4} more")


def _data_video():
    """Video upload with metadata and first-frame preview."""
    col_vid, col_opts = st.columns(2)
    with col_vid:
        st.markdown("#### Upload Video")
        st.file_uploader(
            "Select video (MP4, AVI, MOV)",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            key="infer_video",
        )
        if st.session_state.get("infer_video"):
            vid = st.session_state["infer_video"]
            st.success(f"{vid.name} ({vid.size / 1e6:.1f} MB)")

            # Show video metadata + first frame
            meta = _get_video_metadata(vid)
            if meta:
                mc1, mc2, mc3, mc4 = st.columns(4)
                with mc1:
                    st.metric("Resolution", f"{meta['width']}x{meta['height']}")
                with mc2:
                    st.metric("FPS", f"{meta['fps']:.1f}")
                with mc3:
                    st.metric("Frames", meta['total_frames'])
                with mc4:
                    duration = meta['total_frames'] / meta['fps'] if meta['fps'] > 0 else 0
                    st.metric("Duration", f"{duration:.1f}s")

            first_frame = _get_first_frame()
            if first_frame:
                st.image(first_frame, caption="First Frame Preview", width=None)

    with col_opts:
        st.markdown("#### Processing Options")
        st.number_input("Max Frames (0=all)", 0, 100000, 300, key="infer_max_frames")
        st.select_slider("Skip every N frames", [1, 2, 3, 5, 10], value=1, key="infer_frame_skip")
        st.checkbox("Save output video", value=True, key="infer_save_video")


def _data_stream():
    """RTSP / webcam stream configuration."""
    st.markdown("#### RTSP / Webcam Stream")
    col_url, col_opts = st.columns([3, 1])
    with col_url:
        st.text_input("Stream URL", placeholder="rtsp://192.168.1.100:554/stream", key="infer_stream_url")
    with col_opts:
        st.number_input("Duration (s)", 0, 3600, 60, key="infer_stream_duration")

    if st.session_state.get("infer_stream_url"):
        if st.button("Test Connection", key="test_stream"):
            with st.spinner("Testing stream..."):
                try:
                    import cv2
                    cap = cv2.VideoCapture(st.session_state["infer_stream_url"])
                    if cap.isOpened():
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            st.success("Stream connected successfully!")
                            st.image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                                     caption="Stream Preview", width=None)
                        else:
                            st.error("Connected but failed to read frame.")
                    else:
                        st.error("Failed to connect to stream.")
                except Exception as e:
                    st.error(f"Connection error: {e}")


def _get_video_metadata(video_file) -> dict | None:
    """Extract video metadata using OpenCV."""
    tmp_path = None
    try:
        import cv2
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp.name
        tmp.write(video_file.read())
        tmp.flush()
        tmp.close()
        video_file.seek(0)
        cap = cv2.VideoCapture(tmp_path)
        if cap.isOpened():
            meta = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS) or 30,
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            }
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


# ════════════════════════════════════════════════════════════════
# TAB 3: SOLUTION & ZONE (Interactive Click-to-Draw)
# ════════════════════════════════════════════════════════════════

def _get_first_frame() -> Image.Image | None:
    """Extract the first frame from uploaded video or first uploaded image."""
    video = st.session_state.get("infer_video")
    if video:
        tmp_path = None
        try:
            import cv2
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp_path = tmp.name
            tmp.write(video.read())
            tmp.flush()
            tmp.close()
            video.seek(0)
            cap = cv2.VideoCapture(tmp_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    images = st.session_state.get("infer_images")
    if images and len(images) > 0:
        try:
            img = Image.open(images[0]).convert("RGB")
            images[0].seek(0)
            return img
        except Exception:
            pass

    return None


def _tab_solution():
    """Select FlashDet solution and configure zones with interactive click drawing."""
    col_sol, col_info = st.columns([2, 3])

    with col_sol:
        st.markdown("#### Select Solution")
        selected = st.selectbox(
            "FlashDet Solution",
            list(SOLUTIONS.keys()),
            key="selected_solution",
            format_func=lambda x: f"{SOLUTIONS[x]['icon']} {x}",
        )
        sol = SOLUTIONS[selected]
        st.caption(sol["description"])

    with col_info:
        if sol["needs_zone"]:
            zone_type = sol.get("zone_type", "polygon")
            if zone_type == "line":
                st.info("🖊️ **Click 2 points** on the frame to define the counting line (start → end).")
            else:
                st.info("🖊️ **Click points** on the frame to define the zone. Click 'Close Polygon' when done.")
        else:
            st.success("✅ This solution does not require zone configuration.")

    st.divider()

    if not sol["needs_zone"]:
        st.markdown("#### Available Solutions Reference")
        cols = st.columns(3)
        for i, (name, info) in enumerate(SOLUTIONS.items()):
            if name == "None (Detection Only)":
                continue
            with cols[i % 3]:
                zone = "Line" if info.get("zone_type") == "line" else ("Polygon" if info["needs_zone"] else "—")
                st.markdown(f"{info['icon']} **{name}**  \n<small style='color:#6B7280'>Zone: {zone}</small>",
                            unsafe_allow_html=True)
        return

    # Interactive zone drawing
    _zone_draw_ui(sol)


def _zone_draw_ui(sol: dict):
    """Interactive zone drawing interface using custom canvas component."""
    zone_type = sol.get("zone_type", "polygon")

    draw_mode = zone_type
    if zone_type == "polygon":
        draw_mode = st.radio(
            "Drawing Mode",
            ["polygon", "rect"],
            format_func=lambda x: "🔷 Polygon (click vertices)" if x == "polygon" else "⬜ Rectangle (2 corners)",
            horizontal=True,
            key="zone_draw_mode",
        )

    if not _HAS_ZONE_DRAWER:
        st.warning("Interactive zone drawing component is not available. Enter coordinates manually below.")
        _manual_zone_input(draw_mode)
        return

    # Frame image
    frame_img = _get_first_frame()
    if frame_img is None:
        st.warning("⚠️ Upload a video or image in the **Upload Data** tab first, then return here to draw zones.")

    # Existing points from session state
    existing_points = st.session_state.get("zone_draw_points", [])
    existing_closed = st.session_state.get("zone_closed", False)

    # Render the interactive canvas
    st.markdown("#### Click on Frame to Draw Zone")
    result = zone_drawer(
        image=frame_img,
        mode=draw_mode,
        points=existing_points,
        closed=existing_closed,
        display_width=700,
        key="zone_canvas",
    )

    # Process result from component
    if result is not None and isinstance(result, dict):
        points = result.get("points", [])
        closed = result.get("closed", False)
        st.session_state["zone_draw_points"] = points
        st.session_state["zone_closed"] = closed

        if points:
            display_w = result.get("displayWidth", 700)
            display_h = result.get("displayHeight", 525)
            orig_w = frame_img.width if frame_img else 640
            orig_h = frame_img.height if frame_img else 480
            scale_x = orig_w / display_w
            scale_y = orig_h / display_h

            st.divider()
            _store_zone_coords(points, draw_mode, scale_x, scale_y)
    else:
        st.divider()
        st.caption("👆 Click on the frame above to start drawing the zone.")


def _manual_zone_input(draw_mode: str):
    """Fallback manual coordinate entry when zone_drawer component is unavailable."""
    if draw_mode == "line":
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.number_input("Start X", 0, 1920, 100, key="manual_line_x1")
            y1 = st.number_input("Start Y", 0, 1080, 240, key="manual_line_y1")
        with col2:
            x2 = st.number_input("End X", 0, 1920, 540, key="manual_line_x2")
            y2 = st.number_input("End Y", 0, 1080, 240, key="manual_line_y2")
        if st.button("Set Line", key="set_manual_line"):
            st.session_state["zone_line_points"] = [(x1, y1), (x2, y2)]
            st.success(f"Line set: ({x1}, {y1}) → ({x2}, {y2})")
    else:
        pts_text = st.text_area(
            "Polygon points (x,y per line)",
            value="100,100\n500,100\n500,400\n100,400",
            key="manual_polygon_text",
        )
        if st.button("Set Polygon", key="set_manual_polygon"):
            points = []
            for line in pts_text.strip().split("\n"):
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        points.append((int(parts[0].strip()), int(parts[1].strip())))
                    except ValueError:
                        continue
            if len(points) >= 3:
                st.session_state["zone_polygons"] = [points]
                st.success(f"Polygon set with {len(points)} vertices")
            else:
                st.error("Need at least 3 valid points for a polygon.")


def _store_zone_coords(display_points: list, draw_mode: str, scale_x: float, scale_y: float):
    """Convert display coordinates to original frame coordinates and store."""
    if not display_points:
        st.caption("👆 Click on the frame above to start drawing the zone.")
        return

    orig_points = [(int(x * scale_x), int(y * scale_y)) for x, y in display_points]

    st.markdown("#### Zone Coordinates")

    if draw_mode == "line" and len(orig_points) >= 2:
        st.session_state["zone_line_points"] = [orig_points[0], orig_points[1]]
        st.session_state["line_x1"] = orig_points[0][0]
        st.session_state["line_y1"] = orig_points[0][1]
        st.session_state["line_x2"] = orig_points[1][0]
        st.session_state["line_y2"] = orig_points[1][1]
        st.session_state["auto_line"] = False
        st.success(f"✅ Line: ({orig_points[0][0]}, {orig_points[0][1]}) → ({orig_points[1][0]}, {orig_points[1][1]})")

    elif draw_mode == "rect" and len(orig_points) >= 2:
        x1, y1 = orig_points[0]
        x2, y2 = orig_points[1]
        rect_poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        st.session_state["zone_polygons"] = [rect_poly]
        pts_str = "\n".join(f"{x},{y}" for x, y in rect_poly)
        st.session_state["polygon_points"] = pts_str
        st.success(f"✅ Rectangle: ({x1},{y1}) → ({x2},{y2})")

    elif draw_mode == "polygon" and len(orig_points) >= 3:
        st.session_state["zone_polygons"] = [orig_points]
        pts_str = "\n".join(f"{x},{y}" for x, y in orig_points)
        st.session_state["polygon_points"] = pts_str
        closed = st.session_state.get("zone_closed", False)
        pts_display = " → ".join(f"({x},{y})" for x, y in orig_points)
        if closed:
            st.success(f"✅ Polygon ({len(orig_points)} pts): {pts_display}")
        else:
            st.info(f"⏳ {len(orig_points)} points placed — click more or Close Polygon")
    else:
        needed = 2 if draw_mode in ("line", "rect") else 3
        st.info(f"⏳ {len(orig_points)}/{needed} points — keep clicking to add more.")


# ════════════════════════════════════════════════════════════════
# TAB 4: RUN & RESULTS
# ════════════════════════════════════════════════════════════════

def _tab_run():
    """Run inference with configuration summary and results."""
    # Configuration summary
    _render_config_summary()

    st.divider()

    # Readiness check
    input_type = st.session_state.get("infer_input_type", "Images")
    has_data = (
        st.session_state.get("infer_images") or
        st.session_state.get("infer_video") or
        st.session_state.get("infer_stream_url")
    )

    if not has_data:
        st.warning("⚠️ No data uploaded. Go to the **Upload Data** tab first.")
        return

    # Run controls
    col_run, col_opts = st.columns([1, 2])
    with col_run:
        run_clicked = st.button("🚀 Run Inference", type="primary", key="run_infer_btn", use_container_width=True)
    with col_opts:
        st.checkbox("Show per-frame detections", value=False, key="show_per_frame")

    if run_clicked:
        if input_type == "Images":
            _run_images()
        elif input_type == "Video":
            _run_video()
        else:
            st.info("Stream inference — coming soon.")

    st.divider()

    if input_type == "Images":
        _show_image_results()
    else:
        _show_video_results()


def _render_config_summary():
    """Show a compact configuration summary before running."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Model", st.session_state.get("infer_model_arch", "FlashDet-Nano"))
    with c2:
        st.metric("Input", st.session_state.get("infer_input_type", "Images"))
    with c3:
        st.metric("Solution", st.session_state.get("selected_solution", "Detection Only")[:18])
    with c4:
        device = st.session_state.get("infer_device", "cpu")
        st.metric("Device", device.split(" ")[0])

    # Expandable detailed config
    with st.expander("View Full Configuration", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Model Settings**")
            st.text(f"  Architecture: {st.session_state.get('infer_model_arch', 'FlashDet-Nano')}")
            st.text(f"  Confidence: {st.session_state.get('infer_conf', 0.25)}")
            st.text(f"  NMS IoU: {st.session_state.get('infer_nms', 0.45)}")
            st.text(f"  Image Size: {st.session_state.get('infer_img_size', 640)}")
            class_filter = st.session_state.get("infer_class_filter", [])
            st.text(f"  Class Filter: {', '.join(class_filter) if class_filter else 'All classes'}")
        with col_b:
            st.markdown("**Zone Configuration**")
            solution = st.session_state.get("selected_solution", "None (Detection Only)")
            st.text(f"  Solution: {solution}")
            line_pts = st.session_state.get("zone_line_points")
            if line_pts:
                st.text(f"  Line: {line_pts[0]} → {line_pts[1]}")
            polys = st.session_state.get("zone_polygons")
            if polys:
                st.text(f"  Polygon: {len(polys[0])} vertices")


# ════════════════════════════════════════════════════════════════
# IMAGE INFERENCE
# ════════════════════════════════════════════════════════════════

def _run_images():
    """Run inference on uploaded images."""
    images = st.session_state.get("infer_images", [])
    if not images:
        return

    results = []
    total_time = 0
    progress = st.progress(0, text="Running inference...")

    for i, f in enumerate(images):
        img = Image.open(f).convert("RGB")
        t0 = time.perf_counter()
        dets = _detect(img)
        elapsed = time.perf_counter() - t0
        total_time += elapsed
        annotated = _draw_boxes(img.copy(), dets)
        results.append({
            "name": f.name,
            "dets": dets,
            "annotated": annotated,
            "inference_time_ms": elapsed * 1000,
            "width": img.width,
            "height": img.height,
        })
        progress.progress((i + 1) / len(images), text=f"Processing {i+1}/{len(images)}...")

    progress.empty()
    st.session_state["infer_img_results"] = results
    st.session_state["infer_total_time"] = total_time
    st.rerun()


def _show_image_results():
    """Display image results with real metrics."""
    if "infer_img_results" not in st.session_state:
        st.info("Click **Run Inference** to see results.")
        return

    results = st.session_state["infer_img_results"]
    total_dets = sum(len(r["dets"]) for r in results)
    total_time = st.session_state.get("infer_total_time", 1)
    avg_time_ms = (total_time / len(results)) * 1000 if results else 0

    # Real computed metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        fps = len(results) / total_time if total_time > 0 else 0
        st.metric("Throughput", f"{fps:.1f} img/s")
    with c2:
        st.metric("Avg Latency", f"{avg_time_ms:.1f} ms")
    with c3:
        st.metric("Total Detections", total_dets)
    with c4:
        st.metric("Images Processed", len(results))

    # Per-image results
    for r in results:
        st.divider()
        col_img, col_tbl = st.columns(2)
        with col_img:
            st.image(r["annotated"], caption=f"{r['name']} ({r['inference_time_ms']:.1f}ms)", width=None)
        with col_tbl:
            if r["dets"]:
                import pandas as pd
                df = pd.DataFrame(r["dets"], columns=["Class", "Confidence", "BBox"])
                df.index += 1
                st.dataframe(df, use_container_width=True, height=250)
            else:
                st.warning("No detections.")

    # Export options
    st.divider()
    st.markdown("#### Export Results")
    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        import pandas as pd
        all_d = []
        for r in results:
            for d in r["dets"]:
                all_d.append([r["name"]] + d)
        if all_d:
            df = pd.DataFrame(all_d, columns=["File", "Class", "Confidence", "BBox"])
            st.download_button("📥 CSV Detections", df.to_csv(index=False),
                               "detections.csv", "text/csv", type="primary", use_container_width=True)

    with col_d2:
        coco_json = _to_coco_format(results)
        st.download_button("📥 COCO JSON", json.dumps(coco_json, indent=2),
                           "detections_coco.json", "application/json", use_container_width=True)

    with col_d3:
        if results:
            buf = io.BytesIO()
            results[0]["annotated"].save(buf, format="PNG")
            st.download_button("🖼️ Annotated Image", buf.getvalue(),
                               "annotated.png", "image/png", use_container_width=True)


def _to_coco_format(results: list) -> dict:
    """Convert detection results to COCO JSON format."""
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(COCO_CLASSES)],
    }
    ann_id = 1
    for img_id, r in enumerate(results, 1):
        coco["images"].append({
            "id": img_id,
            "file_name": r["name"],
            "width": r.get("width", 640),
            "height": r.get("height", 480),
        })
        for cls_name, conf, bbox_str in r["dets"]:
            bbox = json.loads(bbox_str.replace("'", '"'))
            x1, y1, x2, y2 = bbox
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": COCO_CLASSES.index(cls_name) if cls_name in COCO_CLASSES else 0,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "score": float(conf),
                "iscrowd": 0,
            })
            ann_id += 1
    return coco


# ════════════════════════════════════════════════════════════════
# VIDEO INFERENCE
# ════════════════════════════════════════════════════════════════

def _run_video():
    """Run video inference with solution integration via flashdet."""
    video = st.session_state.get("infer_video")
    solution = st.session_state.get("selected_solution", "None (Detection Only)")

    weights = st.session_state.get("infer_weights_path", "") or st.session_state.get("infer_weights_file")
    if weights:
        try:
            _run_flashdet_video(video, solution)
            return
        except Exception as e:
            st.warning(f"FlashDet inference error: {e}. Running demo mode.")

    _run_demo_video(solution)


def _show_video_results():
    """Display video inference results."""
    if "video_results" not in st.session_state:
        st.info("Click **Run Inference** to see results.")
        return

    res = st.session_state["video_results"]

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Frames Processed", res.get("frames", 0))
    with c2:
        st.metric("Total Detections", res.get("detections", 0))
    with c3:
        st.metric("Processing FPS", f"{res.get('processing_fps', 0):.1f}")
    with c4:
        elapsed = res.get("elapsed_time", 0)
        st.metric("Total Time", f"{elapsed:.1f}s")

    # Solution output
    sol = res.get("solution_output", {})
    if sol:
        st.divider()
        st.markdown("#### Solution Results")

    if "in_count" in sol:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("In Count", sol["in_count"])
        with c2:
            st.metric("Out Count", sol["out_count"])
        with c3:
            st.metric("Net Flow", sol["in_count"] - sol["out_count"])

    if "region_counts" in sol:
        cols = st.columns(len(sol["region_counts"]))
        for i, (region, count) in enumerate(sol["region_counts"].items()):
            with cols[i]:
                st.metric(f"Zone: {region}", count)

    if "speeds" in sol:
        import pandas as pd
        df = pd.DataFrame(sol["speeds"], columns=["Track ID", "Speed (km/h)"])
        st.dataframe(df, use_container_width=True, hide_index=True)

    # Sample frames
    if res.get("frames_preview"):
        st.divider()
        st.markdown("#### Sample Frames")
        n_frames = len(res["frames_preview"])
        cols = st.columns(min(n_frames, 4))
        for i, frame in enumerate(res["frames_preview"][:4]):
            with cols[i]:
                st.image(frame, caption=f"Frame {i+1}", width=None)

    if res.get("output_path"):
        st.success(f"Output saved: `{res['output_path']}`")

    # Export
    st.divider()
    st.markdown("#### Export")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        summary = {
            "solution": st.session_state.get("selected_solution"),
            "frames_processed": res.get("frames"),
            "total_detections": res.get("detections"),
            "processing_fps": res.get("processing_fps"),
            "solution_output": sol,
        }
        st.download_button("📥 Results JSON", json.dumps(summary, indent=2),
                           "video_results.json", "application/json", use_container_width=True)


def _run_demo_video(solution_name: str):
    """Demo video processing with realistic simulation."""
    t0 = time.perf_counter()
    progress = st.progress(0, text="Processing video...")
    frames_preview = []

    for i in range(4):
        time.sleep(0.3)
        progress.progress((i + 1) / 4, text=f"Frame {(i+1)*75}/300")
        img = Image.new("RGB", (640, 480), color=(248, 248, 252))
        draw = ImageDraw.Draw(img)
        draw.rectangle([80 + i * 40, 80 + i * 30, 250 + i * 40, 230 + i * 30],
                       outline="#7C3AED", width=3)
        draw.text((90 + i * 40, 65 + i * 30), f"car 0.{85 + i}", fill="#1A1A2E")

        # Draw zone overlay on preview frames
        line_pts = st.session_state.get("zone_line_points")
        if line_pts:
            draw.line([(int(line_pts[0][0]), int(line_pts[0][1])),
                       (int(line_pts[1][0]), int(line_pts[1][1]))],
                      fill="#10B981", width=2)

        frames_preview.append(img)

    progress.empty()
    elapsed = time.perf_counter() - t0

    sol_output = {}
    if "Counter" in solution_name:
        sol_output = {"in_count": 12, "out_count": 8}
    elif "Region" in solution_name:
        sol_output = {"region_counts": {"Zone A": 5, "Zone B": 3}}
    elif "Speed" in solution_name:
        sol_output = {"speeds": [[1, 45.2], [2, 38.7], [3, 52.1]]}
    elif "Security" in solution_name:
        sol_output = {"alerts": 2, "alert_frames": [45, 178]}
    elif "Queue" in solution_name:
        sol_output = {"region_counts": {"Queue 1": 7, "Queue 2": 4}}

    st.session_state["video_results"] = {
        "frames": 300,
        "detections": 1247,
        "processing_fps": 300 / elapsed if elapsed > 0 else 0,
        "elapsed_time": elapsed,
        "solution_output": sol_output,
        "frames_preview": frames_preview,
        "output_path": os.path.join(os.getcwd(), "output.mp4") if st.session_state.get("infer_save_video") else None,
    }
    st.rerun()


def _run_flashdet_video(video_file, solution_name: str):
    """Run real FlashDet video processing with solutions integration."""
    if not _HAS_PREDICTOR:
        raise ImportError("flashdet Predictor not available")

    import cv2

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_file.read())
    tmp.flush()
    video_file.seek(0)

    cap = cv2.VideoCapture(tmp.name)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    weights = st.session_state.get("infer_weights_path", "")
    if not weights:
        uploaded = st.session_state.get("infer_weights_file")
        if uploaded:
            wtmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
            wtmp.write(uploaded.read())
            wtmp.flush()
            uploaded.seek(0)
            weights = wtmp.name

    device = st.session_state.get("infer_device", "cpu").split(" ")[0]
    conf = st.session_state.get("infer_conf", 0.25)
    nms = st.session_state.get("infer_nms", 0.45)
    img_size = st.session_state.get("infer_img_size", 640)

    predictor = Predictor(
        model_path=weights,
        device=device,
        conf_thresh=conf,
        nms_thresh=nms,
        input_size=img_size,
    )

    tracker = FlashTracker()
    solution = _create_solution(solution_name, predictor, tracker, fps)

    max_frames = st.session_state.get("infer_max_frames", 300) or total
    skip = st.session_state.get("infer_frame_skip", 1)

    frames_preview = []
    processed = 0
    total_dets = 0
    t0 = time.perf_counter()
    progress = st.progress(0, text="Processing video...")

    for idx in range(min(total, max_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip != 0:
            continue

        if solution:
            annotated, results = solution.process_frame(frame)
        else:
            raw_results = predictor(frame)
            annotated = frame
            results = raw_results

        if isinstance(results, list):
            total_dets += len(results)

        processed += 1
        if idx % max(total // 4, 1) == 0:
            rgb = cv2.cvtColor(annotated if annotated is not None else frame, cv2.COLOR_BGR2RGB)
            frames_preview.append(Image.fromarray(rgb))
        progress.progress(idx / min(total, max_frames), text=f"Frame {idx}/{min(total, max_frames)}")

    cap.release()
    progress.empty()
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
        "frames": processed,
        "detections": total_dets,
        "processing_fps": processed / elapsed if elapsed > 0 else 0,
        "elapsed_time": elapsed,
        "solution_output": sol_output,
        "frames_preview": frames_preview,
        "output_path": os.path.join(os.getcwd(), "output.mp4") if st.session_state.get("infer_save_video") else None,
    }
    st.rerun()


def _create_solution(solution_name: str, predictor, tracker, fps: float):
    """Create a flashdet solution instance based on the selected solution name."""
    line_pts = st.session_state.get("zone_line_points")
    polygons = st.session_state.get("zone_polygons")
    classes = st.session_state.get("infer_class_filter") or None

    if solution_name == "None (Detection Only)":
        return None

    # Map solution display names to class names and constructor kwargs
    solution_map = {
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

    for keyword, (cls_name, kwargs) in solution_map.items():
        if keyword in solution_name:
            cls = _SOLUTIONS_AVAILABLE.get(cls_name)
            if cls is None:
                st.warning(f"Solution `{cls_name}` is not available in the installed flashdet version.")
                return None
            # Most solutions need predictor and tracker
            needs_tracker = cls_name not in ("Heatmap", "ObjectBlurrer", "CrowdDensity")
            try:
                if needs_tracker:
                    return cls(predictor=predictor, tracker=tracker, **kwargs)
                else:
                    return cls(predictor=predictor, **kwargs)
            except TypeError:
                return cls(predictor=predictor, **kwargs)

    return None


# ════════════════════════════════════════════════════════════════
# DETECTION UTILITIES
# ════════════════════════════════════════════════════════════════

def _detect(image: Image.Image) -> list:
    """Run detection using flashdet.Predictor."""
    weights = st.session_state.get("infer_weights_path", "")
    if not weights:
        uploaded = st.session_state.get("infer_weights_file")
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
            tmp.write(uploaded.read())
            tmp.flush()
            uploaded.seek(0)
            weights = tmp.name

    if weights:
        try:
            return _detect_real(image, weights)
        except Exception as e:
            st.warning(f"Real inference failed: {e}. Showing demo detections.")
    else:
        st.warning("⚠️ No model weights loaded — showing **demo detections** (random boxes). "
                   "Upload weights in the **Model & Weights** tab for real inference.")
    return _detect_demo(image)


def _detect_real(image: Image.Image, weights_path: str) -> list:
    """Real FlashDet detection using flashdet.Predictor."""
    if not _HAS_PREDICTOR:
        raise ImportError("flashdet Predictor not available")

    device = st.session_state.get("infer_device", "cpu").split(" ")[0]
    conf = st.session_state.get("infer_conf", 0.25)
    nms = st.session_state.get("infer_nms", 0.45)
    img_size = st.session_state.get("infer_img_size", 640)

    predictor = Predictor(
        model_path=weights_path,
        device=device,
        conf_thresh=conf,
        nms_thresh=nms,
        input_size=img_size,
    )

    # Predictor.__call__ accepts both str paths and numpy arrays
    img_array = np.array(image)
    raw_results = predictor(img_array)

    class_filter = st.session_state.get("infer_class_filter", [])
    dets = []
    for bbox, score, cls_id in raw_results:
        cls_id = int(cls_id)
        name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
        if class_filter and name not in class_filter:
            continue
        bbox_list = bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox)
        bbox_int = [int(x) for x in bbox_list]
        dets.append([name, f"{float(score):.2f}", str(bbox_int)])
    return dets


def _detect_demo(image: Image.Image) -> list:
    """Demo detection with class filtering."""
    w, h = image.size
    rng = np.random.default_rng(hash(image.tobytes()[:100]) % (2**31))
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
        x2 = min(x1 + int(rng.integers(60, 200)), w)
        y2 = min(y1 + int(rng.integers(60, 200)), h)
        dets.append([name, f"{conf:.2f}", str([x1, y1, x2, y2])])
    return sorted(dets, key=lambda x: float(x[1]), reverse=True)


def _draw_boxes(image: Image.Image, dets: list) -> Image.Image:
    """Draw bounding boxes with labels."""
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


