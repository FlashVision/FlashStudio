"""Inference — Data tab."""

import os
import tempfile
import streamlit as st
from PIL import Image
from flashstudio.constants import (
    INFER_MAX_FRAMES, INFER_STREAM_DURATION,
    INFER_FRAME_SKIP, MAX_PREVIEW_IMAGES,
)


def _advance_to_run():
    """After adding input data, move to Run (or Model if none is selected)."""
    if not (
        st.session_state.get("infer_images")
        or st.session_state.get("infer_video")
        or st.session_state.get("infer_stream_url")
    ):
        return
    try:
        from flashstudio.pages.inference.run.detection import _resolve_weights_path

        has_model = bool(_resolve_weights_path()) or bool(
            st.session_state.get("infer_demo_mode")
        )
    except Exception:
        has_model = False
    st.session_state["_infer_tab_next"] = "Run" if has_model else "Model"
    if st.session_state.get("infer_input_type") == "RTSP":
        st.session_state["_infer_run_mode"] = "Live"


def _tab_data():
    input_type = st.radio(
        "Input", ["Images", "Video", "RTSP"], horizontal=True,
        key="infer_input_type",
    )
    st.caption("Supported sources: images, local video, or a live RTSP stream.")

    left, right = st.columns([3, 2])

    with left:
        with st.container(border=True):
            st.markdown("#### Source")
            if input_type == "Images":
                uploaded = st.file_uploader(
                    "Images", type=["jpg", "jpeg", "png", "webp", "bmp"],
                    accept_multiple_files=True, key="infer_images",
                    label_visibility="collapsed",
                    on_change=_advance_to_run,
                )
                if uploaded:
                    cols = st.columns(min(len(uploaded), MAX_PREVIEW_IMAGES))
                    for i, f in enumerate(uploaded[:MAX_PREVIEW_IMAGES]):
                        with cols[i]:
                            st.image(f, caption=f.name[:10], use_container_width=True)
                    if len(uploaded) > MAX_PREVIEW_IMAGES:
                        st.caption(f"+{len(uploaded) - MAX_PREVIEW_IMAGES} more")
                else:
                    st.caption("Drop images here or click Upload.")

            elif input_type == "Video":
                st.file_uploader(
                    "Video", type=["mp4", "avi", "mov", "mkv"],
                    key="infer_video", label_visibility="collapsed",
                    on_change=_advance_to_run,
                )
                vid = st.session_state.get("infer_video")
                if vid:
                    st.caption(f"{vid.name} ({vid.size / 1e6:.1f}MB)")
                    meta = _get_video_metadata(vid)
                    if meta:
                        mc = st.columns(4)
                        with mc[0]:
                            st.metric("Resolution", f"{meta['width']}×{meta['height']}")
                        with mc[1]:
                            st.metric("FPS", f"{meta['fps']:.0f}")
                        with mc[2]:
                            st.metric("Frames", meta["total_frames"])
                        with mc[3]:
                            st.metric("Duration", f"{meta['total_frames'] / max(meta['fps'], 1):.0f}s")
                else:
                    st.caption("Upload an MP4/AVI/MOV file.")

            else:
                st.text_input(
                    "RTSP URL", placeholder="rtsp://192.168.1.100:554/stream",
                    key="infer_stream_url", label_visibility="collapsed",
                )
                if st.session_state.get("infer_stream_url"):
                    st.caption(f"`{st.session_state['infer_stream_url']}`")
                else:
                    st.caption("Paste an RTSP URL to test the connection.")

    with right:
        with st.container(border=True):
            st.markdown("#### Settings")
            if input_type == "Images":
                count = len(st.session_state.get("infer_images") or [])
                st.metric("Selected images", count)
                st.caption("JPEG, PNG, WEBP, BMP")
                st.caption("Tip: batch inference runs every image and reports per-image detections.")
            elif input_type == "Video":
                st.number_input("Max frames (0 = all)", 0, 100000,
                                INFER_MAX_FRAMES, key="infer_max_frames")
                st.select_slider("Frame skip", [1, 2, 3, 5, 10],
                                 value=INFER_FRAME_SKIP, key="infer_frame_skip")
                st.checkbox("Save output video", True, key="infer_save_video")
            else:
                st.number_input("Duration (s)", 0, 3600,
                                INFER_STREAM_DURATION, key="infer_stream_duration")
                if st.button("Test connection", key="test_stream",
                             use_container_width=True):
                    _test_stream()
                st.caption("Choose Run → Live for continuous RTSP inference.")


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
