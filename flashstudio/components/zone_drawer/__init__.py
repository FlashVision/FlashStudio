"""Zone drawing component using streamlit-drawable-canvas for bi-directional data."""

import json
import io
import numpy as np
import streamlit as st
from PIL import Image

try:
    # Patch for streamlit-drawable-canvas 0.9.x + Streamlit 1.30+
    # st.elements.image.image_to_url was removed; provide a shim
    import streamlit.elements.image as _img_mod
    if not hasattr(_img_mod, "image_to_url"):
        def _image_to_url(image, width, clamp, channels, output_format, image_id, allow_emoji=True):
            """Compatibility shim for removed image_to_url."""
            import base64
            buf = io.BytesIO()
            if isinstance(image, Image.Image):
                image.save(buf, format="PNG")
            elif isinstance(image, np.ndarray):
                Image.fromarray(image).save(buf, format="PNG")
            else:
                return ""
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"
        _img_mod.image_to_url = _image_to_url

    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except (ImportError, Exception):
    _HAS_CANVAS = False


def zone_drawer(
    image: Image.Image | None = None,
    mode: str = "polygon",
    points: list | None = None,
    closed: bool = False,
    display_width: int = 700,
    key: str | None = None,
) -> dict | None:
    if not _HAS_CANVAS:
        st.warning("Install `streamlit-drawable-canvas` for interactive drawing.")
        return None

    _MODE_MAP = {"polygon": "polygon", "rect": "rect", "line": "line"}
    canvas_mode = _MODE_MAP.get(mode, "polygon")

    if image is not None:
        aspect = image.height / image.width
        display_height = int(display_width * aspect)
        bg = image.resize((display_width, display_height))
    else:
        display_height = int(display_width * 0.6)
        bg = Image.new("RGB", (display_width, display_height), (26, 26, 46))

    initial_drawing = None
    if points and len(points) >= 2:
        if image:
            sx = display_width / image.width
            sy = display_height / image.height
        else:
            sx = sy = 1.0
        scaled = [[p[0] * sx, p[1] * sy] for p in points]
        initial_drawing = _build_initial_drawing(scaled, mode, closed, display_width, display_height)

    canvas_result = st_canvas(
        fill_color="rgba(124, 58, 237, 0.15)",
        stroke_width=3,
        stroke_color="#7C3AED",
        background_image=bg,
        drawing_mode=canvas_mode,
        point_display_radius=0,
        width=display_width,
        height=display_height,
        key=key or "zone_canvas",
        initial_drawing=initial_drawing,
    )

    if canvas_result is None or canvas_result.json_data is None:
        return None

    objects = canvas_result.json_data.get("objects", [])
    if not objects:
        return None

    extracted_points = []
    if image:
        sx = image.width / display_width
        sy = image.height / display_height
    else:
        sx = sy = 1.0

    for obj in objects:
        obj_type = obj.get("type", "")
        left = obj.get("left", 0)
        top = obj.get("top", 0)

        if obj_type == "rect":
            w = obj.get("width", 0) * obj.get("scaleX", 1)
            h = obj.get("height", 0) * obj.get("scaleY", 1)
            extracted_points = [
                [int((left) * sx), int((top) * sy)],
                [int((left + w) * sx), int((top) * sy)],
                [int((left + w) * sx), int((top + h) * sy)],
                [int((left) * sx), int((top + h) * sy)],
            ]
        elif obj_type == "line":
            x1 = obj.get("x1", 0) + left
            y1 = obj.get("y1", 0) + top
            x2 = obj.get("x2", 0) + left
            y2 = obj.get("y2", 0) + top
            extracted_points = [
                [int(x1 * sx), int(y1 * sy)],
                [int(x2 * sx), int(y2 * sy)],
            ]
        elif obj_type == "polygon" or obj_type == "path":
            path = obj.get("path", [])
            for cmd in path:
                if len(cmd) >= 3 and cmd[0] in ("M", "L"):
                    extracted_points.append([int((cmd[1] + left) * sx), int((cmd[2] + top) * sy)])

    if not extracted_points:
        return None

    return {
        "points": extracted_points,
        "closed": closed or mode in ("rect", "polygon"),
        "mode": mode,
        "displayWidth": display_width,
        "displayHeight": display_height,
    }


def _build_initial_drawing(points, mode, closed, dw, dh):
    objects = []
    if mode == "rect" and len(points) >= 2:
        x1, y1 = points[0]
        x2, y2 = points[-1]
        objects.append({
            "type": "rect",
            "left": min(x1, x2),
            "top": min(y1, y2),
            "width": abs(x2 - x1),
            "height": abs(y2 - y1),
            "fill": "rgba(124, 58, 237, 0.15)",
            "stroke": "#7C3AED",
            "strokeWidth": 3,
        })
    elif mode == "line" and len(points) >= 2:
        objects.append({
            "type": "line",
            "x1": points[0][0],
            "y1": points[0][1],
            "x2": points[1][0],
            "y2": points[1][1],
            "left": 0,
            "top": 0,
            "stroke": "#7C3AED",
            "strokeWidth": 3,
        })
    elif len(points) >= 3:
        path = [["M", points[0][0], points[0][1]]]
        for p in points[1:]:
            path.append(["L", p[0], p[1]])
        if closed:
            path.append(["L", points[0][0], points[0][1]])
        objects.append({
            "type": "path",
            "path": path,
            "left": 0,
            "top": 0,
            "fill": "rgba(124, 58, 237, 0.15)" if closed else "transparent",
            "stroke": "#7C3AED",
            "strokeWidth": 3,
        })
    return {"version": "5.3.0", "objects": objects}
