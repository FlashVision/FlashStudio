"""Custom Streamlit component for interactive zone drawing on video frames."""

import os
import base64
import io
import streamlit.components.v1 as components
from PIL import Image

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))
_component_func = components.declare_component("zone_drawer", path=_COMPONENT_DIR)


def zone_drawer(
    image: Image.Image | None = None,
    mode: str = "polygon",
    points: list | None = None,
    closed: bool = False,
    display_width: int = 700,
    key: str | None = None,
) -> dict | None:
    """
    Render interactive zone drawing canvas.

    Args:
        image: Background PIL Image (video frame)
        mode: Drawing mode - 'line', 'polygon', or 'rect'
        points: Existing points to pre-draw [[x,y], ...]
        closed: Whether the shape is closed
        display_width: Width of the canvas in pixels
        key: Streamlit widget key

    Returns:
        Dict with {points, closed, mode, displayWidth, displayHeight} or None
    """
    image_data = None
    img_width = 640
    img_height = 480

    if image is not None:
        img_width = image.width
        img_height = image.height
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        image_data = f"data:image/jpeg;base64,{b64}"

    aspect = img_height / img_width
    display_height = int(display_width * aspect)

    result = _component_func(
        imageData=image_data,
        mode=mode,
        points=points or [],
        closed=closed,
        width=img_width,
        height=img_height,
        displayWidth=display_width,
        displayHeight=display_height,
        key=key,
        default=None,
    )

    return result
