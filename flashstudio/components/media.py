"""Shared image rendering helpers.

Both the dataset preview and the training ground-truth views need tiles that
occupy the same window regardless of an image's aspect ratio. Embedding the
image in a fixed-height, letterboxed window keeps mixed portrait/landscape
rows aligned instead of looking ragged.
"""

import base64
import html
import io
import mimetypes
import os

import streamlit as st
from PIL import Image


def image_data_uri(source) -> str:
    """Return a ``data:`` URI for a file path or a PIL image."""
    if isinstance(source, (str, os.PathLike)):
        mime = mimetypes.guess_type(str(source))[0] or "image/jpeg"
        with open(source, "rb") as fh:
            payload = base64.b64encode(fh.read()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    image = source
    if not isinstance(image, Image.Image):
        raise TypeError("source must be a path or a PIL.Image")

    buffer = io.BytesIO()
    fmt, mime = "JPEG", "image/jpeg"
    if image.mode in ("RGBA", "LA", "P"):
        fmt, mime = "PNG", "image/png"
    elif image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffer, format=fmt, quality=90)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def render_fixed_image(source, caption: str = "", height: int = 240):
    """Render an image in a fixed-height window (letterboxed, never cropped)."""
    uri = image_data_uri(source)
    label = html.escape(str(caption))
    st.markdown(
        f'<div class="fs-image-window" style="width:100%;height:{int(height)}px;'
        "border-radius:10px;overflow:hidden;background:#f3f4f6;"
        "border:1px solid #e5e7eb;display:flex;align-items:center;"
        'justify-content:center;">'
        f'<img src="{uri}" alt="{label}" '
        'style="width:100%;height:100%;object-fit:contain;" />'
        "</div>"
        '<div style="text-align:center;font-size: var(--fs-text-sm);'
        "color:var(--fs-muted);margin-top:.3rem;white-space:nowrap;"
        "overflow:hidden;text-overflow:ellipsis;"
        f'">{label}</div>',
        unsafe_allow_html=True,
    )
