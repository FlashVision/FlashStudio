"""Custom Streamlit component for interactive zone drawing on video frames."""

import os
import base64
import io
import streamlit.components.v1 as components
from PIL import Image

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    toolbar_height = 50
    total_height = display_height + toolbar_height

    html = _build_html(image_data, mode, points or [], closed,
                       display_width, display_height, img_width, img_height)

    # #region agent log
    import json as _json_dbg, time as _time_dbg
    with open("/home/ggoswami/Project/Gaurav/FlashVision/FlashStudio/.cursor/debug-b7c49a.log", "a") as _f_dbg:
        _f_dbg.write(_json_dbg.dumps({"sessionId":"b7c49a","location":"zone_drawer/__init__.py:zone_drawer","message":"zone_drawer_render","data":{"has_image":image_data is not None,"mode":mode,"display_width":display_width,"display_height":display_height,"total_height":total_height,"img_width":img_width,"img_height":img_height,"html_len":len(html)},"timestamp":int(_time_dbg.time()*1000),"hypothesisId":"H3"})+"\n")
    # #endregion

    result = components.html(html, height=total_height, scrolling=False)
    return None


def _build_html(image_data, mode, points, closed, dw, dh, iw, ih):
    points_json = str(points).replace("'", '"')
    image_src = image_data or ""
    return f"""
<!DOCTYPE html>
<html>
<head>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif; background: transparent; }}
.container {{ position: relative; display: inline-block; }}
canvas {{ cursor: crosshair; border-radius: 8px; display: block; border: 2px solid #E5E7EB; }}
.toolbar {{
    display: flex; gap: 8px; padding: 8px 0; align-items: center;
}}
.btn {{
    padding: 6px 14px; border-radius: 6px; border: 1px solid #E8E8EF;
    background: #fff; cursor: pointer; font-size: 13px; font-weight: 500;
    transition: all 0.15s;
}}
.btn:hover {{ border-color: #7C3AED; color: #7C3AED; }}
.btn-primary {{ background: #7C3AED; color: #fff; border-color: #7C3AED; }}
.btn-primary:hover {{ background: #6D28D9; }}
.btn-danger {{ color: #EF4444; border-color: #FCA5A5; }}
.btn-danger:hover {{ background: #FEF2F2; }}
.status {{ font-size: 12px; color: #6B7280; margin-left: auto; }}
.mode-label {{ font-size: 12px; font-weight: 600; color: #7C3AED; text-transform: uppercase; }}
</style>
</head>
<body>
<div class="toolbar">
    <span class="mode-label">{mode.upper()}</span>
    <button class="btn btn-danger" onclick="clearAll()">Clear</button>
    <button class="btn" onclick="undoLast()">Undo</button>
    <button class="btn btn-primary" id="closeBtn" onclick="closePolygon()" style="display:none">Close Polygon</button>
    <span class="status" id="status">Click on canvas to draw</span>
</div>
<div class="container">
    <canvas id="canvas" width="{dw}" height="{dh}"></canvas>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let points = {points_json};
let closed = {'true' if closed else 'false'};
let bgImage = null;
const mode = '{mode}';
const maxPoints = (mode === 'line' || mode === 'rect') ? 2 : 50;

// Load background
const imgSrc = "{image_src}";
if (imgSrc) {{
    bgImage = new Image();
    bgImage.onload = () => draw();
    bgImage.src = imgSrc;
}} else {{
    draw();
}}
updateUI();

canvas.addEventListener('click', (e) => {{
    if (closed) return;
    if (points.length >= maxPoints) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);
    points.push([x, y]);
    if ((mode === 'line' || mode === 'rect') && points.length === 2) {{
        closed = true;
    }}
    draw();
    updateUI();
}});

function clearAll() {{
    points = [];
    closed = false;
    draw();
    updateUI();
}}

function undoLast() {{
    if (points.length > 0) {{
        points.pop();
        closed = false;
        draw();
        updateUI();
    }}
}}

function closePolygon() {{
    if (mode === 'polygon' && points.length >= 3) {{
        closed = true;
        draw();
        updateUI();
    }}
}}

function draw() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (bgImage) {{
        ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);
    }} else {{
        ctx.fillStyle = '#1A1A2E';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#9CA3AF';
        ctx.font = '16px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('No image — upload in Data tab first, or draw directly', canvas.width/2, canvas.height/2 - 10);
        ctx.fillText('Click anywhere to place points', canvas.width/2, canvas.height/2 + 15);
    }}
    if (points.length === 0) return;

    const color = '#7C3AED';
    const colorFill = 'rgba(124, 58, 237, 0.15)';

    if (mode === 'line' && points.length === 2) {{
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        ctx.lineTo(points[1][0], points[1][1]);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.setLineDash([]);
        ctx.stroke();
        const mx = (points[0][0] + points[1][0]) / 2;
        const my = (points[0][1] + points[1][1]) / 2;
        ctx.beginPath();
        ctx.arc(mx, my, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#10B981';
        ctx.fill();
    }} else if (mode === 'rect' && points.length === 2) {{
        const x = Math.min(points[0][0], points[1][0]);
        const y = Math.min(points[0][1], points[1][1]);
        const w = Math.abs(points[1][0] - points[0][0]);
        const h = Math.abs(points[1][1] - points[0][1]);
        ctx.fillStyle = colorFill;
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
    }} else if (points.length >= 2) {{
        ctx.beginPath();
        ctx.moveTo(points[0][0], points[0][1]);
        for (let i = 1; i < points.length; i++) {{
            ctx.lineTo(points[i][0], points[i][1]);
        }}
        if (closed) {{
            ctx.closePath();
            ctx.fillStyle = colorFill;
            ctx.fill();
        }}
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.stroke();
    }}

    // Draw numbered points
    points.forEach((pt, i) => {{
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 8, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(i + 1), pt[0], pt[1]);
    }});
}}

function updateUI() {{
    const closeBtn = document.getElementById('closeBtn');
    const status = document.getElementById('status');
    closeBtn.style.display = (mode === 'polygon' && points.length >= 3 && !closed) ? 'inline-block' : 'none';

    if (closed) {{
        if (mode === 'line') status.textContent = 'Line set (' + points.length + ' points)';
        else if (mode === 'rect') status.textContent = 'Rectangle set';
        else status.textContent = 'Polygon closed (' + points.length + ' vertices)';
    }} else {{
        const needed = (mode === 'line' || mode === 'rect') ? 2 : 3;
        status.textContent = points.length + '/' + needed + '+ points — click to add';
    }}
}}
</script>
</body>
</html>
"""
