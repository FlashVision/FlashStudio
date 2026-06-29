"""Inference — Solution tab."""

import streamlit as st
from flashstudio.constants import (
    INFER_DISPLAY_WIDTH, INFER_DEFAULT_RESOLUTION,
    ZONE_MAX_X, ZONE_MAX_Y, ZONE_DEFAULT_LINE, ZONE_DEFAULT_POLYGON,
)
from flashstudio.pages.inference._common import (
    SOLUTIONS, _SOLUTIONS_AVAILABLE, _solution_classes, _HAS_ZONE_DRAWER,
)
from flashstudio.pages.inference.data.tab import _get_first_frame

if _HAS_ZONE_DRAWER:
    from flashstudio.components.zone_drawer import zone_drawer


def _tab_solution():
    sc1, sc2 = st.columns([2, 3])
    with sc1:
        selected = st.selectbox("Solution", list(SOLUTIONS.keys()), key="selected_solution",
                                format_func=lambda x: x)
        sol = SOLUTIONS[selected]
        st.caption(sol["desc"])
        avail = len(_SOLUTIONS_AVAILABLE)
        total = len(_solution_classes)
        st.caption(f"{avail}/{total} solutions available")

    with sc2:
        if sol["needs_zone"]:
            st.info("Zone required — draw below using Polygon, Rectangle, or Line.")
        else:
            st.caption("Zone optional — draw one to limit detection area.")
        has_line = bool(st.session_state.get("zone_line_points"))
        has_poly = bool(st.session_state.get("zone_polygons"))
        if has_line or has_poly:
            zone_type = "Line" if has_line else "Polygon"
            st.success(f"Zone set: {zone_type}")
        elif sol["needs_zone"]:
            st.warning("No zone defined yet")

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
        result = zone_drawer(image=frame_img, mode=draw_mode, points=existing_points,
                             closed=existing_closed, display_width=INFER_DISPLAY_WIDTH)
        if result and result.get("points"):
            pts = result["points"]
            img_w = frame_img.size[0] if frame_img else INFER_DEFAULT_RESOLUTION[0]
            img_h = frame_img.size[1] if frame_img else INFER_DEFAULT_RESOLUTION[1]
            dw = result.get("displayWidth", INFER_DISPLAY_WIDTH)
            dh = result.get("displayHeight", INFER_DEFAULT_RESOLUTION[1])
            sx = img_w / dw if dw else 1
            sy = img_h / dh if dh else 1
            _store_zone_coords(pts, draw_mode, sx, sy)
    else:
        st.warning("Zone drawing component not available.")

    with st.expander("Set coordinates manually", expanded=not _HAS_ZONE_DRAWER):
        _manual_zone_input(draw_mode)


def _manual_zone_input(draw_mode):
    if draw_mode == "line":
        c1, c2 = st.columns(2)
        with c1:
            x1 = st.number_input("X1", 0, ZONE_MAX_X, ZONE_DEFAULT_LINE[0][0], key="manual_line_x1")
            y1 = st.number_input("Y1", 0, ZONE_MAX_Y, ZONE_DEFAULT_LINE[0][1], key="manual_line_y1")
        with c2:
            x2 = st.number_input("X2", 0, ZONE_MAX_X, ZONE_DEFAULT_LINE[1][0], key="manual_line_x2")
            y2 = st.number_input("Y2", 0, ZONE_MAX_Y, ZONE_DEFAULT_LINE[1][1], key="manual_line_y2")
        if st.button("Set Line", key="set_manual_line"):
            st.session_state["zone_line_points"] = [(x1, y1), (x2, y2)]
            st.success(f"({x1},{y1}) → ({x2},{y2})")
    else:
        pts = st.text_area("Points (x,y per line)", ZONE_DEFAULT_POLYGON, key="manual_polygon_text")
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
