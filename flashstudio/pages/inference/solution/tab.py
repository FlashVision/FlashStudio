"""Inference — Solution tab."""

import streamlit as st
from PIL import ImageDraw
from flashstudio.constants import (
    INFER_DISPLAY_WIDTH,
    ZONE_MAX_X, ZONE_MAX_Y, ZONE_DEFAULT_LINE, ZONE_DEFAULT_POLYGON,
)
from flashstudio.pages.inference._common import (
    SOLUTIONS, _SOLUTIONS_AVAILABLE, _solution_classes, _HAS_ZONE_DRAWER,
)
from flashstudio.pages.inference.data.tab import _get_first_frame

if _HAS_ZONE_DRAWER:
    from flashstudio.components.zone_drawer import zone_drawer


def _tab_solution():
    sc1, sc2 = st.columns([2, 1])
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

    left, right = st.columns([2, 1])
    with left:
        with st.container(border=True):
            st.markdown("#### Zone editor")
            draw_mode = st.radio(
                "Drawing mode", mode_options, index=default_idx, horizontal=True,
                key="zone_draw_mode",
                format_func=lambda x: {"polygon": "Polygon", "rect": "Rectangle", "line": "Line"}[x],
            )
            st.checkbox("Add another zone", value=False, key="zone_append",
                        help="Keep existing polygon/rectangle zones and append the next drawing.")

            frame_img = _get_first_frame()

            if draw_mode == "line":
                existing_points = st.session_state.get("zone_line_points") or []
            else:
                polygons = st.session_state.get("zone_polygons") or []
                existing_points = polygons[-1] if polygons else []
            existing_closed = bool(existing_points) and draw_mode in ("rect", "polygon")

            if frame_img is None:
                st.caption("No input image yet — the canvas is blank. Add an image or video in the Data tab.")
            else:
                st.caption("Draw on the first frame; use Clear zone to reset.")

            if _HAS_ZONE_DRAWER:
                result = zone_drawer(image=frame_img, mode=draw_mode, points=existing_points,
                                     closed=existing_closed, display_width=INFER_DISPLAY_WIDTH)
                if result and result.get("points"):
                    # zone_drawer already returns points in the source image's
                    # coordinate space; scaling again here was a bug.
                    _store_zone_coords(result["points"], draw_mode)
            else:
                st.warning("Zone drawing component not available.")

            if st.button("Clear zone", key="clear_zone_btn"):
                for key in ("zone_line_points", "zone_polygons", "zone_draw_points", "zone_closed"):
                    st.session_state.pop(key, None)
                st.rerun()

    with right:
        with st.container(border=True):
            st.markdown("#### Zone status")
            line = st.session_state.get("zone_line_points") or []
            polygons = st.session_state.get("zone_polygons") or []
            if draw_mode == "line" and line:
                st.caption(f"Line · {tuple(line[0])} → {tuple(line[1])}")
            elif polygons:
                shape = "Rectangle" if draw_mode == "rect" and len(polygons) == 1 else "Polygon"
                st.caption(f"{shape} · {len(polygons)} zone(s) · {len(polygons[-1])} points")
            else:
                st.caption("No zone set yet.")

            if frame_img is not None and (line or polygons):
                preview = frame_img.copy().convert("RGB")
                draw = ImageDraw.Draw(preview)
                for zone in polygons:
                    pts = [tuple(p) for p in zone]
                    if len(pts) >= 2:
                        draw.line(pts + [pts[0]], fill="#7C3AED", width=4)
                if len(line) >= 2:
                    draw.line([tuple(line[0]), tuple(line[1])], fill="#F59E0B", width=4)
                st.image(preview, caption="Zone preview", use_container_width=True)

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
            st.session_state.pop("zone_polygons", None)
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
                if draw_mode == "rect":
                    xs = [p[0] for p in parsed]
                    ys = [p[1] for p in parsed]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
                    st.session_state["zone_polygons"] = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]
                    st.session_state.pop("zone_line_points", None)
                    st.success(f"Rectangle: ({x1},{y1}) → ({x2},{y2})")
                else:
                    st.session_state["zone_polygons"] = [parsed]
                    st.session_state.pop("zone_line_points", None)
                    st.success(f"{len(parsed)} vertices")


def _store_zone_coords(points, draw_mode):
    """Persist canvas points (already in source-image coordinates)."""
    orig = [(int(x), int(y)) for x, y in points]
    if draw_mode == "line" and len(orig) >= 2:
        st.session_state["zone_line_points"] = [orig[0], orig[1]]
        st.session_state.pop("zone_polygons", None)
        st.success(f"Line: {orig[0]} → {orig[1]}")
        return
    if draw_mode in ("rect", "polygon"):
        if draw_mode == "rect" and len(orig) >= 2:
            xs = [p[0] for p in orig]
            ys = [p[1] for p in orig]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            zone = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            _save_zone(zone)
            st.session_state.pop("zone_line_points", None)
            st.success(f"Rect: ({x1},{y1})→({x2},{y2})")
        elif len(orig) >= 3:
            _save_zone(orig)
            st.session_state.pop("zone_line_points", None)
            st.success(f"Polygon: {len(orig)} pts") if st.session_state.get("zone_closed") else st.info(
                f"{len(orig)} pts, click more or close"
            )


def _save_zone(zone):
    """Store or append a polygon/rectangle zone."""
    if st.session_state.get("zone_append"):
        zones = list(st.session_state.get("zone_polygons") or [])
        if zone not in zones:
            zones.append(zone)
        st.session_state["zone_polygons"] = zones
    else:
        st.session_state["zone_polygons"] = [zone]
