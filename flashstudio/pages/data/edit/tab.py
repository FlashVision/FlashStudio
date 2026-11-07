"""Data — Manual annotation editor."""

import json
import os

import streamlit as st
from PIL import Image, ImageDraw

from flashstudio.constants import BBOX_COLORS_HEX
from flashstudio.pages.data._common import _load_data_prompt, make_thumbnail
from flashstudio.pages.data.fix.annotations import load_coco
from flashstudio.pages.data.fix.tab import _coco_path
from flashstudio.utils import flash


def _save_rows(ann_path: str, data: dict, image_id: int, rows: list, classes: list):
    """Replace one image's boxes with edited rows and write a backup."""
    import shutil
    cats_by_name = {c.get("name"): c for c in data.get("categories", [])}
    # Add missing categories.
    for name in classes:
        if name and name not in cats_by_name:
            new_id = max([c.get("id", 0) for c in data.get("categories", [])] + [0]) + 1
            data.setdefault("categories", []).append({"id": new_id, "name": name, "supercategory": "object"})
            cats_by_name[name] = data["categories"][-1]
    cat_by_name = {c.get("name"): c.get("id") for c in data.get("categories", [])}

    image = next((i for i in data.get("images", []) if i.get("id") == image_id), None)
    iw = float(image.get("width", 0) or 0) if image else 0
    ih = float(image.get("height", 0) or 0) if image else 0

    data["annotations"] = [a for a in data.get("annotations", []) if a.get("image_id") != image_id]
    next_id = max([a.get("id", 0) for a in data.get("annotations", [])] + [0]) + 1
    added = 0
    for row in rows:
        name = str(row.get("Class", "") or "").strip()
        if not name:
            continue
        try:
            x1, y1, x2, y2 = [float(row.get(k, 0) or 0) for k in ("x1", "y1", "x2", "y2")]
        except (TypeError, ValueError):
            continue
        if iw > 0:
            x1, x2 = max(0.0, min(x1, iw)), max(0.0, min(x2, iw))
        if ih > 0:
            y1, y2 = max(0.0, min(y1, ih)), max(0.0, min(y2, ih))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        data["annotations"].append({
            "id": next_id,
            "image_id": image_id,
            "category_id": cat_by_name[name],
            "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
            "area": round((x2 - x1) * (y2 - y1), 2),
            "iscrowd": 0,
        })
        next_id += 1
        added += 1
    shutil.copy2(ann_path, ann_path + ".bak")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return added


_CANVAS_W, _CANVAS_H = 860, 460


def _bump_edit_canvas(image_id: int) -> None:
    """Reset the drawing canvas so newly added boxes clear from the layer."""
    nonce_map = dict(st.session_state.get("_edit_canvas_nonce", {}))
    nonce_map[image_id] = nonce_map.get(image_id, 0) + 1
    st.session_state["_edit_canvas_nonce"] = nonce_map


def _draw_boxes_on_canvas(image, rows, scale, off_x, off_y,
                          opacity=0.9, show_labels=True):
    """Project an image to the fixed canvas and burn in existing boxes."""
    from PIL import ImageColor

    base = make_thumbnail(image, (_CANVAS_W, _CANVAS_H)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = max(0, min(255, int(255 * opacity)))
    for i, row in enumerate(rows):
        try:
            x1, y1, x2, y2 = [
                float(row.get(k, 0) or 0) for k in ("x1", "y1", "x2", "y2")
            ]
        except (TypeError, ValueError):
            continue
        cx1, cy1 = x1 * scale + off_x, y1 * scale + off_y
        cx2, cy2 = x2 * scale + off_x, y2 * scale + off_y
        rgb = ImageColor.getrgb(BBOX_COLORS_HEX[i % len(BBOX_COLORS_HEX)])
        color = (rgb[0], rgb[1], rgb[2], alpha)
        draw.rectangle([cx1, cy1, cx2, cy2], outline=color, width=2)
        label = str(row.get("Class", "") or "")
        if show_labels and label:
            draw.text((cx1 + 2, max(cy1 - 12, 0)), label, fill=color)
    return Image.alpha_composite(base, overlay).convert("RGB")


def _drawn_rectangles(result, scale, off_x, off_y, img_w, img_h):
    """Convert canvas rectangles back into original image coordinates."""
    if result is None or getattr(result, "json_data", None) is None:
        return []
    rects = []
    for obj in result.json_data.get("objects", []):
        if obj.get("type") != "rect":
            continue
        width = float(obj.get("width", 0) or 0) * abs(
            float(obj.get("scaleX", 1) or 1)
        )
        height = float(obj.get("height", 0) or 0) * abs(
            float(obj.get("scaleY", 1) or 1)
        )
        left = float(obj.get("left", 0) or 0)
        top = float(obj.get("top", 0) or 0)
        x1, y1 = (left - off_x) / scale, (top - off_y) / scale
        x2, y2 = x1 + width / scale, y1 + height / scale
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        x1, x2 = max(0.0, min(x1, img_w)), max(0.0, min(x2, img_w))
        y1, y2 = max(0.0, min(y1, img_h)), max(0.0, min(y2, img_h))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            rects.append((x1, y1, x2, y2))
    return rects


def _render_edit():
    # One compact header row: split · filter · image navigation · status.
    head = st.columns(
        [1.75, 1.5, 0.42, 0.42, 3.9, 0.42, 0.42, 1.9], gap="small"
    )
    with head[0]:
        split = st.radio(
            "Split", ["Train", "Validation"], horizontal=True,
            key="edit_split", label_visibility="collapsed",
            help="Dataset split to edit",
        )

    img_key = "train_img_path" if split == "Train" else "val_img_path"
    img_dir = st.session_state.get(img_key, "")
    if not img_dir or not os.path.isdir(img_dir):
        _load_data_prompt(
            f"No {split.lower()} images directory is configured. Load a dataset first.",
            "edit_load_data",
        )
        return

    ann_path = _coco_path(img_dir, "train" if split == "Train" else "val")
    if not ann_path:
        st.info("No COCO annotation file found for this split.")
        return

    try:
        data = load_coco(ann_path)
    except Exception as exc:
        st.error(f"Could not load annotations: {exc}")
        return

    images = data.get("images", [])
    if not images:
        st.info("No images in the annotation file.")
        return

    with head[1]:
        filt = st.selectbox(
            "Filter", ["All images", "Needs labels", "Labeled"],
            key="edit_filter", label_visibility="collapsed",
            help="List every image, only the unlabeled ones, or only the labeled ones",
        )

    box_counts = {}
    for ann in data.get("annotations", []):
        aid = ann.get("image_id")
        box_counts[aid] = box_counts.get(aid, 0) + 1

    def _matches(rec):
        count = box_counts.get(rec.get("id"), 0)
        if filt == "Needs labels":
            return count == 0
        if filt == "Labeled":
            return count > 0
        return True

    visible = [r for r in images if r.get("file_name") and _matches(r)]
    names = [r.get("file_name") for r in visible]
    if not names:
        st.info(f'No images match the "{filt}" filter. Pick another filter.')
        return

    # Navigation buttons queue a target and rerun; apply it before the
    # selectbox is created so its widget state is never modified in place.
    pending_goto = st.session_state.pop("_edit_goto_name", None)
    if pending_goto in names:
        st.session_state["edit_image_select"] = pending_goto

    current_name = st.session_state.get("edit_image_select")
    if current_name not in names:
        current_name = names[0]
        st.session_state["edit_image_select"] = current_name
    current_idx = names.index(current_name)
    current_rec = next(
        (r for r in visible if r.get("file_name") == current_name), None
    )
    if not current_rec:
        return
    pending_key = f"_edit_new_boxes_{current_rec['id']}"
    has_unsaved = bool(st.session_state.get(pending_key))

    def _goto(index: int):
        st.session_state["_edit_goto_name"] = names[index]
        st.rerun()

    with head[2]:
        if st.button("⏮", key="edit_first", help="First image",
                     use_container_width=True,
                     disabled=current_idx == 0 or has_unsaved):
            _goto(0)
    with head[3]:
        if st.button("◀", key="edit_prev", help="Previous image",
                     use_container_width=True,
                     disabled=current_idx == 0 or has_unsaved):
            _goto(current_idx - 1)
    with head[4]:
        st.selectbox(
            "Image", names, key="edit_image_select",
            label_visibility="collapsed",
        )
    with head[5]:
        if st.button("▶", key="edit_next", help="Next image",
                     use_container_width=True,
                     disabled=current_idx >= len(names) - 1 or has_unsaved):
            _goto(current_idx + 1)
    with head[6]:
        if st.button("⏭", key="edit_last", help="Last image",
                     use_container_width=True,
                     disabled=current_idx >= len(names) - 1 or has_unsaved):
            _goto(len(names) - 1)

    image_rec = current_rec
    image_path = os.path.join(img_dir, current_name)
    if not os.path.isfile(image_path):
        st.warning(f"Image file is missing: `{image_path}`")
        return

    classes = [
        c.get("name") for c in data.get("categories", []) if c.get("name")
    ]
    pending = list(st.session_state.get(pending_key, []))
    cat_names = {
        c.get("id"): c.get("name", str(c.get("id")))
        for c in data.get("categories", [])
    }
    pairs = []
    for ann in data.get("annotations", []):
        if ann.get("image_id") != image_rec["id"]:
            continue
        b = ann.get("bbox") or []
        if len(b) != 4:
            continue
        pairs.append((ann, {
            "Class": cat_names.get(
                ann.get("category_id"), str(ann.get("category_id"))
            ),
            "x1": round(float(b[0]), 1),
            "y1": round(float(b[1]), 1),
            "x2": round(float(b[0]) + float(b[2]), 1),
            "y2": round(float(b[1]) + float(b[3]), 1),
        }))
    saved_anns = [a for a, _ in pairs]
    rows = [r for _, r in pairs] + pending

    orig_w = float(image_rec.get("width") or 0)
    orig_h = float(image_rec.get("height") or 0)
    try:
        image = Image.open(image_path)
        native_w, native_h = image.size
        if orig_w <= 0 or orig_h <= 0:
            orig_w, orig_h = float(native_w), float(native_h)
        # Decode at a reduced resolution for the projection only; annotation
        # coordinates below stay in the real image width/height.
        image.draft("RGB", (_CANVAS_W * 2, _CANVAS_H * 2))
        if image.size[0] > _CANVAS_W * 2 or image.size[1] > _CANVAS_H * 2:
            image.thumbnail((_CANVAS_W * 2, _CANVAS_H * 2))
        image = image.convert("RGB")
    except Exception as exc:
        st.error(f"Could not open image: {exc}")
        return
    img_w, img_h = orig_w, orig_h

    with head[7]:
        note = (
            f"{current_idx + 1} / {len(names)} · "
            f"{img_w:.0f}×{img_h:.0f} · {len(rows)} boxes"
        )
        if has_unsaved:
            note += " · unsaved"
        st.caption(note)

    scale = min(_CANVAS_W / max(img_w, 1), _CANVAS_H / max(img_h, 1))
    off_x = (_CANVAS_W - img_w * scale) / 2
    off_y = (_CANVAS_H - img_h * scale) / 2

    from flashstudio.components import zone_drawer as _zd

    rail_col, canvas_col, panel_col = st.columns(
        [0.8, 4.2, 2.1], gap="small"
    )
    with rail_col:
        st.caption("Tools")
        tool = st.radio(
            "Tool", ["▭ Rect", "✥ Move"], key="edit_tool",
            label_visibility="collapsed",
            help="Rect draws new boxes; Move adjusts shapes on the canvas",
        )
        st.slider(
            "Opacity", 0.2, 1.0, 0.9, step=0.1, key="edit_box_opacity",
            help="Fade the boxes so the image underneath stays readable",
        )
        st.checkbox(
            "Labels", value=True, key="edit_show_labels",
            help="Show class names on the canvas",
        )
    drawing_mode = "rect" if tool.startswith("▭") else "transform"
    box_opacity = float(st.session_state.get("edit_box_opacity", 0.9))
    show_labels = bool(st.session_state.get("edit_show_labels", True))

    canvas_bg = _draw_boxes_on_canvas(
        image, rows, scale, off_x, off_y,
        opacity=box_opacity, show_labels=show_labels,
    )

    canvas_result = None
    nonce = int(
        st.session_state.get("_edit_canvas_nonce", {}).get(image_rec["id"], 0)
    )
    with canvas_col:
        if getattr(_zd, "_HAS_CANVAS", False):
            canvas_result = _zd.st_canvas(
                fill_color="rgba(34, 197, 94, 0.12)",
                stroke_width=2,
                stroke_color="#22C55E",
                background_image=canvas_bg,
                drawing_mode=drawing_mode,
                width=_CANVAS_W,
                height=_CANVAS_H,
                key=f"edit_canvas_{image_rec['id']}_{nonce}",
            )
        else:
            st.image(canvas_bg, width=_CANVAS_W)
            st.warning(
                "Install `streamlit-drawable-canvas` to draw boxes on the image. "
                "Numeric editing below still works."
            )

    with panel_col:
        obj_tab, labels_tab = st.tabs(["Objects", "Labels"])
        with obj_tab:
            st.caption(f"{len(rows)} boxes — pick a class to change it")
            with st.container(height=max(240, _CANVAS_H - 70), border=False):
                if not rows:
                    st.caption("No boxes yet. Draw one on the image.")
                class_changes = []

                def _delete_box(index: int):
                    import shutil

                    if index < len(saved_anns):
                        ann_id = saved_anns[index].get("id")
                        data["annotations"] = [
                            a for a in data.get("annotations", [])
                            if a.get("id") != ann_id
                        ]
                        shutil.copy2(ann_path, ann_path + ".bak")
                        with open(ann_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                    else:
                        pending_index = index - len(saved_anns)
                        if 0 <= pending_index < len(pending):
                            pending.pop(pending_index)
                            st.session_state[pending_key] = pending
                    _bump_edit_canvas(image_rec["id"])
                    flash("Deleted box", "info")
                    st.rerun()

                for i, row in enumerate(rows[:200]):
                    color = BBOX_COLORS_HEX[i % len(BBOX_COLORS_HEX)]
                    label = str(row.get("Class", "") or "—")
                    try:
                        x1 = float(row.get("x1", 0) or 0)
                        y1 = float(row.get("y1", 0) or 0)
                        x2 = float(row.get("x2", 0) or 0)
                        y2 = float(row.get("y2", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    rc1, rc2, rc3, rc4 = st.columns(
                        [2.6, 2.2, 0.55, 0.55], gap="small"
                    )
                    with rc1:
                        options = classes or [label]
                        try:
                            start = options.index(label)
                        except ValueError:
                            start = 0
                        new_cls = st.selectbox(
                            "Class", options, index=start,
                            key=f"edit_cls_{image_rec['id']}_{i}",
                            label_visibility="collapsed",
                        )
                    with rc2:
                        st.markdown(
                            f'<div style="font-size: var(--fs-text-xs);'
                            f'color:var(--fs-muted);padding-top:.5rem;'
                            f'white-space:nowrap;">'
                            f'<span style="color:{color};">&#9632;</span> '
                            f'{x1:.0f},{y1:.0f} → {x2:.0f},{y2:.0f}</div>',
                            unsafe_allow_html=True,
                        )
                    with rc3:
                        if st.button(
                            "⧉", key=f"edit_dup_{image_rec['id']}_{i}",
                            help="Duplicate this box", use_container_width=True,
                        ):
                            pending.append(dict(row))
                            st.session_state[pending_key] = pending
                            _bump_edit_canvas(image_rec["id"])
                            flash(
                                "Box duplicated — press Save boxes to keep it.",
                                "info",
                            )
                            st.rerun()
                    with rc4:
                        if st.button(
                            "×", key=f"edit_del_{image_rec['id']}_{i}",
                            help="Delete this box", use_container_width=True,
                        ):
                            _delete_box(i)
                    if new_cls and new_cls != label:
                        class_changes.append((i, new_cls))
                if len(rows) > 200:
                    st.caption(f"+{len(rows) - 200} more — use Numeric editor")

            if class_changes:
                import shutil

                cat_by_name = {
                    c.get("name"): c.get("id")
                    for c in data.get("categories", [])
                }
                changed_saved = False
                for index, new_cls in class_changes:
                    if index < len(saved_anns):
                        cat_id = cat_by_name.get(new_cls)
                        if cat_id is not None:
                            saved_anns[index]["category_id"] = cat_id
                            changed_saved = True
                    else:
                        pending_index = index - len(saved_anns)
                        if 0 <= pending_index < len(pending):
                            pending[pending_index]["Class"] = new_cls
                if changed_saved:
                    shutil.copy2(ann_path, ann_path + ".bak")
                    with open(ann_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                st.session_state[pending_key] = pending
                _bump_edit_canvas(image_rec["id"])
                flash("Class updated.", "success")
                st.rerun()

        with labels_tab:
            if st.session_state.pop("_edit_clear_label", False):
                st.session_state["edit_new_label"] = ""
            counts = {}
            for row in rows:
                key = str(row.get("Class", "") or "—")
                counts[key] = counts.get(key, 0) + 1
            if not counts:
                st.caption("No labels on this image yet.")
            for i, cls in enumerate(classes):
                color = BBOX_COLORS_HEX[i % len(BBOX_COLORS_HEX)]
                st.markdown(
                    f'<div style="display:flex;gap:.4rem;align-items:center;'
                    f'padding:.18rem 0;">'
                    f'<span style="color:{color};font-size: var(--fs-text-sm);">&#9632;</span>'
                    f'<span style="font-size: var(--fs-text-md);">{cls}</span>'
                    f'<span style="font-size: var(--fs-text-sm);color:var(--fs-muted);'
                    f'margin-left:auto;">{counts.get(cls, 0)}</span></div>',
                    unsafe_allow_html=True,
                )
            st.divider()
            lc1, lc2 = st.columns([3, 1], gap="small")
            with lc1:
                new_label = st.text_input(
                    "New class", key="edit_new_label",
                    label_visibility="collapsed",
                    placeholder="Add a class name…",
                )
            with lc2:
                add_label = st.button(
                    "Add", key="edit_add_label", use_container_width=True,
                    disabled=not (new_label or "").strip(),
                )
            if add_label:
                name = new_label.strip()
                existing = {c.get("name") for c in data.get("categories", [])}
                if name and name not in existing:
                    import shutil

                    new_id = max(
                        [c.get("id", 0) for c in data.get("categories", [])] + [0]
                    ) + 1
                    data.setdefault("categories", []).append(
                        {"id": new_id, "name": name, "supercategory": "object"}
                    )
                    shutil.copy2(ann_path, ann_path + ".bak")
                    with open(ann_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    st.session_state["_edit_clear_label"] = True
                    flash(f"Class `{name}` added.", "success")
                    st.rerun()

    ac1, ac2, ac3, ac4 = st.columns([2, 1.4, 1, 1], gap="small")
    with ac1:
        new_class = (
            st.selectbox(
                "New box class", classes,
                key=f"edit_new_class_{image_rec['id']}",
                label_visibility="collapsed",
                help="Class applied to boxes you draw",
            )
            if classes else ""
        )
    with ac2:
        add_drawn = st.button(
            "Add drawn boxes",
            key=f"edit_add_drawn_{image_rec['id']}",
            use_container_width=True,
            disabled=not classes,
        )
    with ac3:
        save_clicked = st.button(
            "Save boxes", type="primary", key="edit_save",
            use_container_width=True,
        )
    with ac4:
        delete_clicked = st.button(
            "Delete all", key="edit_delete_all", use_container_width=True,
        )

    with st.expander("Numeric editor", expanded=False):
        edited = st.data_editor(
            rows,
            num_rows="dynamic",
            use_container_width=True,
            height=240,
            key=f"edit_grid_{image_rec['id']}_{len(pending)}",
            column_config={
                "Class": st.column_config.SelectboxColumn(
                    "Class", options=classes, required=True,
                ),
                "x1": st.column_config.NumberColumn("x1", format="%.1f"),
                "y1": st.column_config.NumberColumn("y1", format="%.1f"),
                "x2": st.column_config.NumberColumn("x2", format="%.1f"),
                "y2": st.column_config.NumberColumn("y2", format="%.1f"),
            },
        )

    if add_drawn:
        new_rects = _drawn_rectangles(
            canvas_result, scale, off_x, off_y, img_w, img_h
        )
        if not new_rects:
            st.info("Draw at least one rectangle on the image first.")
        else:
            pending.extend(
                {
                    "Class": new_class,
                    "x1": round(x1, 1), "y1": round(y1, 1),
                    "x2": round(x2, 1), "y2": round(y2, 1),
                }
                for x1, y1, x2, y2 in new_rects
            )
            st.session_state[pending_key] = pending
            _bump_edit_canvas(image_rec["id"])
            flash(
                f"Added {len(new_rects)} box(es) to the object list. "
                "Press Save boxes to write them.",
                "info",
            )
            st.rerun()

    if save_clicked:
        try:
            added = _save_rows(
                ann_path, data, image_rec["id"], edited, classes
            )
            st.session_state.pop(pending_key, None)
            _bump_edit_canvas(image_rec["id"])
            flash(f"Saved {added} box(es) for `{current_name}`.", "success")
            st.rerun()
        except Exception as exc:
            st.error(f"Save failed: {exc}")

    if delete_clicked:
        data["annotations"] = [
            a for a in data.get("annotations", [])
            if a.get("image_id") != image_rec["id"]
        ]
        import shutil

        shutil.copy2(ann_path, ann_path + ".bak")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        st.session_state.pop(pending_key, None)
        _bump_edit_canvas(image_rec["id"])
        flash(f"Removed all boxes from `{current_name}`.", "info")
        st.rerun()
