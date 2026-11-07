"""Data — Manual annotation editor."""

import json
import os

import streamlit as st
from PIL import Image, ImageDraw

from flashstudio.pages.data.fix.annotations import load_coco
from flashstudio.pages.data.fix.tab import _coco_path
from flashstudio.pages.data._common import _load_data_prompt
from flashstudio.constants import BBOX_COLORS_HEX


def _image_files(img_dir: str) -> list:
    if not os.path.isdir(img_dir):
        return []
    return sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    )


def _box_rows(data: dict, image_id: int) -> list:
    cats = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    rows = []
    for ann in data.get("annotations", []):
        if ann.get("image_id") != image_id:
            continue
        b = ann.get("bbox") or [0, 0, 0, 0]
        if len(b) != 4:
            continue
        rows.append({
            "Class": cats.get(ann.get("category_id"), str(ann.get("category_id"))),
            "x1": round(float(b[0]), 1),
            "y1": round(float(b[1]), 1),
            "x2": round(float(b[0]) + float(b[2]), 1),
            "y2": round(float(b[1]) + float(b[3]), 1),
        })
    return rows


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


def _render_edit():
    st.caption("Select an image, edit its boxes and classes, then save. A `.bak` backup is created.")

    split = st.radio("Split", ["Train", "Validation"], horizontal=True, key="edit_split")
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
    names = [img.get("file_name") for img in images if img.get("file_name")]
    if not names:
        st.info("No images in the annotation file.")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        selected_name = st.selectbox("Image", names, key="edit_image_select")
    with c2:
        st.caption(f"`{len(images)}` images")

    image_rec = next((i for i in images if i.get("file_name") == selected_name), None)
    if not image_rec:
        return
    image_path = os.path.join(img_dir, selected_name)
    if not os.path.isfile(image_path):
        st.warning(f"Image file is missing: `{image_path}`")
        return

    classes = [c.get("name") for c in data.get("categories", []) if c.get("name")]
    rows = _box_rows(data, image_rec["id"])

    edited = st.data_editor(
        rows,
        num_rows="dynamic",
        use_container_width=True,
        key=f"edit_grid_{image_rec['id']}",
        column_config={
            "Class": st.column_config.SelectboxColumn("Class", options=classes, required=True),
            "x1": st.column_config.NumberColumn("x1", format="%.1f"),
            "y1": st.column_config.NumberColumn("y1", format="%.1f"),
            "x2": st.column_config.NumberColumn("x2", format="%.1f"),
            "y2": st.column_config.NumberColumn("y2", format="%.1f"),
        },
    )

    # Preview with the current edited boxes.
    try:
        preview = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(preview)
        for i, row in enumerate(edited):
            try:
                x1, y1, x2, y2 = [float(row.get(k, 0) or 0) for k in ("x1", "y1", "x2", "y2")]
            except (TypeError, ValueError):
                continue
            color = BBOX_COLORS_HEX[i % len(BBOX_COLORS_HEX)]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 2, max(y1 - 12, 0)), str(row.get("Class", "")), fill=color)
        st.image(preview, caption=selected_name, use_container_width=True)
    except Exception as exc:
        st.caption(f"Preview unavailable: {exc}")

    b1, b2, b3 = st.columns([1, 1, 3])
    with b1:
        if st.button("Save boxes", type="primary", key="edit_save", use_container_width=True):
            try:
                added = _save_rows(ann_path, data, image_rec["id"], edited, classes)
                st.success(f"Saved {added} box(es) for `{selected_name}`.")
                st.session_state["_data_tab_next"] = "Preview"
                st.rerun()
            except Exception as exc:
                st.error(f"Save failed: {exc}")
    with b2:
        if st.button("Delete all", key="edit_delete_all", use_container_width=True):
            data["annotations"] = [
                a for a in data.get("annotations", []) if a.get("image_id") != image_rec["id"]
            ]
            import shutil
            shutil.copy2(ann_path, ann_path + ".bak")
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            st.success(f"Removed all boxes from `{selected_name}`.")
            st.session_state["_data_tab_next"] = "Preview"
            st.rerun()
    with b3:
        st.caption("Add a row, choose a class, enter x1/y1/x2/y2, then save.")
