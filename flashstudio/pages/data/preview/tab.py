"""Preview tab — image display with COCO/YOLO bbox overlay."""

import os
import json
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS_ALL, BBOX_COLORS_RGB
from flashstudio.components.media import render_fixed_image
from flashstudio.pages.data._common import _load_data_prompt, make_thumbnail


def _render_preview():
    train_path = st.session_state.get("train_img_path", "")
    val_path = st.session_state.get("val_img_path", "")
    splits = []
    if train_path and os.path.isdir(train_path):
        splits.append("Train")
    if val_path and os.path.isdir(val_path):
        splits.append("Validation")
    if not splits:
        _load_data_prompt("No dataset loaded yet.", "preview_load_data")
        return

    if st.session_state.get("prev_split") not in splits:
        st.session_state["prev_split"] = splits[0]

    # One compact header row: split · boxes · navigation · position · jump.
    head = st.columns(
        [1.7, 0.8, 0.42, 0.42, 1.6, 0.42, 0.42, 1.0], gap="small"
    )
    with head[0]:
        split = st.radio(
            "Split", splits, horizontal=True, key="prev_split",
            label_visibility="collapsed", help="Dataset split to preview",
        )
    preview_path = train_path if split == "Train" else val_path
    if not preview_path or not os.path.isdir(preview_path):
        st.warning("Path not set.")
        return

    all_imgs = sorted(
        os.path.relpath(os.path.join(root, name), preview_path)
        for root, _dirs, files in os.walk(preview_path)
        for name in files
        if name.lower().endswith(IMG_EXTENSIONS_ALL)
    )
    if not all_imgs:
        st.warning("No images.")
        return

    with head[1]:
        bbox = st.checkbox(
            "Boxes", value=True, key="prev_bb",
            help="Draw annotation boxes on the images",
        )

    # Two large tiles per page: no count control, no layout jumps.
    page_size = 2
    if "prev_start" not in st.session_state:
        st.session_state["prev_start"] = 0
    if st.session_state["prev_start"] >= len(all_imgs):
        st.session_state["prev_start"] = 0
    start = st.session_state["prev_start"]
    max_start = max(len(all_imgs) - page_size, 0)

    def _goto(new_start: int):
        st.session_state["prev_start"] = max(0, min(new_start, max_start))
        st.rerun()

    with head[2]:
        if st.button("⏮", key="prev_first", help="First page",
                     use_container_width=True, disabled=start <= 0):
            _goto(0)
    with head[3]:
        if st.button("◀", key="prev_btn", help="Previous page",
                     use_container_width=True, disabled=start <= 0):
            _goto(start - page_size)

    end_idx = min(start + page_size, len(all_imgs))
    with head[4]:
        st.caption(f"{start + 1}–{end_idx} of {len(all_imgs):,} images")

    with head[5]:
        if st.button("▶", key="next_btn", help="Next page",
                     use_container_width=True, disabled=start >= max_start):
            _goto(start + page_size)
    with head[6]:
        if st.button("⏭", key="prev_last", help="Last page",
                     use_container_width=True, disabled=start >= max_start):
            _goto(max_start)

    # Jump box: keep its value in step with the visible page without
    # clobbering what the user is typing (Streamlit forbids writing widget
    # state after the widget exists, so a pending key bridges the rerun).
    pending_jump = st.session_state.pop("_prev_jump_pending", None)
    if pending_jump is not None:
        st.session_state["prev_jump"] = pending_jump
        st.session_state["_prev_jump_shown"] = pending_jump
    elif st.session_state.get("_prev_jump_shown") != start + 1:
        st.session_state["prev_jump"] = start + 1
        st.session_state["_prev_jump_shown"] = start + 1
    with head[7]:
        if "prev_jump" not in st.session_state:
            st.session_state["prev_jump"] = start + 1
        jump = st.number_input(
            "Image #", min_value=1, max_value=len(all_imgs),
            key="prev_jump",
            label_visibility="collapsed", help="Jump to an image number",
        )
    if jump is not None and int(jump) != start + 1:
        target = max(0, (int(jump) - 1) // page_size * page_size)
        st.session_state["prev_start"] = target
        st.session_state["_prev_jump_pending"] = target + 1
        st.rerun()

    # Load COCO annotations if available.  Search the split directory, a
    # parent folder, and an annotations/ sibling so uploads with the usual
    # dataset/train/images + dataset/annotations layout still preview boxes.
    from flashstudio.utils.dataset_paths import resolve_annotation_file

    split_name = "val" if split == "Validation" else "train"
    ann_file = resolve_annotation_file(
        preview_path,
        split=split_name,
        hint=st.session_state.get(f"{split_name}_ann_path", ""),
    )
    ann_data = None
    if ann_file and os.path.isfile(ann_file):
        try:
            with open(ann_file) as f:
                ann_data = json.load(f)
        except Exception:
            pass

    img_id_map, ann_by_img, cat_map = {}, {}, {}
    if ann_data:
        for c in ann_data.get("categories", []):
            cat_map[c["id"]] = c["name"]
        for im in ann_data.get("images", []):
            file_name = im.get("file_name", "")
            img_id_map[file_name] = im["id"]
            # Some COCO exports store subdirectories in file_name while the
            # preview lists the image directory recursively/flat.
            img_id_map.setdefault(os.path.basename(file_name), im["id"])
        for a in ann_data.get("annotations", []):
            ann_by_img.setdefault(a["image_id"], []).append(a)

    # YOLO label directory — check session state and the sibling labels folder
    from flashstudio.pages.data.fix.annotations import _yolo_label_dir

    yolo_label_dir = _yolo_label_dir(
        preview_path,
        st.session_state.get(f"{split_name}_label_path", ""),
    )
    try:
        has_labels = bool(yolo_label_dir) and any(
            name.endswith(".txt") for name in os.listdir(yolo_label_dir)
        )
    except OSError:
        has_labels = False
    if not has_labels:
        yolo_label_dir = ""

    from PIL import Image, ImageDraw

    def _label_path(img_name: str) -> str:
        if not yolo_label_dir:
            return ""
        lbl = os.path.join(
            yolo_label_dir, os.path.splitext(img_name)[0] + ".txt"
        )
        if os.path.isfile(lbl):
            return lbl
        lbl = os.path.join(
            yolo_label_dir,
            os.path.splitext(os.path.basename(img_name))[0] + ".txt",
        )
        return lbl if os.path.isfile(lbl) else ""

    def _boxes_for(img_name: str, w_img: int, h_img: int) -> list:
        """Boxes for one image in original coordinates: (x1, y1, x2, y2, color)."""
        boxes = []
        if ann_data and img_name in img_id_map:
            for a in ann_by_img.get(img_id_map[img_name], []):
                b = a.get("bbox", [])
                if len(b) == 4:
                    x, y, w, h = b
                    color = BBOX_COLORS_RGB[
                        a.get("category_id", 0) % len(BBOX_COLORS_RGB)
                    ]
                    boxes.append((x, y, x + w, y + h, color))
        elif yolo_label_dir and not ann_data:
            lbl = _label_path(img_name)
            if lbl:
                with open(lbl) as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy = float(parts[1]), float(parts[2])
                            bw, bh = float(parts[3]), float(parts[4])
                            x1 = (cx - bw / 2) * w_img
                            y1 = (cy - bh / 2) * h_img
                            x2 = (cx + bw / 2) * w_img
                            y2 = (cy + bh / 2) * h_img
                            color = BBOX_COLORS_RGB[
                                cls_id % len(BBOX_COLORS_RGB)
                            ]
                            boxes.append((x1, y1, x2, y2, color))
        return boxes

    selected = all_imgs[start:end_idx]
    cols = st.columns(2)
    for idx, col in enumerate(cols):
        if idx >= len(selected):
            break
        img_name = selected[idx]
        with col:
            try:
                img = Image.open(
                    os.path.join(preview_path, img_name)
                ).convert("RGB")
                w_img, h_img = img.size
                boxes = _boxes_for(img_name, w_img, h_img)
                if boxes and bbox:
                    draw = ImageDraw.Draw(img)
                    for x1, y1, x2, y2, color in boxes:
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                thumb = make_thumbnail(img, (480, 360))
                caption = f"{img_name[:18]} · {w_img}×{h_img}"
                if boxes:
                    caption += (
                        f" · {len(boxes)} box" if len(boxes) == 1
                        else f" · {len(boxes)} boxes"
                    )
                render_fixed_image(thumb, caption, height=360)
            except Exception as e:
                st.error(str(e)[:40])
