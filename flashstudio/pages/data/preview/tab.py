"""Preview tab — image display with COCO/YOLO bbox overlay."""

import os
import json
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS_ALL, BBOX_COLORS_RGB, MAX_PREVIEW_COLS
from flashstudio.pages.data._common import _load_data_prompt


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

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        split = st.radio("Split", splits, horizontal=True, key="prev_split")
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

    with c2:
        max_count = min(12, len(all_imgs))
        if max_count <= 1:
            n = 1
        else:
            n = st.slider("Count", 1, max_count, min(4, max_count),
                          key="prev_n")
    with c3:
        bbox = st.checkbox("Boxes", value=True, key="prev_bb")

    if "prev_start" not in st.session_state:
        st.session_state["prev_start"] = 0
    if st.session_state["prev_start"] >= len(all_imgs):
        st.session_state["prev_start"] = 0
    start = st.session_state["prev_start"]

    nav_prev, nav_info, nav_next = st.columns([1, 3, 1])
    with nav_prev:
        if st.button("Prev", disabled=(start <= 0), key="prev_btn"):
            st.session_state["prev_start"] = max(0, start - n)
            st.rerun()
    with nav_info:
        end_idx = min(start + n, len(all_imgs))
        st.markdown(f'<p style="text-align:center;margin:0;padding:0.3rem 0;color:#6B7280;">'
                    f'{start + 1}–{end_idx} of {len(all_imgs)}</p>', unsafe_allow_html=True)
    with nav_next:
        if st.button("Next", disabled=(start + n >= len(all_imgs)), key="next_btn"):
            st.session_state["prev_start"] = min(start + n, max(len(all_imgs) - n, 0))
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
    selected = all_imgs[start:start + n]

    total_boxes = 0
    for row in range(0, len(selected), MAX_PREVIEW_COLS):
        cols = st.columns(MAX_PREVIEW_COLS)
        for j, col in enumerate(cols):
            idx = row + j
            if idx >= len(selected):
                break
            img_name = selected[idx]
            with col:
                try:
                    img = Image.open(os.path.join(preview_path, img_name)).convert("RGB")
                    w_img, h_img = img.size
                    if bbox:
                        draw = ImageDraw.Draw(img)
                        drawn = False
                        if ann_data and img_name in img_id_map:
                            for a in ann_by_img.get(img_id_map[img_name], []):
                                b = a.get("bbox", [])
                                if len(b) == 4:
                                    x, y, w, h = b
                                    color = BBOX_COLORS_RGB[a.get("category_id", 0) % len(BBOX_COLORS_RGB)]
                                    draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
                                    total_boxes += 1
                                    drawn = True
                        if not drawn and yolo_label_dir and not ann_data:
                            lbl_name = os.path.splitext(img_name)[0] + ".txt"
                            lbl_path = os.path.join(yolo_label_dir, lbl_name)
                            if not os.path.isfile(lbl_path):
                                lbl_path = os.path.join(
                                    yolo_label_dir,
                                    os.path.splitext(os.path.basename(img_name))[0] + ".txt",
                                )
                            if os.path.isfile(lbl_path):
                                with open(lbl_path) as lf:
                                    for line in lf:
                                        parts = line.strip().split()
                                        if len(parts) >= 5:
                                            cls_id = int(parts[0])
                                            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                            x1 = int((cx - bw / 2) * w_img)
                                            y1 = int((cy - bh / 2) * h_img)
                                            x2 = int((cx + bw / 2) * w_img)
                                            y2 = int((cy + bh / 2) * h_img)
                                            color = BBOX_COLORS_RGB[cls_id % len(BBOX_COLORS_RGB)]
                                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                                            total_boxes += 1
                    st.image(img, caption=f"{img_name[:15]} {w_img}x{h_img}", use_container_width=True)
                except Exception as e:
                    st.error(str(e)[:40])

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Images", len(all_imgs))
    with m2:
        ann_count = len(ann_data.get("annotations", [])) if ann_data else total_boxes
        st.metric("Annotations", ann_count)
    with m3:
        cat_count = len(cat_map) if cat_map else "—"
        st.metric("Categories", cat_count)
