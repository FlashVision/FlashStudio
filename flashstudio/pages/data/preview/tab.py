"""Preview tab — image display with COCO/YOLO bbox overlay."""

import os
import json
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS, BBOX_COLORS_RGB, MAX_PREVIEW_COLS


def _render_preview():
    train_path = st.session_state.get("train_img_path", "")
    val_path = st.session_state.get("val_img_path", "")
    if not train_path or not os.path.isdir(train_path):
        st.info("No dataset loaded yet.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        split = st.radio("Split", ["Train", "Val"], horizontal=True, key="prev_split")
    preview_path = train_path if split == "Train" else val_path
    if not preview_path or not os.path.isdir(preview_path):
        st.warning("Path not set.")
        return

    all_imgs = sorted([f for f in os.listdir(preview_path) if f.lower().endswith(IMG_EXTENSIONS)])
    if not all_imgs:
        st.warning("No images.")
        return

    with c2:
        n = st.slider("Count", 1, min(12, len(all_imgs)), min(4, len(all_imgs)), key="prev_n")
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

    # Load COCO annotations if available
    ann_file = os.path.join(preview_path, "_annotations.coco.json")
    ann_data = None
    if os.path.isfile(ann_file):
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
            img_id_map[im["file_name"]] = im["id"]
        for a in ann_data.get("annotations", []):
            ann_by_img.setdefault(a["image_id"], []).append(a)

    # YOLO label directory — check sibling 'labels' folder or session state
    yolo_label_dir = st.session_state.get(f"{split.lower()}_label_path", "")
    if not yolo_label_dir or not os.path.isdir(yolo_label_dir):
        candidate = preview_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        if os.path.isdir(candidate):
            yolo_label_dir = candidate
        else:
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
                        if not drawn and yolo_label_dir:
                            lbl_name = os.path.splitext(img_name)[0] + ".txt"
                            lbl_path = os.path.join(yolo_label_dir, lbl_name)
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
