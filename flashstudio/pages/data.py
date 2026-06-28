"""FlashStudio — Data Page (ultra-compact, no-scroll)."""

import os
import json
import streamlit as st
try:
    from flashdet import download_dataset, list_datasets
    from flashdet.data import convert_dataset, detect_dataset_format, verify_dataset, summarize_coco_root
    _HAS_FLASHDET = True
except ImportError:
    _HAS_FLASHDET = False

_FLASHDET_INSTALL_MSG = "FlashDet required — install with: `pip install flashdet`"


def _extract_classes_from_coco_json(ann_path: str) -> list[str]:
    """Read class names from a COCO-format annotation JSON file."""
    try:
        with open(ann_path, encoding="utf-8") as f:
            data = json.load(f)
        cats = data.get("categories", [])
        if not cats:
            return []
        sorted_cats = sorted(cats, key=lambda c: c.get("id", 0))
        return [c["name"] for c in sorted_cats]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return []


def _auto_detect_classes():
    """Try to auto-detect classes from the loaded dataset annotation."""
    train_path = st.session_state.get("train_img_path", "")
    if not train_path or not os.path.isdir(train_path):
        return []

    ann_file = os.path.join(train_path, "_annotations.coco.json")
    if not os.path.isfile(ann_file):
        json_files = [f for f in os.listdir(train_path) if f.endswith(".json")]
        if json_files:
            ann_file = os.path.join(train_path, json_files[0])
        else:
            return []

    return _extract_classes_from_coco_json(ann_file)


def _get_native_datasets():
    if not _HAS_FLASHDET:
        return []
    try:
        return list_datasets()
    except Exception:
        return []


def _check_network() -> bool:
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen("http://images.cocodataset.org", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _dataset_already_downloaded(dataset_id: str, output_dir: str | None = None) -> str | None:
    if output_dir is None:
        output_dir = os.path.join("data", dataset_id)
    if not os.path.isdir(output_dir):
        return None
    t = os.path.join(output_dir, "train")
    v = os.path.join(output_dir, "valid")
    if os.path.isdir(t) and os.path.isdir(v) and os.listdir(t) and os.listdir(v):
        return output_dir
    return None


EXTERNAL_DATASETS = [
    {"name": "COCO128", "imgs": 128, "cls": 80, "sz": "7MB", "fmt": "COCO", "url": "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"},
    {"name": "Objects365", "imgs": 5000, "cls": 365, "sz": "~2GB", "fmt": "COCO", "url": "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json"},
    {"name": "GlobalWheat", "imgs": 4700, "cls": 1, "sz": "~3GB", "fmt": "COCO", "url": "https://zenodo.org/record/4298502"},
    {"name": "VisDrone", "imgs": 6471, "cls": 10, "sz": "~2GB", "fmt": "Custom", "url": "https://github.com/VisDrone/VisDrone-Dataset"},
    {"name": "SKU-110K", "imgs": 11762, "cls": 1, "sz": "~3GB", "fmt": "CSV", "url": "https://github.com/eg4000/SKU110K_CVPR19"},
    {"name": "WIDER Face", "imgs": 32203, "cls": 1, "sz": "~1.5GB", "fmt": "Custom", "url": "http://shuoyang1213.me/WIDERFACE/"},
    {"name": "Cityscapes", "imgs": 5000, "cls": 30, "sz": "~11GB", "fmt": "Custom", "url": "https://www.cityscapes-dataset.com/"},
    {"name": "BDD100K", "imgs": 100000, "cls": 10, "sz": "~6GB", "fmt": "Custom", "url": "https://bdd-data.berkeley.edu/"},
    {"name": "KITTI", "imgs": 7481, "cls": 3, "sz": "~12GB", "fmt": "Custom", "url": "http://www.cvlibs.net/datasets/kitti/eval_object.php"},
    {"name": "xView", "imgs": 1400, "cls": 60, "sz": "~4GB", "fmt": "GeoJSON", "url": "http://xviewdataset.org/"},
    {"name": "OpenImages V7", "imgs": 1900000, "cls": 600, "sz": "~500GB", "fmt": "CSV", "url": "https://storage.googleapis.com/openimages/web/index.html"},
    {"name": "LVIS v1", "imgs": 164000, "cls": 1203, "sz": "~20GB", "fmt": "COCO", "url": "https://www.lvisdataset.org/"},
]

QUICK_START = [
    {"id": "sample", "name": "FlashDet Sample", "imgs": 50, "cls": 80, "sz": "~1GB"},
    {"id": "coco2017-val", "name": "COCO 2017 Val", "imgs": 5000, "cls": 80, "sz": "~1GB"},
    {"id": "voc2007", "name": "VOC 2007", "imgs": 9963, "cls": 20, "sz": "~900MB"},
]


def render_data_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Data")
    show_flashes()

    # Status banner
    dataset = st.session_state.get("dataset_name")
    if dataset:
        train_path = st.session_state.get("train_img_path", "")
        val_path = st.session_state.get("val_img_path", "")
        EXT = (".jpg", ".png", ".jpeg")
        tc = len([f for f in os.listdir(train_path) if f.lower().endswith(EXT)]) if train_path and os.path.isdir(train_path) else 0
        vc = len([f for f in os.listdir(val_path) if f.lower().endswith(EXT)]) if val_path and os.path.isdir(val_path) else 0
        st.success(f"**{dataset}** — {tc} train / {vc} val images")

    tab_upload, tab_download, tab_preview, tab_verify = st.tabs(["Upload", "Download", "Preview", "Verify"])

    with tab_upload:
        _render_upload()
    with tab_download:
        _render_download()
    with tab_preview:
        _render_preview()
    with tab_verify:
        _render_verify()


# ════════════════════════════════════════
# UPLOAD
# ════════════════════════════════════════

def _render_upload():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Train")
            train_upload = st.file_uploader("ZIP/TAR", type=["zip", "tar", "gz"], key="train_data_upload")
            manual_train = st.text_input(
                "Images dir", placeholder="/path/to/train/",
                value=st.session_state.get("train_img_path", ""),
                key="_train_img_input",
            )
            if manual_train:
                st.session_state["train_img_path"] = manual_train

    with col2:
        with st.container(border=True):
            st.markdown("#### Val")
            val_upload = st.file_uploader("ZIP/TAR", type=["zip", "tar", "gz"], key="val_data_upload")
            manual_val = st.text_input(
                "Images dir", placeholder="/path/to/valid/",
                value=st.session_state.get("val_img_path", ""),
                key="_val_img_input",
            )
            if manual_val:
                st.session_state["val_img_path"] = manual_val

    fc1, fc2, fc3 = st.columns([2, 1, 2])
    with fc1:
        st.selectbox("Format", ["COCO JSON", "Pascal VOC", "YOLO TXT"], key="ann_format")
    with fc2:
        pass
    with fc3:
        if train_upload:
            st.session_state["dataset_name"] = train_upload.name
            if not st.session_state.get("train_img_path"):
                if st.button("Extract", key="extract_train", use_container_width=True, type="primary"):
                    _extract_upload(train_upload, "train")
        if val_upload and not st.session_state.get("val_img_path"):
            if st.button("Extract Val", key="extract_val", use_container_width=True):
                _extract_upload(val_upload, "val")

    # ── Classes section ──
    _render_class_config()

    _render_conversion_hint()


def _render_class_config():
    """Render class configuration: auto-detect from annotation, load from .txt, or manual edit."""
    from flashstudio.utils import flash

    with st.container(border=True):
        st.markdown("#### Classes")

        # Auto-detect button + .txt upload in one row
        cc1, cc2, cc3 = st.columns([2, 2, 1])
        with cc1:
            if st.button("Auto-detect from annotation", key="btn_auto_classes",
                         use_container_width=True, type="primary"):
                detected = _auto_detect_classes()
                if detected:
                    st.session_state["class_names"] = "\n".join(detected)
                    st.session_state["num_classes"] = len(detected)
                    flash(f"Detected {len(detected)} classes from annotation", "success")
                    st.rerun()
                else:
                    flash("No classes found — load a dataset with COCO annotations first", "warning")
                    st.rerun()

        with cc2:
            cls_file = st.file_uploader("Load classes.txt", type=["txt"],
                                        key="class_txt_upload", label_visibility="collapsed")
            if cls_file:
                content = cls_file.read().decode("utf-8", errors="ignore")
                names = [line.strip() for line in content.strip().split("\n") if line.strip()]
                if names:
                    st.session_state["class_names"] = "\n".join(names)
                    st.session_state["num_classes"] = len(names)
                    flash(f"Loaded {len(names)} classes from file", "success")
                    st.rerun()

        with cc3:
            nc = st.session_state.get("num_classes", 0)
            st.metric("Count", nc if nc else "—")

        # Auto-detect on first load if classes not set
        current_classes = st.session_state.get("class_names", "")
        if not current_classes.strip() and st.session_state.get("train_img_path"):
            detected = _auto_detect_classes()
            if detected:
                st.session_state["class_names"] = "\n".join(detected)
                st.session_state["num_classes"] = len(detected)

        # Editable text area
        class_text = st.text_area(
            "Class names (one per line)",
            value=st.session_state.get("class_names", ""),
            height=100,
            key="_class_names_input",
            placeholder="person\ncar\nbicycle\n...",
        )
        if class_text != st.session_state.get("class_names", ""):
            names = [n.strip() for n in class_text.strip().split("\n") if n.strip()]
            st.session_state["class_names"] = "\n".join(names)
            st.session_state["num_classes"] = len(names)

        # Download button for current classes
        if current_classes.strip():
            st.download_button(
                "Download classes.txt",
                current_classes.strip() + "\n",
                file_name="classes.txt",
                mime="text/plain",
                use_container_width=True,
            )


def _render_conversion_hint():
    train_path = st.session_state.get("train_img_path", "")
    if not train_path or not os.path.isdir(train_path):
        return
    ann_format = st.session_state.get("ann_format", "COCO JSON")
    detected = st.session_state.get("detected_format", "")
    needs = "YOLO" in ann_format or "VOC" in ann_format or detected in ("txt", "voc")
    if not needs:
        coco_ann = os.path.join(train_path, "_annotations.coco.json")
        if not os.path.isfile(coco_ann) and _HAS_FLASHDET:
            parent = os.path.dirname(train_path)
            det_fmt = detect_dataset_format(parent)
            if det_fmt in ("txt", "voc"):
                needs = True
                st.session_state["detected_format"] = det_fmt
    if needs:
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("Convert to COCO", type="primary", key="manual_convert", use_container_width=True):
                source = os.path.dirname(train_path) if os.path.basename(train_path) in ("images", "train") else train_path
                tfmt = "txt" if "YOLO" in ann_format or detected == "txt" else "voc"
                _convert_to_coco(source, "train", tfmt)
        with c2:
            st.caption(f"Format detected: **{detected or ann_format}** → auto-convert to COCO JSON")


# ════════════════════════════════════════
# DOWNLOAD
# ════════════════════════════════════════

def _render_download():
    # Quick start — compact single row each
    st.markdown("#### Quick Start")
    for idx, ds in enumerate(QUICK_START):
        did = ds["id"]
        existing = _dataset_already_downloaded(did)
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f'<div style="font-size:0.84rem;"><b>{ds["name"]}</b> · '
                f'{ds["imgs"]} imgs · {ds["cls"]} cls · {ds["sz"]}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            label = "Use" if existing else "Get"
            if st.button(label, key=f"qs_{idx}", use_container_width=True, type="primary"):
                if existing:
                    _use_existing(existing, ds)
                else:
                    st.session_state.update({"dataset_name": ds["name"], "dataset_classes": ds["cls"], "dataset_id": did})
                    _run_flashdet_download(did)

    st.divider()

    # Native datasets
    native = _get_native_datasets()
    if native:
        with st.expander(f"FlashDet Native ({len(native)} datasets)", expanded=False):
            for i, ds in enumerate(native):
                did = ds.get("id", ds.get("name", "").lower().replace(" ", ""))
                existing = _dataset_already_downloaded(did)
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.caption(f"**{ds.get('name', did)}** · {ds.get('classes', '?')} cls")
                with c2:
                    if st.button("Use" if existing else "Get", key=f"nat_{i}", use_container_width=True):
                        if existing:
                            _use_existing(existing, ds)
                        else:
                            st.session_state.update({"dataset_name": ds.get("name", did), "dataset_id": did, "dataset_classes": ds.get("classes", 80)})
                            _run_flashdet_download(did)

    # External
    with st.expander(f"External Datasets ({len(EXTERNAL_DATASETS)})", expanded=False):
        for i, ds in enumerate(EXTERNAL_DATASETS):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f'<div style="font-size:0.84rem;"><b>{ds["name"]}</b> · '
                    f'{ds["imgs"]} · {ds["cls"]} cls · {ds["sz"]} · {ds["fmt"]}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                url = ds.get("url", "")
                is_manual = not url or "VisDrone" in url
                if st.button("Link" if is_manual else "Get", key=f"ext_{i}", use_container_width=True):
                    if is_manual:
                        st.info(f"Manual: [{url}]({url})")
                    else:
                        st.session_state.update({"dataset_name": ds["name"], "dataset_classes": ds["cls"]})
                        _run_external_download(ds)


# ════════════════════════════════════════
# PREVIEW
# ════════════════════════════════════════

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

    EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    all_imgs = sorted([f for f in os.listdir(preview_path) if f.lower().endswith(EXT)])
    if not all_imgs:
        st.warning("No images.")
        return

    with c2:
        n = st.slider("Count", 1, min(12, len(all_imgs)), 4, key="prev_n")
    with c3:
        bbox = st.checkbox("Boxes", value=True, key="prev_bb")

    if "prev_start" not in st.session_state:
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
    COLORS = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,0,0),(0,128,0)]
    selected = all_imgs[start:start + n]

    total_boxes = 0
    for row in range(0, len(selected), 4):
        cols = st.columns(4)
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
                                    color = COLORS[a.get("category_id", 0) % len(COLORS)]
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
                                            color = COLORS[cls_id % len(COLORS)]
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


# ════════════════════════════════════════
# VERIFY
# ════════════════════════════════════════

def _render_verify():
    train_path = st.session_state.get("train_img_path", "")
    val_path = st.session_state.get("val_img_path", "")
    if not train_path and not val_path:
        st.info("No dataset loaded. Upload or download a dataset first.")
        return

    dataset_root = st.session_state.get("dataset_output_path", "")

    # FlashDet verify
    if dataset_root and os.path.isdir(dataset_root) and _HAS_FLASHDET:
        try:
            ok = verify_dataset(dataset_root)
            st.success("Verification **PASSED**") if ok else st.warning("Verification: issues detected")
        except Exception as e:
            st.caption(str(e)[:60])

        try:
            s = summarize_coco_root(dataset_root)
            if s:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Train", s.get("train_images", 0))
                with m2:
                    st.metric("Val", s.get("val_images", 0))
                with m3:
                    st.metric("Annotations", s.get("total_annotations", 0))
                with m4:
                    st.metric("Categories", s.get("num_categories", 0))
                if s.get("class_distribution"):
                    with st.expander("Class Distribution"):
                        top = sorted(s["class_distribution"].items(), key=lambda x: x[1], reverse=True)[:15]
                        st.bar_chart({n: c for n, c in top})
        except Exception:
            pass
        return

    if not _HAS_FLASHDET and dataset_root and os.path.isdir(dataset_root):
        st.info(_FLASHDET_INSTALL_MSG)

    # Manual count — always show this as fallback
    EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
    ti = [f for f in os.listdir(train_path) if f.lower().endswith(EXT)] if train_path and os.path.isdir(train_path) else []
    vi = [f for f in os.listdir(val_path) if f.lower().endswith(EXT)] if val_path and os.path.isdir(val_path) else []

    # Count YOLO labels if present
    train_label_dir = st.session_state.get("train_label_path", "")
    if not train_label_dir and train_path:
        candidate = train_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        if os.path.isdir(candidate):
            train_label_dir = candidate
    train_labels = len([f for f in os.listdir(train_label_dir) if f.endswith(".txt")]) if train_label_dir and os.path.isdir(train_label_dir) else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Train Images", len(ti))
    with m2:
        st.metric("Val Images", len(vi))
    with m3:
        st.metric("Labels", train_labels if train_labels else "—")
    with m4:
        status = "OK" if ti else "Empty"
        if ti and train_labels and train_labels == len(ti):
            status = "Complete"
        elif ti and train_labels and train_labels != len(ti):
            status = f"Partial ({train_labels}/{len(ti)})"
        st.metric("Status", status)

    if train_path and os.path.isdir(train_path):
        st.caption(f"Train: `{train_path}`")
    if val_path and os.path.isdir(val_path):
        st.caption(f"Val: `{val_path}`")

    # Sample preview
    if ti:
        with st.expander("Samples"):
            cols = st.columns(min(4, len(ti)))
            for i, img_name in enumerate(ti[:4]):
                with cols[i]:
                    try:
                        st.image(os.path.join(train_path, img_name), caption=img_name[:15], use_container_width=True)
                    except Exception:
                        pass


# ════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════

def _extract_upload(uploaded_file, split):
    import zipfile, tarfile, tempfile
    name = uploaded_file.name
    out = os.path.join("data", "uploaded", split)
    os.makedirs(out, exist_ok=True)
    with st.spinner(f"Extracting {name}..."):
        try:
            if name.endswith(".zip"):
                tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
                tmp.write(uploaded_file.read()); tmp.flush(); tmp.close()
                uploaded_file.seek(0)
                with zipfile.ZipFile(tmp.name, "r") as zf:
                    zf.extractall(out)
                os.unlink(tmp.name)
            elif name.endswith((".tar", ".gz", ".tar.gz", ".tgz")):
                tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
                tmp.write(uploaded_file.read()); tmp.flush(); tmp.close()
                uploaded_file.seek(0)
                with tarfile.open(tmp.name, "r:*") as tf:
                    tf.extractall(out)
                os.unlink(tmp.name)
            else:
                st.error(f"Unsupported: {name}")
                return

            root = _find_root(out)
            st.session_state["dataset_output_path"] = root

            fmt = "unknown"
            if _HAS_FLASHDET:
                fmt = detect_dataset_format(root)
            st.session_state["detected_format"] = fmt

            if fmt in ("txt", "voc"):
                _convert_to_coco(root, split, fmt)
                return

            img_dir = _find_images(root)
            if img_dir:
                st.session_state[f"{split}_img_path"] = img_dir
                ann = _find_ann(root)
                if ann:
                    st.session_state[f"{split}_ann_path"] = ann
                label_dir = _find_labels(root)
                if label_dir:
                    st.session_state[f"{split}_label_path"] = label_dir
                st.session_state["dataset_name"] = name.rsplit(".", 1)[0]

                # Auto-detect sibling split (e.g., valid/ alongside train/)
                parent = os.path.dirname(img_dir)
                for sibling, key in [("valid", "val"), ("val", "val"), ("test", "test"), ("train", "train")]:
                    sib_path = os.path.join(parent, sibling)
                    if os.path.isdir(sib_path) and key != split:
                        sk = "val_img_path" if key in ("val",) else f"{key}_img_path"
                        if not st.session_state.get(sk):
                            st.session_state[sk] = sib_path

                # Auto-detect classes from annotation
                detected_cls = _auto_detect_classes()
                if detected_cls and not st.session_state.get("class_names", "").strip():
                    st.session_state["class_names"] = "\n".join(detected_cls)
                    st.session_state["num_classes"] = len(detected_cls)

                from flashstudio.utils import flash
                flash(f"Extracted `{name}` to {img_dir}", "success")
                st.rerun()
            else:
                st.session_state[f"{split}_img_path"] = root
                st.session_state["dataset_name"] = name.rsplit(".", 1)[0]
                from flashstudio.utils import flash
                flash(f"Extracted `{name}` (no images dir found, using root)", "warning")
                st.rerun()
        except Exception as e:
            st.error(f"Extract failed: {e}")


def _find_root(base):
    entries = os.listdir(base)
    if len(entries) == 1:
        s = os.path.join(base, entries[0])
        if os.path.isdir(s):
            return s
    for e in entries:
        full = os.path.join(base, e)
        if os.path.isdir(full) and _HAS_FLASHDET and detect_dataset_format(full) != "unknown":
            return full
    return base


def _convert_to_coco(src, split, src_fmt):
    if not _HAS_FLASHDET:
        st.error(_FLASHDET_INSTALL_MSG)
        return
    out = os.path.join("data", "uploaded", f"{split}_coco")
    cls_names = None
    raw = st.session_state.get("class_names", "")
    if raw.strip():
        cls_names = [c.strip() for c in raw.strip().replace(",", "\n").split("\n") if c.strip()]
    with st.spinner("Converting to COCO..."):
        try:
            stats = convert_dataset(source_dir=src, output_dir=out, target_format="coco", class_names=cls_names)
            if stats.get("status") == "already_in_target_format":
                out = src
            st.session_state["dataset_output_path"] = out
            st.session_state["ann_format"] = "COCO JSON"
            t = os.path.join(out, "train")
            v = os.path.join(out, "valid")
            if not os.path.isdir(v):
                v = os.path.join(out, "val")
            if os.path.isdir(t):
                st.session_state[f"{split}_img_path"] = t
            if os.path.isdir(v):
                st.session_state["val_img_path"] = v
            from flashstudio.utils import flash
            flash("Converted to COCO format", "success")
            st.rerun()
        except Exception as e:
            st.error(f"Conversion failed: {e}")


def _find_images(base):
    EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    best = None
    best_count = 0
    for root, dirs, files in os.walk(base):
        img_count = len([f for f in files if f.lower().endswith(EXT)])
        if img_count > best_count:
            best = root
            best_count = img_count
        for d in dirs:
            if d.lower() in ("images", "imgs"):
                c = os.path.join(root, d)
                for sub_root, _, sub_files in os.walk(c):
                    sc = len([f for f in sub_files if f.lower().endswith(EXT)])
                    if sc > best_count:
                        best = sub_root
                        best_count = sc
    return best if best_count >= 1 else None


def _find_ann(base):
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".json") and ("annotation" in f.lower() or "coco" in f.lower()):
                return os.path.join(root, f)
    return None


def _find_labels(base):
    """Find YOLO-format label directory (folder named 'labels' with .txt files)."""
    for root, dirs, files in os.walk(base):
        for d in dirs:
            if d.lower() == "labels":
                lbl_dir = os.path.join(root, d)
                for sub_root, _, sub_files in os.walk(lbl_dir):
                    if any(f.endswith(".txt") for f in sub_files):
                        return sub_root
    return None


def _use_existing(path, ds_info):
    from flashstudio.utils import flash
    st.session_state["dataset_output_path"] = path
    name = ds_info.get("name", os.path.basename(path))
    st.session_state["dataset_name"] = name
    st.session_state["dataset_classes"] = ds_info.get("classes", ds_info.get("cls", 80))
    if ds_info.get("id"):
        st.session_state["dataset_id"] = ds_info["id"]
    t, v = os.path.join(path, "train"), os.path.join(path, "valid")
    if os.path.isdir(t):
        st.session_state["train_img_path"] = t
    if os.path.isdir(v):
        st.session_state["val_img_path"] = v

    detected_cls = _auto_detect_classes()
    if detected_cls:
        st.session_state["class_names"] = "\n".join(detected_cls)
        st.session_state["num_classes"] = len(detected_cls)

    flash(f"Dataset `{name}` loaded", "success")
    st.rerun()


def _run_flashdet_download(dataset_id):
    if not _HAS_FLASHDET:
        st.error(_FLASHDET_INSTALL_MSG)
        return
    out = st.session_state.get("download_output_dir") or os.path.join("data", dataset_id)
    existing = _dataset_already_downloaded(dataset_id, out)
    if existing:
        st.session_state["dataset_output_path"] = existing
        for d, k in [("train", "train_img_path"), ("valid", "val_img_path")]:
            p = os.path.join(existing, d)
            if os.path.isdir(p):
                st.session_state[k] = p
        detected_cls = _auto_detect_classes()
        if detected_cls:
            st.session_state["class_names"] = "\n".join(detected_cls)
            st.session_state["num_classes"] = len(detected_cls)
        st.rerun()
        return
    if not _check_network():
        st.error("Network unavailable.")
        return
    with st.spinner(f"Downloading {dataset_id}..."):
        try:
            result = download_dataset(dataset_id=dataset_id, output_dir=out)
            st.session_state["dataset_output_path"] = result
            for d, k in [("train", "train_img_path"), ("valid", "val_img_path")]:
                p = os.path.join(result, d)
                if os.path.isdir(p):
                    st.session_state[k] = p
            from flashstudio.utils import flash
            flash(f"Downloaded `{dataset_id}` successfully", "success")
            st.rerun()
        except Exception as e:
            st.error(f"Download failed: {e}")


def _run_external_download(ds):
    import urllib.request, urllib.error, zipfile, tempfile
    url, name = ds.get("url", ""), ds.get("name", "dataset")
    if not url or "VisDrone" in url:
        st.info(f"Manual: [{url}]({url})")
        return
    out = os.path.join("data", name.lower().replace(" ", "_").replace("(", "").replace(")", ""))
    if os.path.isdir(out) and os.listdir(out):
        st.session_state.update({"dataset_output_path": out, "dataset_name": name})
        st.rerun()
        return
    with st.spinner(f"Downloading {name}..."):
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            tmp_path = tmp.name; tmp.close()
            urllib.request.urlretrieve(url, tmp_path)
            if os.path.getsize(tmp_path) < 1000 or not zipfile.is_zipfile(tmp_path):
                st.error(f"Invalid download. Manual: [{url}]({url})")
                return
            os.makedirs(out, exist_ok=True)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(out)
            st.session_state.update({"dataset_output_path": out, "dataset_name": name})
            for d, k in [("train", "train_img_path"), ("valid", "val_img_path")]:
                p = os.path.join(out, d)
                if os.path.isdir(p):
                    st.session_state[k] = p
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
