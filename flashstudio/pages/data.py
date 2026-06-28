"""FlashStudio — Data Upload / Download Page (powered by flashdet)."""

import os
import json
import streamlit as st
from flashdet import download_dataset, list_datasets
from flashdet.data import convert_dataset, detect_dataset_format, verify_dataset, summarize_coco_root


def _get_native_datasets():
    """Fetch available datasets from flashdet registry."""
    try:
        datasets = list_datasets()
        return datasets
    except Exception:
        return []


def _check_network() -> bool:
    """Quick connectivity check (COCO dataset server)."""
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen("http://images.cocodataset.org", timeout=5)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _dataset_already_downloaded(dataset_id: str, output_dir: str | None = None) -> str | None:
    """Return the dataset path if it's already downloaded and looks valid, else None."""
    if output_dir is None:
        output_dir = os.path.join("data", dataset_id)
    if not os.path.isdir(output_dir):
        return None
    train_dir = os.path.join(output_dir, "train")
    valid_dir = os.path.join(output_dir, "valid")
    if os.path.isdir(train_dir) and os.path.isdir(valid_dir):
        train_files = os.listdir(train_dir)
        valid_files = os.listdir(valid_dir)
        if train_files and valid_files:
            return output_dir
    return None


EXTERNAL_DATASETS = [
    {
        "name": "COCO128 (tiny subset)",
        "description": "128 COCO images — download from Ultralytics GitHub manually",
        "images": 128,
        "classes": 80,
        "size": "7 MB",
        "format": "COCO JSON",
        "url": "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip",
        "native": False,
    },
    {
        "name": "Objects365 (tiny subset)",
        "description": "365-class large-scale detection — tiny 5K subset for testing",
        "images": 5000,
        "classes": 365,
        "size": "~2 GB",
        "format": "COCO JSON",
        "url": "https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json",
        "native": False,
    },
    {
        "name": "GlobalWheat 2020 (agriculture)",
        "description": "Wheat head detection — 4.7K images from fields worldwide",
        "images": 4700,
        "classes": 1,
        "size": "~3 GB",
        "format": "COCO JSON",
        "url": "https://zenodo.org/record/4298502",
        "native": False,
    },
    {
        "name": "VisDrone (drone aerial)",
        "description": "Drone aerial images — pedestrians, vehicles from above",
        "images": 6471,
        "classes": 10,
        "size": "~2 GB",
        "format": "Custom → COCO JSON",
        "url": "https://github.com/VisDrone/VisDrone-Dataset",
        "native": False,
    },
    {
        "name": "SKU-110K (retail / dense)",
        "description": "Dense shelf product detection — 110K categories, retail environment",
        "images": 11762,
        "classes": 1,
        "size": "~3 GB",
        "format": "CSV → COCO JSON",
        "url": "https://github.com/eg4000/SKU110K_CVPR19",
        "native": False,
    },
    {
        "name": "WIDER Face (face detection)",
        "description": "Face detection benchmark — 32K images, varying scales",
        "images": 32203,
        "classes": 1,
        "size": "~1.5 GB",
        "format": "Custom → COCO JSON",
        "url": "http://shuoyang1213.me/WIDERFACE/",
        "native": False,
    },
    {
        "name": "Cityscapes (urban driving)",
        "description": "Urban street scenes — 5K fine-annotated images, 30 classes",
        "images": 5000,
        "classes": 30,
        "size": "~11 GB",
        "format": "Custom → COCO JSON",
        "url": "https://www.cityscapes-dataset.com/",
        "native": False,
    },
    {
        "name": "BDD100K (driving)",
        "description": "Berkeley driving dataset — 100K images, 10 detection classes",
        "images": 100000,
        "classes": 10,
        "size": "~6 GB",
        "format": "Custom → COCO JSON",
        "url": "https://bdd-data.berkeley.edu/",
        "native": False,
    },
    {
        "name": "KITTI (autonomous driving)",
        "description": "Autonomous driving — 7.5K training images, cars/pedestrians/cyclists",
        "images": 7481,
        "classes": 3,
        "size": "~12 GB",
        "format": "Custom → COCO JSON",
        "url": "http://www.cvlibs.net/datasets/kitti/eval_object.php",
        "native": False,
    },
    {
        "name": "xView (satellite/overhead)",
        "description": "Overhead satellite imagery — 1M objects across 60 classes",
        "images": 1400,
        "classes": 60,
        "size": "~4 GB",
        "format": "GeoJSON → COCO JSON",
        "url": "http://xviewdataset.org/",
        "native": False,
    },
    {
        "name": "OpenImages V7 (large-scale)",
        "description": "Google Open Images — 1.9M images, 600 classes with detection boxes",
        "images": 1900000,
        "classes": 600,
        "size": "~500 GB (full)",
        "format": "CSV → COCO JSON",
        "url": "https://storage.googleapis.com/openimages/web/index.html",
        "native": False,
    },
    {
        "name": "LVIS v1 (long-tail)",
        "description": "Large Vocabulary Instance Segmentation — 1200+ categories, COCO images",
        "images": 164000,
        "classes": 1203,
        "size": "~20 GB",
        "format": "COCO JSON",
        "url": "https://www.lvisdataset.org/",
        "native": False,
    },
]

QUICK_START_DATASETS = [
    {
        "id": "sample",
        "name": "FlashDet Sample (50 images)",
        "description": "Built-in 50-image COCO subset — fastest way to test the full pipeline (downloads ~1GB cache on first use)",
        "images": 50,
        "classes": 80,
        "size": "~1 GB (first download, cached)",
        "format": "COCO JSON",
    },
    {
        "id": "coco2017-val",
        "name": "COCO 2017 Val (5K images)",
        "description": "COCO 2017 validation set only — 5K images, 80 classes. Good for evaluation and quick training tests.",
        "images": 5000,
        "classes": 80,
        "size": "~1 GB",
        "format": "COCO JSON",
    },
    {
        "id": "voc2007",
        "name": "Pascal VOC 2007 (10K images)",
        "description": "Classic 20-class detection benchmark — 5K train/val + 5K test. Auto-converted to COCO JSON format.",
        "images": 9963,
        "classes": 20,
        "size": "~900 MB",
        "format": "VOC → COCO JSON",
    },
]


def render_data_page():
    """Render data upload and open-source dataset download page."""
    from flashstudio.components.styles import render_page_header
    render_page_header("📦", "Data Setup",
                       "Upload your own dataset or download a ready-to-use open-source dataset.")

    data_mode = st.radio(
        "Choose data source",
        ["📤 Upload Your Dataset", "📥 Download Open-Source Dataset"],
        horizontal=True,
        key="data_mode",
    )

    if data_mode == "📤 Upload Your Dataset":
        _render_upload_tab()
    else:
        _render_download_tab()

    st.divider()
    _render_dataset_status()


def _render_upload_tab():
    """Upload custom dataset."""
    st.markdown("### Upload Your Dataset")
    st.info("Supported formats: **COCO JSON**, **Pascal VOC XML**, **YOLO TXT**. Upload as ZIP/TAR.")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("**Training Data**")
            train_upload = st.file_uploader(
                "Upload training dataset (zip/tar)",
                type=["zip", "tar", "gz"],
                key="train_data_upload",
            )
            st.text_input(
                "Or enter path to annotations",
                placeholder="/content/datasets/train/annotations.json",
                key="train_ann_path",
            )
            st.text_input(
                "Images directory",
                placeholder="/content/datasets/train/images/",
                key="train_img_path",
            )

    with col2:
        with st.container(border=True):
            st.markdown("**Validation Data**")
            val_upload = st.file_uploader(
                "Upload validation dataset (zip/tar)",
                type=["zip", "tar", "gz"],
                key="val_data_upload",
            )
            st.text_input(
                "Or enter path to annotations",
                placeholder="/content/datasets/val/annotations.json",
                key="val_ann_path",
            )
            st.text_input(
                "Images directory",
                placeholder="/content/datasets/val/images/",
                key="val_img_path",
            )

    with st.container(border=True):
        st.markdown("**Dataset Format**")
        col_fmt, col_cls = st.columns(2)
        with col_fmt:
            st.selectbox(
                "Annotation Format",
                ["COCO JSON (recommended)", "Pascal VOC XML", "YOLO TXT"],
                key="ann_format",
            )
        with col_cls:
            st.number_input("Number of Classes", min_value=1, max_value=1000, value=80, key="upload_num_classes")
        st.text_area(
            "Class Names (one per line, optional)",
            placeholder="person\ncar\nbicycle\n...",
            height=100,
            key="class_names",
        )

    if train_upload:
        st.session_state["dataset_name"] = train_upload.name
        st.success(f"✅ Training data uploaded: {train_upload.name} ({train_upload.size / 1e6:.1f} MB)")
        if not st.session_state.get("train_img_path"):
            if st.button("📦 Extract & Set Paths", key="extract_train"):
                _extract_upload(train_upload, "train")

    if val_upload:
        st.success(f"✅ Validation data uploaded: {val_upload.name} ({val_upload.size / 1e6:.1f} MB)")
        if not st.session_state.get("val_img_path"):
            if st.button("📦 Extract & Set Paths", key="extract_val"):
                _extract_upload(val_upload, "val")

    # Manual format conversion for already-extracted datasets
    _render_manual_conversion()


def _extract_upload(uploaded_file, split: str):
    """Extract uploaded ZIP/TAR archive, auto-detect format, convert to COCO if needed."""
    import zipfile
    import tarfile
    import tempfile

    name = uploaded_file.name
    output_base = os.path.join("data", "uploaded", split)
    os.makedirs(output_base, exist_ok=True)

    with st.spinner(f"Extracting `{name}`..."):
        try:
            if name.endswith(".zip"):
                tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp.close()
                uploaded_file.seek(0)
                with zipfile.ZipFile(tmp.name, "r") as zf:
                    zf.extractall(output_base)
                os.unlink(tmp.name)
            elif name.endswith((".tar", ".gz", ".tar.gz", ".tgz")):
                tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
                tmp.write(uploaded_file.read())
                tmp.flush()
                tmp.close()
                uploaded_file.seek(0)
                with tarfile.open(tmp.name, "r:*") as tf:
                    tf.extractall(output_base)
                os.unlink(tmp.name)
            else:
                st.error(f"Unsupported archive format: {name}")
                return

            # Auto-detect dataset format and convert to COCO if needed
            detected_format = detect_dataset_format(output_base)
            st.session_state["detected_format"] = detected_format

            if detected_format == "unknown":
                sub = _find_dataset_root(output_base)
                if sub and sub != output_base:
                    detected_format = detect_dataset_format(sub)
                    if detected_format != "unknown":
                        output_base = sub

            if detected_format in ("txt", "voc"):
                _convert_to_coco(output_base, split, detected_format)
                return

            # Already COCO or unknown — find images directory
            img_dir = _find_images_dir(output_base)
            if img_dir:
                st.session_state[f"{split}_img_path"] = img_dir
                ann_file = _find_annotation_file(output_base)
                if ann_file:
                    st.session_state[f"{split}_ann_path"] = ann_file
                st.success(f"✅ Extracted to: `{img_dir}` (format: {detected_format})")
                st.rerun()
            else:
                st.session_state[f"{split}_img_path"] = output_base
                st.warning(f"Extracted to `{output_base}` — could not auto-detect images subfolder.")
                st.rerun()

        except Exception as e:
            st.error(f"Extraction failed: {e}")


def _find_dataset_root(base_path: str) -> str | None:
    """Find actual dataset root inside extracted archive (may be nested one level)."""
    entries = os.listdir(base_path)
    if len(entries) == 1:
        single = os.path.join(base_path, entries[0])
        if os.path.isdir(single):
            return single
    for entry in entries:
        full = os.path.join(base_path, entry)
        if os.path.isdir(full):
            fmt = detect_dataset_format(full)
            if fmt != "unknown":
                return full
    return base_path


def _convert_to_coco(source_dir: str, split: str, source_format: str):
    """Convert YOLO TXT or Pascal VOC to COCO JSON format for FlashDet training."""
    output_dir = os.path.join("data", "uploaded", f"{split}_coco")

    class_names = None
    raw_names = st.session_state.get("class_names", "")
    if raw_names.strip():
        class_names = [c.strip() for c in raw_names.strip().split("\n") if c.strip()]

    format_label = "YOLO TXT" if source_format == "txt" else "Pascal VOC XML"
    with st.spinner(f"Converting {format_label} → COCO JSON..."):
        try:
            stats = convert_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                target_format="coco",
                class_names=class_names,
            )

            if stats.get("status") == "already_in_target_format":
                st.info("Dataset is already in COCO format.")
                output_dir = source_dir
            else:
                st.success(
                    f"✅ Converted {format_label} → COCO JSON\n\n"
                    f"Images: {stats.get('num_images', '?')} · "
                    f"Annotations: {stats.get('num_annotations', '?')} · "
                    f"Classes: {stats.get('num_classes', '?')}"
                )

            train_dir = os.path.join(output_dir, "train")
            valid_dir = os.path.join(output_dir, "valid")
            if not os.path.isdir(valid_dir):
                valid_dir = os.path.join(output_dir, "val")

            if os.path.isdir(train_dir):
                st.session_state[f"{split}_img_path"] = train_dir
            elif os.path.isdir(output_dir):
                st.session_state[f"{split}_img_path"] = output_dir

            if os.path.isdir(valid_dir):
                st.session_state["val_img_path"] = valid_dir

            st.session_state["dataset_output_path"] = output_dir
            st.session_state["ann_format"] = "COCO JSON (recommended)"
            st.rerun()

        except Exception as e:
            st.error(f"Format conversion failed: {e}\n\nSource: `{source_dir}` → Target: COCO")


def _find_images_dir(base_path: str) -> str | None:
    """Find the most likely images directory in an extracted archive."""
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for root, dirs, files in os.walk(base_path):
        image_files = [f for f in files if f.lower().endswith(IMAGE_EXTS)]
        if len(image_files) >= 2:
            return root
        if any(d.lower() in ("images", "imgs", "train", "val") for d in dirs):
            for d in dirs:
                if d.lower() in ("images", "imgs"):
                    candidate = os.path.join(root, d)
                    imgs = [f for f in os.listdir(candidate) if f.lower().endswith(IMAGE_EXTS)]
                    if imgs:
                        return candidate
    return None


def _find_annotation_file(base_path: str) -> str | None:
    """Find COCO annotation JSON file in extracted archive."""
    for root, _dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith(".json") and ("annotation" in f.lower() or "coco" in f.lower()):
                return os.path.join(root, f)
    return None


def _render_manual_conversion():
    """Allow user to manually trigger format conversion on an existing path."""
    train_path = st.session_state.get("train_img_path", "")
    if not train_path or not os.path.isdir(train_path):
        return

    ann_format = st.session_state.get("ann_format", "COCO JSON (recommended)")
    detected = st.session_state.get("detected_format", "")

    needs_conversion = (
        "YOLO" in ann_format or "VOC" in ann_format or
        detected in ("txt", "voc")
    )

    if not needs_conversion:
        # Check if COCO annotation file exists
        coco_ann = os.path.join(train_path, "_annotations.coco.json")
        if not os.path.isfile(coco_ann):
            parent = os.path.dirname(train_path)
            det_fmt = detect_dataset_format(parent)
            if det_fmt in ("txt", "voc"):
                needs_conversion = True
                st.session_state["detected_format"] = det_fmt

    if needs_conversion:
        st.divider()
        with st.container(border=True):
            st.markdown("### 🔄 Format Conversion Required")
            src_fmt = detected or ("YOLO TXT" if "YOLO" in ann_format else "Pascal VOC")
            st.warning(
                f"Your dataset appears to be in **{src_fmt}** format. "
                f"FlashDet requires **COCO JSON** format for training.\n\n"
                f"Click below to auto-convert."
            )

            col_btn, col_info = st.columns([1, 2])
            with col_btn:
                if st.button("🔄 Convert to COCO", type="primary", key="manual_convert",
                             use_container_width=True):
                    source = os.path.dirname(train_path) if os.path.basename(train_path) in ("images", "train") else train_path
                    target_fmt = "txt" if "YOLO" in ann_format or detected == "txt" else "voc"
                    _convert_to_coco(source, "train", target_fmt)
            with col_info:
                st.caption(
                    "This will:\n"
                    "1. Detect annotation format automatically\n"
                    "2. Convert to COCO JSON format\n"
                    "3. Create train/valid splits\n"
                    "4. Set paths for training"
                )


def _render_download_tab():
    """Download datasets — quick-start tiny data first, then FlashDet native, then external."""
    # Quick-start tiny datasets — always visible at top
    st.markdown("### ⚡ Quick Start — Tiny Datasets")
    st.caption("Download a small dataset to test the full pipeline.")

    for idx, ds in enumerate(QUICK_START_DATASETS):
        dataset_id = ds.get("id", "")
        existing = _dataset_already_downloaded(dataset_id)

        with st.container(border=True):
            col_info, col_stats, col_action = st.columns([3, 2, 1])
            with col_info:
                st.markdown(f"**{ds['name']}**")
                st.caption(ds["description"])
            with col_stats:
                st.markdown(
                    f"📸 {ds['images']} images · 🏷️ {ds['classes']} classes · "
                    f"💾 {ds['size']} · 📋 {ds['format']}"
                )
            with col_action:
                if existing:
                    if st.button("✅ Use Cached", key=f"quick_start_dl_{idx}", use_container_width=True, type="primary"):
                        _use_existing_dataset(existing, ds)
                else:
                    if st.button("⚡ Download", key=f"quick_start_dl_{idx}", use_container_width=True, type="primary"):
                        st.session_state["dataset_name"] = ds["name"]
                        st.session_state["dataset_classes"] = ds["classes"]
                        st.session_state["dataset_id"] = dataset_id
                        _run_flashdet_download(dataset_id)

    st.divider()

    # FlashDet native datasets
    st.markdown("### FlashDet Native Datasets")
    st.caption("Powered by `flashdet.download_dataset()` — auto-handled format conversion and split.")

    native_datasets = _get_native_datasets()

    if native_datasets:
        for i, ds in enumerate(native_datasets):
            dataset_id = ds.get("id", ds.get("name", "").lower().replace(" ", ""))
            existing = _dataset_already_downloaded(dataset_id)

            with st.container(border=True):
                col_info, col_stats, col_action = st.columns([3, 2, 1])

                with col_info:
                    st.markdown(f"**{ds.get('name', ds.get('id', 'Unknown'))}**")
                    st.caption(ds.get("description", ""))

                with col_stats:
                    classes = ds.get("classes", "?")
                    fmt = ds.get("format", "COCO JSON")
                    st.markdown(f"🏷️ {classes} classes · 📋 {fmt}")

                with col_action:
                    if existing:
                        if st.button("✅ Use Cached", key=f"native_dl_{i}", use_container_width=True):
                            _use_existing_dataset(existing, ds)
                    else:
                        if st.button("⬇️ Download", key=f"native_dl_{i}", use_container_width=True):
                            st.session_state["dataset_name"] = ds.get("name", dataset_id)
                            st.session_state["dataset_id"] = dataset_id
                            st.session_state["dataset_classes"] = ds.get("classes", 80)
                            _run_flashdet_download(dataset_id)
    else:
        st.warning("Could not fetch dataset list from flashdet. Showing defaults.")
        _render_fallback_native_datasets()

    st.divider()
    st.markdown("### External / Open-Source Datasets")
    st.caption("Third-party open-source datasets for object detection.")

    for i, ds in enumerate(EXTERNAL_DATASETS):
        with st.container(border=True):
            col_info, col_stats, col_action = st.columns([3, 2, 1])

            with col_info:
                st.markdown(f"**{ds['name']}**")
                st.caption(ds["description"])

            with col_stats:
                st.markdown(
                    f"📸 {ds['images']} images · 🏷️ {ds['classes']} classes · "
                    f"💾 {ds['size']} · 📋 {ds['format']}"
                )

            with col_action:
                url = ds.get("url", "")
                is_manual = not url or url.startswith("https://github.com/VisDrone")
                if is_manual:
                    if st.button("🔗 Info", key=f"ext_dl_{i}", use_container_width=True):
                        st.info(f"Manual download required — visit: [{url}]({url})")
                else:
                    if st.button("⬇️ Download", key=f"ext_dl_{i}", use_container_width=True):
                        st.session_state["dataset_name"] = ds["name"]
                        st.session_state["dataset_classes"] = ds["classes"]
                        _run_external_download(ds)


def _render_fallback_native_datasets():
    """Fallback dataset list when flashdet.list_datasets() fails."""
    fallback = [
        {"id": "sample", "name": "Sample (demo)", "classes": 80, "description": "Built-in demo dataset"},
        {"id": "coco2017", "name": "COCO 2017", "classes": 80, "description": "Full COCO 2017 — 118K train + 5K val"},
        {"id": "coco2017-val", "name": "COCO 2017 (val only)", "classes": 80, "description": "COCO 2017 val set only"},
        {"id": "voc2007", "name": "VOC 2007", "classes": 20, "description": "Pascal VOC 2007"},
        {"id": "voc2012", "name": "VOC 2012", "classes": 20, "description": "Pascal VOC 2012"},
    ]
    for i, ds in enumerate(fallback):
        with st.container(border=True):
            col_info, col_action = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{ds['name']}**")
                st.caption(ds["description"])
            with col_action:
                if st.button("⬇️ Download", key=f"fallback_dl_{i}", use_container_width=True):
                    st.session_state["dataset_name"] = ds["name"]
                    st.session_state["dataset_id"] = ds["id"]
                    st.session_state["dataset_classes"] = ds["classes"]
                    _run_flashdet_download(ds["id"])


def _render_dataset_status():
    """Show current dataset selection status."""
    st.markdown("### ✅ Current Dataset")
    dataset = st.session_state.get("dataset_name", None)

    if dataset:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset", dataset)
        with col2:
            num_cls = st.session_state.get("dataset_classes", st.session_state.get("upload_num_classes", 80))
            st.metric("Classes", num_cls)
        with col3:
            fmt = st.session_state.get("ann_format", "COCO JSON")
            st.metric("Format", fmt.split("(")[0].strip() if "(" in fmt else fmt)

        dataset_path = st.session_state.get("dataset_output_path")
        if dataset_path:
            st.caption(f"📁 Path: `{dataset_path}`")

        _render_dataset_verification()

        st.success("Dataset ready! Click **Next** to choose your model.")
    else:
        st.warning("No dataset selected yet. Upload or download a dataset above.")


def _render_dataset_verification():
    """Verification panel: image counts, sample images, annotation stats, validity check."""
    train_path = st.session_state.get("train_img_path", "")
    val_path = st.session_state.get("val_img_path", "")
    train_ann = st.session_state.get("train_ann_path", "")
    val_ann = st.session_state.get("val_ann_path", "")

    if not train_path and not val_path:
        return

    st.markdown("### 🔍 Dataset Verification")

    # Use flashdet's verify_dataset for thorough validation
    dataset_root = st.session_state.get("dataset_output_path", "")
    if dataset_root and os.path.isdir(dataset_root):
        try:
            is_valid = verify_dataset(dataset_root)
            if is_valid:
                st.success("✅ FlashDet dataset verification: **PASSED** — ready for training")
            else:
                st.warning("⚠️ FlashDet dataset verification: issues detected. Check paths and annotations.")
        except Exception as e:
            st.caption(f"Verification check: {e}")

    # Use summarize_coco_root for rich statistics
    if dataset_root and os.path.isdir(dataset_root):
        try:
            summary = summarize_coco_root(dataset_root)
            if summary:
                with st.expander("📊 FlashDet Dataset Summary", expanded=True):
                    cols = st.columns(4)
                    with cols[0]:
                        st.metric("Train Images", summary.get("train_images", 0))
                    with cols[1]:
                        st.metric("Val Images", summary.get("val_images", 0))
                    with cols[2]:
                        st.metric("Total Annotations", summary.get("total_annotations", 0))
                    with cols[3]:
                        st.metric("Categories", summary.get("num_categories", 0))

                    if summary.get("class_distribution"):
                        sorted_classes = sorted(summary["class_distribution"].items(),
                                                key=lambda x: x[1], reverse=True)[:15]
                        st.markdown("**Class distribution (top 15):**")
                        st.bar_chart({name: count for name, count in sorted_classes})
        except Exception:
            pass

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
    valid = True
    issues = []

    # Count images
    train_images = []
    val_images = []
    if train_path and os.path.isdir(train_path):
        train_images = [f for f in os.listdir(train_path) if f.lower().endswith(IMAGE_EXTS)]
    elif train_path:
        valid = False
        issues.append(f"Train image path not found: `{train_path}`")

    if val_path and os.path.isdir(val_path):
        val_images = [f for f in os.listdir(val_path) if f.lower().endswith(IMAGE_EXTS)]
    elif val_path:
        valid = False
        issues.append(f"Val image path not found: `{val_path}`")

    if not train_images and train_path:
        valid = False
        issues.append("No images found in train directory")

    # Check for COCO annotation file (critical for training)
    coco_train_ann = os.path.join(train_path, "_annotations.coco.json") if train_path else ""
    if train_path and os.path.isdir(train_path) and not os.path.isfile(coco_train_ann):
        if not train_ann or not os.path.isfile(train_ann):
            issues.append(
                f"⚠️ Missing `_annotations.coco.json` in train directory. "
                f"FlashDet Trainer expects this file at: `{coco_train_ann}`"
            )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train Images", len(train_images))
    with col2:
        st.metric("Val Images", len(val_images))
    with col3:
        total = len(train_images) + len(val_images)
        st.metric("Total Images", total)
    with col4:
        if valid and total > 0 and not issues:
            st.metric("Status", "✅ Valid")
        elif total > 0:
            st.metric("Status", "⚠️ Issues")
        else:
            st.metric("Status", "❌ Empty")

    # Sample images preview
    with st.expander("🖼️ Sample Images Preview", expanded=False):
        if train_images:
            st.caption("**Training samples:**")
            sample_train = train_images[:4]
            cols = st.columns(min(len(sample_train), 4))
            for i, img_name in enumerate(sample_train):
                img_path = os.path.join(train_path, img_name)
                with cols[i]:
                    try:
                        st.image(img_path, caption=img_name[:20], use_container_width=True)
                    except Exception:
                        st.caption(f"⚠️ {img_name}")
        if val_images:
            st.caption("**Validation samples:**")
            sample_val = val_images[:4]
            cols = st.columns(min(len(sample_val), 4))
            for i, img_name in enumerate(sample_val):
                img_path = os.path.join(val_path, img_name)
                with cols[i]:
                    try:
                        st.image(img_path, caption=img_name[:20], use_container_width=True)
                    except Exception:
                        st.caption(f"⚠️ {img_name}")

    # Annotation statistics (COCO JSON)
    ann_stats = _parse_annotation_stats(train_path, val_path, train_ann, val_ann)
    if ann_stats:
        with st.expander("📊 Annotation Statistics", expanded=False):
            if ann_stats.get("train"):
                st.markdown("**Training Annotations:**")
                ts = ann_stats["train"]
                tc1, tc2, tc3 = st.columns(3)
                with tc1:
                    st.metric("Annotated Images", ts.get("num_images", 0))
                with tc2:
                    st.metric("Total Annotations", ts.get("num_annotations", 0))
                with tc3:
                    st.metric("Categories", ts.get("num_categories", 0))
                if ts.get("class_distribution"):
                    st.markdown("**Class distribution (top 10):**")
                    sorted_classes = sorted(ts["class_distribution"].items(),
                                            key=lambda x: x[1], reverse=True)[:10]
                    class_data = {name: count for name, count in sorted_classes}
                    st.bar_chart(class_data)

            if ann_stats.get("val"):
                st.markdown("**Validation Annotations:**")
                vs = ann_stats["val"]
                vc1, vc2, vc3 = st.columns(3)
                with vc1:
                    st.metric("Annotated Images", vs.get("num_images", 0))
                with vc2:
                    st.metric("Total Annotations", vs.get("num_annotations", 0))
                with vc3:
                    st.metric("Categories", vs.get("num_categories", 0))

    # Issues summary
    if issues:
        for issue in issues:
            st.warning(issue)


def _parse_annotation_stats(train_path: str, val_path: str,
                            train_ann: str, val_ann: str) -> dict:
    """Parse COCO annotation files and return statistics."""
    stats = {}

    for split, img_path, ann_path in [("train", train_path, train_ann),
                                       ("val", val_path, val_ann)]:
        ann_file = ann_path
        if not ann_file and img_path:
            candidate = os.path.join(img_path, "_annotations.coco.json")
            if os.path.isfile(candidate):
                ann_file = candidate

        if not ann_file or not os.path.isfile(ann_file):
            continue

        try:
            with open(ann_file, "r") as f:
                coco = json.load(f)

            images = coco.get("images", [])
            annotations = coco.get("annotations", [])
            categories = coco.get("categories", [])

            cat_id_to_name = {c["id"]: c["name"] for c in categories}
            class_dist = {}
            for ann in annotations:
                cat_id = ann.get("category_id", 0)
                name = cat_id_to_name.get(cat_id, f"class_{cat_id}")
                class_dist[name] = class_dist.get(name, 0) + 1

            stats[split] = {
                "num_images": len(images),
                "num_annotations": len(annotations),
                "num_categories": len(categories),
                "class_distribution": class_dist,
            }
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    return stats


def _use_existing_dataset(path: str, ds_info: dict):
    """Set session state from an already-downloaded dataset."""
    st.session_state["dataset_output_path"] = path
    st.session_state["dataset_name"] = ds_info.get("name", os.path.basename(path))
    st.session_state["dataset_classes"] = ds_info.get("classes", 80)
    if ds_info.get("id"):
        st.session_state["dataset_id"] = ds_info["id"]
    train_dir = os.path.join(path, "train")
    val_dir = os.path.join(path, "valid")
    if os.path.isdir(train_dir):
        st.session_state["train_img_path"] = train_dir
    if os.path.isdir(val_dir):
        st.session_state["val_img_path"] = val_dir
    st.success(f"✅ Using cached dataset at: `{path}`")
    st.rerun()


def _run_flashdet_download(dataset_id: str):
    """Execute dataset download using flashdet Python API."""
    output_dir = st.session_state.get("download_output_dir", None)
    if output_dir is None:
        output_dir = os.path.join("data", dataset_id)

    existing = _dataset_already_downloaded(dataset_id, output_dir)
    if existing:
        st.session_state["dataset_output_path"] = existing
        train_dir = os.path.join(existing, "train")
        val_dir = os.path.join(existing, "valid")
        if os.path.isdir(train_dir):
            st.session_state["train_img_path"] = train_dir
        if os.path.isdir(val_dir):
            st.session_state["val_img_path"] = val_dir
        st.success(f"✅ Dataset already exists at: `{existing}`")
        st.rerun()
        return

    if not _check_network():
        st.error(
            "**Network unavailable.** Cannot reach dataset servers.\n\n"
            "Please check your internet connection and try again. "
            "If you're behind a firewall, the download requires access to `images.cocodataset.org`."
        )
        return

    st.info(
        f"Downloading **{dataset_id}** via `flashdet.download_dataset()`. "
        "This may take a while for large datasets (COCO val2017 is ~1 GB). "
        "Progress is shown in the terminal."
    )

    with st.spinner(f"Downloading `{dataset_id}` — this may take several minutes..."):
        try:
            result_path = download_dataset(
                dataset_id=dataset_id,
                output_dir=output_dir,
            )
            st.session_state["dataset_output_path"] = result_path

            train_dir = os.path.join(result_path, "train")
            val_dir = os.path.join(result_path, "valid")
            if os.path.isdir(train_dir):
                st.session_state["train_img_path"] = train_dir
            if os.path.isdir(val_dir):
                st.session_state["val_img_path"] = val_dir

            st.success(f"✅ Dataset downloaded to: `{result_path}`")
            st.rerun()

        except ValueError as e:
            st.error(f"**Invalid dataset ID:** {e}")
        except FileNotFoundError as e:
            st.error(f"**File not found during setup:** {e}")
        except RuntimeError as e:
            err_msg = str(e)
            if "Download failed" in err_msg:
                st.error(
                    f"**Download failed.** The dataset server may be unreachable.\n\n"
                    f"Details: `{err_msg}`\n\n"
                    "Try again later, or download the dataset manually."
                )
            else:
                st.error(f"**Runtime error:** {e}")
        except (OSError, IOError) as e:
            st.error(
                f"**File system error:** {e}\n\n"
                "Check disk space and write permissions."
            )
        except Exception as e:
            error_type = type(e).__name__
            st.error(f"**Download failed ({error_type}):** {e}")


def _run_external_download(dataset_info: dict):
    """Download external dataset via URL."""
    import urllib.request
    import urllib.error
    import zipfile
    import tempfile

    url = dataset_info.get("url", "")
    name = dataset_info.get("name", "dataset")

    if not url or url.startswith("#") or "github.com/VisDrone" in url:
        st.info(f"**Manual download required.** Visit: [{url}]({url})")
        return

    output_base = os.path.join("data", name.lower().replace(" ", "_").replace("(", "").replace(")", ""))

    if os.path.isdir(output_base) and os.listdir(output_base):
        st.session_state["dataset_output_path"] = output_base
        st.session_state["dataset_name"] = name
        st.success(f"✅ Dataset already exists at: `{output_base}`")
        st.rerun()
        return

    with st.spinner(f"Downloading `{name}` ({dataset_info.get('size', '?')})..."):
        tmp_path = None
        try:
            tmp_file = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
            tmp_path = tmp_file.name
            tmp_file.close()

            urllib.request.urlretrieve(url, tmp_path)

            file_size = os.path.getsize(tmp_path)
            if file_size < 1000:
                st.error(
                    f"Downloaded file is too small ({file_size} bytes). "
                    "The URL may be invalid or redirecting to an error page.\n\n"
                    f"Try downloading manually from: [{url}]({url})"
                )
                return

            # Validate it's actually a zip before extracting
            if not zipfile.is_zipfile(tmp_path):
                st.error(
                    "**Downloaded file is not a valid ZIP archive.** "
                    "The URL may have changed or is redirecting to an HTML page.\n\n"
                    f"Try downloading manually from: [{url}]({url})"
                )
                return

            os.makedirs(output_base, exist_ok=True)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                zf.extractall(output_base)

            st.session_state["dataset_output_path"] = output_base
            st.session_state["dataset_name"] = name

            train_dir = os.path.join(output_base, "train")
            val_dir = os.path.join(output_base, "valid")
            if os.path.isdir(train_dir):
                st.session_state["train_img_path"] = train_dir
            if os.path.isdir(val_dir):
                st.session_state["val_img_path"] = val_dir

            st.success(f"✅ Downloaded and extracted to: `{output_base}`")
            st.rerun()

        except zipfile.BadZipFile:
            st.error(
                "**Downloaded file is not a valid ZIP archive.** "
                "The URL may have changed or is serving an HTML error page.\n\n"
                f"Try downloading manually from: [{url}]({url})"
            )
        except urllib.error.URLError as e:
            st.error(
                f"**Network error:** Could not reach `{url}`.\n\n"
                f"Details: {e.reason if hasattr(e, 'reason') else e}"
            )
        except OSError as e:
            st.error(f"**File system error:** {e}")
        except Exception as e:
            st.error(f"**Download failed ({type(e).__name__}):** {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
