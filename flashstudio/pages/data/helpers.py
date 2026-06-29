"""Ingest helper functions — extraction, conversion, download runners."""

import os
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS, DEFAULT_DATA_DIR, INFER_NUM_CLASSES
from flashstudio.pages.data._common import (
    _HAS_FLASHDET, _FLASHDET_INSTALL_MSG,
    _auto_detect_classes, _check_network, _dataset_already_downloaded,
)

try:
    from flashdet import download_dataset
    from flashdet.data import convert_dataset, detect_dataset_format
except ImportError:
    pass


def _extract_upload(uploaded_file, split):
    import zipfile, tarfile, tempfile
    name = uploaded_file.name
    out = os.path.join(DEFAULT_DATA_DIR, "uploaded", split)
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
                _existing_cls = st.session_state.get("class_names", "")
                if isinstance(_existing_cls, list):
                    _existing_cls = "\n".join(_existing_cls)
                if detected_cls and not _existing_cls.strip():
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
    out = os.path.join(DEFAULT_DATA_DIR, "uploaded", f"{split}_coco")
    cls_names = None
    raw = st.session_state.get("class_names", "")
    if isinstance(raw, list):
        raw = "\n".join(raw)
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
    best = None
    best_count = 0
    for root, dirs, files in os.walk(base):
        img_count = len([f for f in files if f.lower().endswith(IMG_EXTENSIONS)])
        if img_count > best_count:
            best = root
            best_count = img_count
        for d in dirs:
            if d.lower() in ("images", "imgs"):
                c = os.path.join(root, d)
                for sub_root, _, sub_files in os.walk(c):
                    sc = len([f for f in sub_files if f.lower().endswith(IMG_EXTENSIONS)])
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
    st.session_state["dataset_classes"] = ds_info.get("classes", ds_info.get("cls", INFER_NUM_CLASSES))
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
    out = st.session_state.get("download_output_dir") or os.path.join(DEFAULT_DATA_DIR, dataset_id)
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
            detected_cls = _auto_detect_classes()
            if detected_cls:
                st.session_state["class_names"] = "\n".join(detected_cls)
                st.session_state["num_classes"] = len(detected_cls)
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
    out = os.path.join(DEFAULT_DATA_DIR, name.lower().replace(" ", "_").replace("(", "").replace(")", ""))
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
