"""Ingest helper functions — extraction, conversion, download runners."""

import os
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS, INFER_NUM_CLASSES
from flashstudio.pages.data._common import (
    _HAS_FLASHDET, _FLASHDET_INSTALL_MSG,
    _auto_detect_classes, _check_network, _dataset_already_downloaded,
    _get_data_root,
)

try:
    from flashdet import download_dataset
    from flashdet.data import convert_dataset, detect_dataset_format
except ImportError:
    pass


def _extract_upload(uploaded_file, split):
    """Extract an uploaded ZIP/TAR into a staging directory, then swap it in.

    Extraction is staged so a corrupt upload never destroys the dataset that
    is already loaded for this project.  TAR members are validated against
    path traversal and unsafe links.
    """
    import zipfile
    import tarfile
    import tempfile
    import shutil

    name = uploaded_file.name
    data_root = _get_data_root()
    out = os.path.join(data_root, "uploaded", split)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    staging = tempfile.mkdtemp(prefix=f".{split}_", dir=os.path.dirname(out))
    archive_path = None
    moved = False

    try:
        if name.endswith(".zip"):
            suffix = ".zip"
        elif name.endswith((".tar", ".tar.gz", ".tgz")) or (name.endswith(".gz") and ".tar." in name):
            suffix = ".tar.gz"
        else:
            st.error(f"Unsupported archive type: {name}")
            return

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        archive_path = tmp.name
        tmp.write(uploaded_file.read())
        tmp.flush()
        tmp.close()
        uploaded_file.seek(0)

        if suffix == ".zip":
            if not zipfile.is_zipfile(archive_path):
                st.error(f"❌ **Invalid ZIP file:** `{name}` is not a valid zip archive.")
                return
            with zipfile.ZipFile(archive_path, "r") as zf:
                members = zf.infolist()
                total = len(members)
                if total == 0:
                    st.error(f"❌ **Empty archive:** `{name}` contains no files.")
                    return
                if total > 100_000 or sum(m.file_size for m in members) > 50 * 1024 ** 3:
                    st.error(f"❌ **Archive too large:** `{name}` exceeds the extraction limit.")
                    return
                progress = st.progress(0, text=f"Extracting {name}  —  0 / {total} files")
                img_count = 0
                for i, member in enumerate(members, 1):
                    zf.extract(member, staging)
                    if member.filename.lower().endswith(IMG_EXTENSIONS):
                        img_count += 1
                    progress.progress(i / total, text=f"Extracting {name}  —  {i} / {total} files  ·  {img_count} images")
                progress.progress(1.0, text=f"✓ Extracted {total} files  ·  {img_count} images")
        else:
            try:
                with tarfile.open(archive_path, "r:*") as tf:
                    members = tf.getmembers()
                    total = len(members)
                    if total == 0:
                        st.error(f"❌ **Empty archive:** `{name}` contains no files.")
                        return
                    if total > 100_000:
                        st.error(f"❌ **Archive too large:** `{name}` has too many members.")
                        return
                    _validate_tar_members(members, staging)
                    progress = st.progress(0, text=f"Extracting {name}  —  0 / {total} files")
                    img_count = 0
                    for i, member in enumerate(members, 1):
                        _extract_tar_member(tf, member, staging)
                        if member.name.lower().endswith(IMG_EXTENSIONS):
                            img_count += 1
                        progress.progress(i / total, text=f"Extracting {name}  —  {i} / {total} files  ·  {img_count} images")
                    progress.progress(1.0, text=f"✓ Extracted {total} files  ·  {img_count} images")
            except tarfile.TarError:
                st.error(f"❌ **Invalid TAR file:** `{name}` is not a valid tar archive.")
                return

        if not os.listdir(staging):
            st.error(f"❌ **Empty archive:** `{name}` contains no files.")
            return

        # Replace the previous split atomically only after the new archive has
        # been fully extracted and validated.
        if os.path.islink(out):
            os.unlink(out)
        elif os.path.isdir(out):
            shutil.rmtree(out)
        elif os.path.isfile(out):
            os.unlink(out)
        os.replace(staging, out)
        moved = True

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

            parent = os.path.dirname(img_dir)
            for sibling, key in [("valid", "val"), ("val", "val"), ("test", "test"), ("train", "train")]:
                sib_path = os.path.join(parent, sibling)
                if os.path.isdir(sib_path) and key != split:
                    sk = "val_img_path" if key in ("val",) else f"{key}_img_path"
                    if not st.session_state.get(sk):
                        st.session_state[sk] = sib_path

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
    finally:
        if not moved and os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
        if archive_path and os.path.exists(archive_path):
            try:
                os.unlink(archive_path)
            except OSError:
                pass


def _validate_tar_members(members, dest):
    """Reject TAR members that would escape *dest* (including via links)."""
    base = os.path.realpath(os.path.abspath(dest))
    for member in members:
        target = os.path.realpath(os.path.join(dest, member.name))
        if target != base and not target.startswith(base + os.sep):
            raise ValueError(f"Unsafe path in TAR archive: {member.name!r}")
        if member.isdev():
            raise ValueError(f"Device file in TAR archive: {member.name!r}")
        if member.issym() or member.islnk():
            link = os.path.realpath(os.path.join(os.path.dirname(target), member.linkname))
            if link != base and not link.startswith(base + os.sep):
                raise ValueError(f"Unsafe link in TAR archive: {member.name!r}")


def _extract_tar_member(tf, member, dest):
    """Extract one already-validated TAR member, using the data filter if available."""
    try:
        tf.extract(member, dest, filter="data")
    except TypeError:  # Python < 3.12
        tf.extract(member, dest)

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
    out = os.path.join(_get_data_root(), "uploaded", f"{split}_coco")
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
            st.session_state["_pending_ann_format"] = "COCO JSON"
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
    t = os.path.join(path, "train")
    v = os.path.join(path, "valid")
    if not os.path.isdir(v):
        v = os.path.join(path, "val")
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
    out = st.session_state.get("download_output_dir") or os.path.join(_get_data_root(), dataset_id)
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


def _copy_or_link(src: str, dst: str):
    if os.path.exists(dst):
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        import shutil

        shutil.copy2(src, dst)


def _class_names_for_download(name: str, labels_dir: str | None) -> list[str]:
    """Choose class names for a downloaded YOLO dataset."""
    from flashstudio.constants import COCO_CLASSES

    max_id = -1
    if labels_dir and os.path.isdir(labels_dir):
        for label in os.listdir(labels_dir):
            if not label.endswith(".txt"):
                continue
            try:
                with open(os.path.join(labels_dir, label), encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            max_id = max(max_id, int(float(parts[0])))
            except (OSError, ValueError):
                continue

    if "coco" in name.lower() and 0 <= max_id < len(COCO_CLASSES):
        return list(COCO_CLASSES)

    raw = st.session_state.get("class_names", "")
    if isinstance(raw, list):
        existing = [str(c).strip() for c in raw if str(c).strip()]
    else:
        existing = [c.strip() for c in str(raw).splitlines() if c.strip()]
    if len(existing) > max_id and existing:
        return existing
    return [f"class_{i}" for i in range(max_id + 1)]


def _finalize_downloaded_dataset(root: str, name: str) -> str | None:
    """Normalize a downloaded dataset so it becomes trainable.

    Handles COCO folders and YOLO `images/` + `labels/` archives (such as
    COCO128) by creating a train/valid split and converting YOLO to COCO.
    Returns the loaded image directory or ``None``.
    """
    from flashstudio.constants import COCO_CLASSES  # noqa: F401
    from flashstudio.utils import flash

    images_dir = _find_images(root)
    labels_dir = _find_labels(root)
    train_coco = os.path.join(root, "train")
    valid_coco = os.path.join(root, "valid")
    if not os.path.isdir(valid_coco):
        valid_coco = os.path.join(root, "val")

    # Already a COCO-style dataset.
    if os.path.isdir(train_coco) and os.path.isdir(valid_coco):
        st.session_state["dataset_output_path"] = root
        st.session_state["dataset_name"] = name
        st.session_state["train_img_path"] = train_coco
        st.session_state["val_img_path"] = valid_coco
        detected = _auto_detect_classes()
        if detected:
            st.session_state["class_names"] = "\n".join(detected)
            st.session_state["num_classes"] = len(detected)
        flash(f"Dataset `{name}` loaded", "success")
        return train_coco

    if not images_dir:
        st.error("Downloaded archive did not contain an image folder.")
        return None

    # YOLO-style archive: normalise into train/valid, then convert to COCO.
    if labels_dir and _HAS_FLASHDET:
        import shutil
        import tempfile

        images = sorted(
            f for f in os.listdir(images_dir)
            if f.lower().endswith(IMG_EXTENSIONS)
        )
        if not images:
            st.error("Downloaded archive contained no images.")
            return None

        split_at = max(1, int(len(images) * 0.8))
        train_imgs, valid_imgs = images[:split_at], images[split_at:]
        if not valid_imgs:
            valid_imgs = train_imgs[-1:]
            train_imgs = train_imgs[:-1] or valid_imgs

        yolo_root = tempfile.mkdtemp(prefix="_yolo_", dir=os.path.dirname(root))
        for split, items in (("train", train_imgs), ("valid", valid_imgs)):
            img_out = os.path.join(yolo_root, split, "images")
            lbl_out = os.path.join(yolo_root, split, "labels")
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(lbl_out, exist_ok=True)
            for filename in items:
                _copy_or_link(os.path.join(images_dir, filename),
                              os.path.join(img_out, filename))
                label = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
                if os.path.isfile(label):
                    _copy_or_link(label, os.path.join(lbl_out, os.path.basename(label)))

        names = _class_names_for_download(name, labels_dir)
        with open(os.path.join(yolo_root, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"train: {os.path.join(yolo_root, 'train', 'images')}\n")
            f.write(f"val: {os.path.join(yolo_root, 'valid', 'images')}\n")
            f.write(f"nc: {len(names)}\nnames:\n")
            for i, class_name in enumerate(names):
                f.write(f"  {i}: {class_name}\n")

        coco_out = os.path.join(os.path.dirname(root), os.path.basename(root) + "_coco")
        try:
            from flashdet.data import convert_dataset

            convert_dataset(yolo_root, coco_out, target_format="coco")
            shutil.rmtree(yolo_root, ignore_errors=True)
        except Exception as exc:
            shutil.rmtree(yolo_root, ignore_errors=True)
            flash(f"YOLO→COCO conversion failed: {exc}", "error")
            st.session_state["train_img_path"] = images_dir
            st.session_state["train_label_path"] = labels_dir
            st.session_state["dataset_name"] = name
            st.session_state["_pending_ann_format"] = "YOLO TXT"
            return images_dir

        train_path = os.path.join(coco_out, "train")
        valid_path = os.path.join(coco_out, "valid")
        if not os.path.isdir(valid_path):
            valid_path = os.path.join(coco_out, "val")
        st.session_state["dataset_output_path"] = coco_out
        st.session_state["dataset_name"] = name
        st.session_state["train_img_path"] = train_path
        st.session_state["val_img_path"] = valid_path
        st.session_state["_pending_ann_format"] = "COCO JSON"
        detected = _auto_detect_classes()
        if detected:
            st.session_state["class_names"] = "\n".join(detected)
            st.session_state["num_classes"] = len(detected)
        else:
            st.session_state["class_names"] = "\n".join(names)
            st.session_state["num_classes"] = len(names)
        flash(f"Dataset `{name}` prepared: {len(train_imgs)} train · {len(valid_imgs)} val", "success")
        return train_path

    # Images without labels: use the folder directly and let the user annotate.
    st.session_state["dataset_output_path"] = root
    st.session_state["dataset_name"] = name
    st.session_state["train_img_path"] = images_dir
    st.session_state["_pending_ann_format"] = "None"
    flash(f"Downloaded `{name}` — no labels found; add annotations before training.", "warning")
    return images_dir


def _run_external_download(ds):
    import urllib.request
    import urllib.error
    import zipfile
    import tempfile
    url, name = ds.get("url", ""), ds.get("name", "dataset")
    if not url or "VisDrone" in url:
        st.info(f"Manual: [{url}]({url})")
        return
    out = os.path.join(_get_data_root(), name.lower().replace(" ", "_").replace("(", "").replace(")", ""))
    if os.path.isdir(out) and os.listdir(out):
        _finalize_downloaded_dataset(out, name)
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
                members = zf.namelist()
                total = len(members)
                progress = st.progress(0, text=f"Extracting {name}  —  0 / {total}")
                img_count = 0
                for i, member in enumerate(members, 1):
                    zf.extract(member, out)
                    if member.lower().endswith(IMG_EXTENSIONS):
                        img_count += 1
                    progress.progress(i / total, text=f"Extracting {name}  —  {i} / {total} files  ·  {img_count} images")
                progress.progress(1.0, text=f"Extracted {total} files  ·  {img_count} images")
            st.session_state.update({"dataset_output_path": out, "dataset_name": name})
            _finalize_downloaded_dataset(out, name)
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
