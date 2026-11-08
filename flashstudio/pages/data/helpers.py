"""Ingest helper functions — extraction, conversion, download runners."""

import json
import os
import signal
import subprocess
import sys
import time
import uuid

import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS, INFER_NUM_CLASSES
from flashstudio.pages.data._common import (
    _HAS_FLASHDET, _FLASHDET_INSTALL_MSG,
    _auto_detect_classes, _dataset_already_downloaded,
    _get_data_root, _is_archive_url, _resolve_download_root,
)
from flashstudio.utils.dataset_paths import (
    ensure_coco_annotation, find_validation_dir, has_images,
    resolve_annotation_file,
)
from flashstudio.utils.dataset_sources import (
    DATASET_URL_MIRRORS as _DATASET_URL_MIRRORS,
)

try:
    from flashdet import download_dataset
    from flashdet.data import convert_dataset, detect_dataset_format
except ImportError:
    pass


def _archive_stem(name: str) -> str:
    lower = name.lower()
    for suffix in (".tar.gz", ".tgz", ".tar", ".zip"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


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
    lower_name = name.lower()
    data_root = _get_data_root()
    out = os.path.join(data_root, "uploaded", split)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    staging = tempfile.mkdtemp(prefix=f".{split}_", dir=os.path.dirname(out))
    archive_path = None
    moved = False

    try:
        if lower_name.endswith(".zip"):
            suffix = ".zip"
        elif lower_name.endswith((".tar", ".tar.gz", ".tgz")) or (
            lower_name.endswith(".gz") and ".tar." in lower_name
        ):
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
                _validate_zip_members(members, staging)
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
        if split == "train" or not st.session_state.get("dataset_output_path"):
            st.session_state["dataset_output_path"] = root

        fmt = "unknown"
        if _HAS_FLASHDET:
            fmt = detect_dataset_format(root)
        if fmt == "unknown" and _find_labels(root):
            # FlashDet does not recognize images/train2017 + labels/train2017
            # (common in COCO128 exports), but the dataset is still YOLO.
            fmt = "txt"
        st.session_state["detected_format"] = fmt

        if fmt in ("txt", "voc"):
            if split == "train" or not st.session_state.get("dataset_name"):
                st.session_state["dataset_name"] = _archive_stem(name)
            if split == "train":
                _clear_split_state("train")
            _convert_to_coco(root, split, fmt)
            return

        split_base = _find_split_base(root, split)
        img_dir = _find_images(split_base) if split_base else _find_images(root)
        if img_dir:
            if split == "train" or not st.session_state.get("dataset_output_path"):
                st.session_state["dataset_output_path"] = root
            if split == "train" or not st.session_state.get("dataset_name"):
                st.session_state["dataset_name"] = _archive_stem(name)
            _clear_split_state(split)
            st.session_state[f"{split}_img_path"] = img_dir
            ann_hint = st.session_state.get(f"{split}_ann_path", "") or _find_ann(split_base or root)
            if ann_hint:
                st.session_state[f"{split}_ann_path"] = ann_hint
            label_dir = _find_labels(split_base or root)
            if label_dir:
                st.session_state[f"{split}_label_path"] = label_dir

            # Auto-find the other split from the same archive.  This handles
            # dataset/train/images + dataset/valid/images as well as the flat
            # dataset/train + dataset/valid layout.
            if split == "train":
                val_dir = find_validation_dir(img_dir, root=root)
                if val_dir:
                    _clear_split_state("val")
                    st.session_state["val_img_path"] = val_dir
                    val_labels = _find_labels(val_dir)
                    if not val_labels:
                        val_labels = _find_labels(os.path.dirname(val_dir))
                    if val_labels:
                        st.session_state["val_label_path"] = val_labels

            if split == "train" and not st.session_state.get("val_img_path"):
                from flashstudio.utils.dataset_split import maybe_auto_split
                from flashstudio.utils import flash as _flash

                _ann = st.session_state.get("train_ann_path") or ann_hint
                _lbl = (
                    st.session_state.get("train_label_path")
                    or _find_labels(split_base or root)
                )
                if maybe_auto_split(img_dir, "", ann_file=_ann, label_dir=_lbl):
                    _flash(
                        "No validation set — created an 80/20 split automatically",
                        "info",
                    )

            train_dir = st.session_state.get("train_img_path", "")
            val_dir = st.session_state.get("val_img_path", "")
            _ensure_dataset_annotations(root, train_dir, val_dir)

            detected_cls = _auto_detect_classes()
            _existing_cls = st.session_state.get("class_names", "")
            if isinstance(_existing_cls, list):
                _existing_cls = "\n".join(_existing_cls)
            if detected_cls and (split == "train" or not _existing_cls.strip()):
                st.session_state["class_names"] = "\n".join(detected_cls)
                st.session_state["num_classes"] = len(detected_cls)

            from flashstudio.utils import flash
            flash(f"Extracted `{name}` to {img_dir}", "success")
            st.session_state["_data_tab_next"] = "Preview"
            st.rerun()
        else:
            st.session_state[f"{split}_img_path"] = root
            if split == "train" or not st.session_state.get("dataset_name"):
                st.session_state["dataset_name"] = _archive_stem(name)
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


def _validate_zip_members(members, dest):
    """Reject ZIP members that would escape *dest* (Zip Slip)."""
    base = os.path.realpath(os.path.abspath(dest))
    for member in members:
        member_name = getattr(member, "filename", member)
        target = os.path.realpath(os.path.join(dest, member_name))
        if target != base and not target.startswith(base + os.sep):
            raise ValueError(f"Unsafe path in ZIP archive: {member_name!r}")


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

def _is_split_like(name: str) -> bool:
    """True for names that are a dataset split, not a wrapper folder."""
    name = name.lower()
    return name in {
        "train", "training", "valid", "val", "validation", "test",
        "images", "imgs", "jpegimages", "labels", "annotations",
    } or name.startswith(("train", "val", "test", "images", "labels"))


def _is_internal_name(name: str) -> bool:
    """True for FlashStudio temporary/internal folders (``_yolo_*``, ``.downloads``)."""
    return name.startswith(("_", "."))


def _find_root(base, _depth=0):
    """Descend through single wrapper folders to the real dataset root.

    Handles archives such as ``VOCdevkit/VOC2007/...`` or
    ``my_dataset/train/images/...`` while never treating a split folder
    (``train``, ``val2017``, ``images``, ...) as the dataset root.
    """
    try:
        entries = sorted(os.listdir(base))
    except OSError:
        return base

    if len(entries) == 1:
        child = os.path.join(base, entries[0])
        if (
            os.path.isdir(child)
            and _depth < 6
            and not _is_split_like(os.path.basename(child))
            and not _is_internal_name(os.path.basename(child))
        ):
            return _find_root(child, _depth + 1)

    # A wrapper may contain several dataset folders (e.g. VOCdevkit/VOC2007
    # plus VOCdevkit/VOC2012).  Pick the first real dataset child, but never
    # mistake a train/val/images split for the dataset root.
    for entry in entries:
        if _is_split_like(entry) or _is_internal_name(entry):
            continue
        full = os.path.join(base, entry)
        if (
            os.path.isdir(full)
            and _HAS_FLASHDET
            and detect_dataset_format(full) != "unknown"
        ):
            return full

    return base


def _find_yolo_pairs(root):
    """Return ``(split, images_dir, labels_dir)`` pairs for a YOLO dataset.

    Handles ``train/images`` + ``train/labels``, ``images/train2017`` +
    ``labels/train2017``, flat ``images`` + ``labels``, and split folders with
    images directly inside them.
    """
    from flashstudio.pages.data.fix.annotations import _yolo_label_dir

    pairs = []
    seen = set()
    for canonical, split_key in (("train", "train"), ("valid", "val"), ("test", "test")):
        split_root = _find_split_base(root, split_key)
        images_dir = _find_images(split_root) if split_root else None
        if not images_dir:
            continue
        labels_dir = _yolo_label_dir(images_dir)
        if not labels_dir or not os.path.isdir(labels_dir):
            continue
        key = (os.path.abspath(images_dir), os.path.abspath(labels_dir))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((canonical, images_dir, labels_dir))

    if not pairs:
        images_dir = _find_images(root)
        labels_dir = _find_labels(root)
        if images_dir and labels_dir:
            pairs.append(("train", images_dir, labels_dir))
    return pairs


def _link_dir_or_copy(src, dst):
    if os.path.exists(dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst, target_is_directory=True)
    except OSError:
        import shutil

        shutil.copytree(src, dst)


def _build_tiny_2class():
    """Create a tiny person/car detection set from the bundled COCO128 sample."""
    import json
    import shutil

    root = os.path.join(_get_data_root(), "coco128", "coco128")
    if not os.path.isdir(root):
        from flashstudio.constants import DEFAULT_DATA_DIR
        root = os.path.join(DEFAULT_DATA_DIR, "coco128", "coco128")
    img_dir = os.path.join(root, "images", "train2017")
    lbl_dir = os.path.join(root, "labels", "train2017")
    if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
        from flashstudio.utils import flash
        flash("Bundled COCO128 sample not found.", "error")
        st.rerun()
        return

    out = os.path.join(_get_data_root(), "uploaded", "tiny2")
    shutil.rmtree(out, ignore_errors=True)

    entries = []
    for name in sorted(os.listdir(lbl_dir)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(lbl_dir, name), "r") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        keep = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if cid in (0, 2):  # COCO: 0=person, 2=car
                keep.append(("0" if cid == 0 else "1") + " " + " ".join(parts[1:5]))
        image = os.path.join(img_dir, os.path.splitext(name)[0] + ".jpg")
        if keep and os.path.isfile(image):
            entries.append((image, keep))
        if len(entries) >= 24:
            break

    if len(entries) < 4:
        from flashstudio.utils import flash
        flash("Not enough person/car images in the bundled sample.", "warning")
        st.rerun()
        return

    split_at = max(1, int(len(entries) * 0.75))
    splits = {
        "train": entries[:split_at],
        "val": entries[split_at:],
    }
    if not splits["val"]:
        splits["val"] = splits["train"][-1:]
        splits["train"] = splits["train"][:-1] or splits["val"]

    from PIL import Image

    state = {}
    for split, split_entries in splits.items():
        s_img = os.path.join(out, split, "images")
        os.makedirs(s_img, exist_ok=True)
        images, annotations = [], []
        ann_id = 1
        for image_payload, lines in split_entries:
            name = os.path.basename(image_payload)
            _copy_or_link(image_payload, os.path.join(s_img, name))
            with Image.open(image_payload) as im:
                width, height = im.size
            image_id = len(images) + 1
            images.append({
                "id": image_id, "file_name": name,
                "width": width, "height": height,
            })
            for line in lines:
                cls, cx, cy, bw, bh = line.split()
                cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                x = (cx - bw / 2) * width
                y = (cy - bh / 2) * height
                w = bw * width
                h = bh * height
                annotations.append({
                    "id": ann_id, "image_id": image_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                })
                ann_id += 1
        json_path = os.path.join(s_img, "_annotations.coco.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({
                "images": images, "annotations": annotations,
                "categories": [
                    {"id": 1, "name": "person", "supercategory": "object"},
                    {"id": 2, "name": "car", "supercategory": "object"},
                ],
            }, fh)
        state[f"{split}_img_path"] = s_img
        state[f"{split}_ann_path"] = json_path
        state[f"{split}_label_path"] = ""

    for key in (
        "_last_verify_ok", "_quality_report", "dataset_output_path",
        "_pending_ann_format", "_fixed_coco_download", "_fixed_coco_name",
    ):
        st.session_state.pop(key, None)
    state.update({
        "dataset_name": "Tiny 2-Class (person, car)",
        "class_names": "person\ncar",
        "num_classes": 2,
        "_pending_ann_format": "COCO JSON",
        "detected_format": "coco",
        "_data_tab_next": "Preview",
    })
    st.session_state.update(state)
    from flashstudio.utils import flash
    flash(f"Built Tiny 2-Class · {len(entries)} images", "success")
    st.rerun()


def _split_yolo_train_pair(images_dir, labels_dir, work_root, val_pct=20):
    """Create train/valid folders from a single YOLO train pair."""
    import shutil

    image_names = sorted(
        name for name in os.listdir(images_dir)
        if name.lower().endswith(IMG_EXTENSIONS)
    )
    if not image_names:
        return False

    val_pct = max(5, min(50, int(val_pct or 20)))
    split_at = int(len(image_names) * (100 - val_pct) / 100.0)
    split_at = min(max(1, split_at), max(1, len(image_names) - 1))
    train_names, valid_names = image_names[:split_at], image_names[split_at:]
    if not valid_names:
        valid_names = train_names[-1:]
        train_names = train_names[:-1] or valid_names

    for split, names in (("train", train_names), ("valid", valid_names)):
        img_out = os.path.join(work_root, split, "images")
        lbl_out = os.path.join(work_root, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for name in names:
            _copy_or_link(os.path.join(images_dir, name), os.path.join(img_out, name))
            label = os.path.join(labels_dir, os.path.splitext(name)[0] + ".txt")
            if os.path.isfile(label):
                _copy_or_link(label, os.path.join(lbl_out, os.path.basename(label)))
    return True


def _convert_yolo_layout(root, out, class_names=None, val_pct=20):
    """Convert a non-standard YOLO layout through a canonical temp folder."""
    import shutil
    import tempfile

    pairs = _find_yolo_pairs(root)
    if not pairs:
        return None

    # Keep the staging folder outside the dataset tree.  If cleanup fails (for
    # example a locked file on Windows) a folder left inside the dataset looks
    # like a valid COCO dataset and can later be auto-selected as the root.
    work_root = tempfile.mkdtemp(prefix="_yolo_")
    try:
        has_valid = any(split == "valid" for split, _i, _l in pairs)
        train_pair = next(
            (pair for pair in pairs if pair[0] == "train"),
            pairs[0] if pairs else None,
        )
        if not has_valid and train_pair:
            if not _split_yolo_train_pair(train_pair[1], train_pair[2], work_root,
                                          val_pct=val_pct):
                return None
        else:
            for split, images_dir, labels_dir in pairs:
                _link_dir_or_copy(
                    images_dir, os.path.join(work_root, split, "images")
                )
                _link_dir_or_copy(
                    labels_dir, os.path.join(work_root, split, "labels")
                )

        names = class_names
        if not names:
            try:
                from flashdet.data.prepare import _read_txt_class_names

                names = _read_txt_class_names(root)
            except Exception:
                names = None
        if not names:
            names = _class_names_for_download(
                os.path.basename(os.path.normpath(root)),
                train_pair[2] if train_pair else None,
            )
        import json

        with open(os.path.join(work_root, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"train: {os.path.join(work_root, 'train', 'images')}\n")
            f.write(f"val: {os.path.join(work_root, 'valid', 'images')}\n")
            f.write(f"nc: {len(names)}\nnames:\n")
            for i, name in enumerate(names):
                f.write(f"  {i}: {json.dumps(name)}\n")

        return convert_dataset(
            source_dir=work_root,
            output_dir=out,
            target_format="coco",
            class_names=names,
        )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def _convert_to_coco(src, split, src_fmt):
    if not _HAS_FLASHDET:
        st.error(_FLASHDET_INSTALL_MSG)
        return
    import shutil

    source_name = os.path.basename(os.path.normpath(src)).replace(" ", "_") or split
    out = os.path.join(
        _get_data_root(), "uploaded", f"{split}_{source_name}_coco"
    )
    # A previous conversion of the same upload must not leak stale files into
    # the new dataset.
    if os.path.isdir(out):
        shutil.rmtree(out, ignore_errors=True)
    cls_names = None
    raw = st.session_state.get("class_names", "")
    if isinstance(raw, list):
        raw = "\n".join(raw)
    if raw.strip():
        cls_names = [c.strip() for c in raw.strip().replace(",", "\n").split("\n") if c.strip()]
    val_pct = int(st.session_state.get("auto_val_pct", 20) or 20)
    with st.spinner("Converting to COCO..."):
        try:
            if src_fmt in ("kitti", "labelme", "csv", "supervisely"):
                from flashstudio.utils.universal_convert import (
                    convert_extra_to_coco,
                )

                stats = convert_extra_to_coco(
                    src, out, val_ratio=val_pct / 100.0,
                )
                if not stats:
                    raise RuntimeError(
                        f"{src_fmt} → COCO conversion produced no images"
                    )
            else:
                try:
                    stats = convert_dataset(
                        source_dir=src, output_dir=out,
                        target_format="coco", class_names=cls_names,
                        val_ratio=val_pct / 100.0,
                    )
                except Exception:
                    if src_fmt != "txt":
                        raise
                    # FlashDet only detects train/images-style YOLO layouts.
                    # Normalize other layouts (images/train2017, flat images, ...)
                    # through a canonical temporary folder.
                    shutil.rmtree(out, ignore_errors=True)
                    stats = _convert_yolo_layout(
                        src, out, cls_names, val_pct=val_pct,
                    )
                    if stats is None:
                        raise
            if stats.get("status") == "already_in_target_format":
                out = src
            st.session_state["dataset_output_path"] = out
            st.session_state["detected_format"] = "coco"
            st.session_state["_pending_ann_format"] = "COCO JSON"
            t = os.path.join(out, "train")
            v = os.path.join(out, "valid")
            if not os.path.isdir(v):
                v = os.path.join(out, "val")
            if os.path.isdir(t):
                st.session_state[f"{split}_img_path"] = _image_dir(t)
            else:
                st.session_state.pop(f"{split}_img_path", None)
            if os.path.isdir(v):
                st.session_state["val_img_path"] = _image_dir(v)
            else:
                st.session_state.pop("val_img_path", None)
            _ensure_dataset_annotations(out)
            detected_cls = _auto_detect_classes()
            if detected_cls:
                st.session_state["class_names"] = "\n".join(detected_cls)
                st.session_state["num_classes"] = len(detected_cls)
            from flashstudio.utils import flash
            flash("Converted to COCO format", "success")
            st.session_state["_data_tab_next"] = "Preview"
            st.rerun()
        except Exception as e:
            st.error(f"Conversion failed: {e}")


def _find_split_base(base, split):
    """Find the directory belonging to *split* inside an extracted archive."""
    names = {
        "train": ("train", "training"),
        "val": ("val", "valid", "validation", "test"),
    }.get(split, (split,))

    def _matches(name):
        name = name.lower()
        if name in names:
            return True
        if split == "train":
            return name.startswith("train")
        if split in ("val", "valid"):
            return name.startswith(("val", "valid", "test"))
        return name.startswith(split)

    candidates = []
    for root, dirs, _ in os.walk(base):
        dirs.sort()
        for d in dirs:
            if _matches(d):
                candidates.append(os.path.join(root, d))
    for candidate in candidates:
        if has_images(candidate):
            return candidate
        if has_images(os.path.join(candidate, "images")):
            return candidate
        if _find_images(candidate):
            return candidate
    return candidates[0] if candidates else ""


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


def _clear_split_state(split):
    """Clear paths, reports, and classes for one split being replaced."""
    keys = (
        ("train_img_path", "train_ann_path", "train_label_path",
         "_quality_report_train", "_fix_report_train", "_last_verify_ok")
        if split == "train"
        else ("val_img_path", "val_ann_path", "val_label_path",
              "_quality_report_val", "_fix_report_val", "_last_verify_ok")
    )
    for key in keys:
        st.session_state.pop(key, None)
    if split == "train":
        for key in ("class_names", "num_classes", "_class_names_synced"):
            st.session_state.pop(key, None)


def _clear_loaded_paths():
    """Remove split/annotation paths when switching to another dataset."""
    for key in ("train_img_path", "val_img_path", "train_ann_path", "val_ann_path",
                "train_label_path", "val_label_path", "_last_verify_ok",
                "_quality_report", "_fixed_coco_download", "_fixed_coco_name",
                "class_names", "num_classes", "_class_names_synced"):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if str(key).startswith(("_quality_report_", "_fix_report_")):
            st.session_state.pop(key, None)


def _image_dir(split_root):
    """Return the directory that directly contains a split's images."""
    if not split_root or not os.path.isdir(split_root):
        return ""
    from flashstudio.utils.dataset_paths import has_images
    if has_images(split_root):
        return split_root
    found = _find_images(split_root)
    return found or split_root


def _ensure_canonical_split_links(root, train_dir="", val_dir=""):
    """Create root/train and root/valid links for non-standard split names.

    FlashDet's verifier and summary helpers only understand ``train`` and
    ``valid``.  This makes official COCO (train2017/val2017) and ``val``
    layouts work without changing the actual data.
    """
    if not root or not os.path.isdir(root):
        return
    root = os.path.abspath(root)

    def _maybe_link(target, link_name):
        if not target or not os.path.isdir(target):
            return
        target = os.path.abspath(target)
        if os.path.dirname(target) != root:
            return
        link = os.path.join(root, link_name)
        if os.path.islink(link):
            if os.path.exists(link) and os.path.realpath(link) == os.path.realpath(target):
                return
            try:
                os.unlink(link)
            except OSError:
                return
        elif os.path.exists(link):
            return
        try:
            os.symlink(target, link, target_is_directory=True)
        except OSError:
            pass

    if train_dir:
        name = os.path.basename(os.path.normpath(train_dir)).lower()
        if name == "train" or name == "training" or name.startswith("train"):
            _maybe_link(train_dir, "train")
    if val_dir:
        name = os.path.basename(os.path.normpath(val_dir)).lower()
        if name in ("val", "valid", "validation") or name.startswith("val"):
            _maybe_link(val_dir, "valid")


def _ensure_dataset_annotations(root, train_dir="", val_dir=""):
    """Normalize annotations for the loaded train/val directories.

    FlashDet requires ``_annotations.coco.json`` inside each split directory.
    Uploaded datasets often keep it in a parent folder or an ``annotations/``
    sibling, so create a link/copy at the expected location and record it in
    session state for the other tabs.
    """
    train_dir = train_dir or st.session_state.get("train_img_path", "")
    val_dir = val_dir or st.session_state.get("val_img_path", "")

    if train_dir and os.path.isdir(train_dir):
        train_ann = ensure_coco_annotation(
            train_dir, split="train",
            hint=st.session_state.get("train_ann_path", ""), root=root,
        )
        if train_ann:
            st.session_state["train_ann_path"] = train_ann
    if val_dir and os.path.isdir(val_dir):
        val_ann = ensure_coco_annotation(
            val_dir, split="val",
            hint=st.session_state.get("val_ann_path", ""), root=root,
        )
        if val_ann:
            st.session_state["val_ann_path"] = val_ann

    _ensure_canonical_split_links(root, train_dir, val_dir)


def _use_existing(path, ds_info):
    from flashstudio.utils import flash
    from flashstudio.utils.dataset_paths import resolve_annotation_file

    name = ds_info.get("name", os.path.basename(path))

    # A downloaded folder may still be VOC/KITTI/LabelMe/CSV/YOLO. Route it
    # through the finalizer instead of loading bare images with no boxes.
    train_dir = os.path.join(path, "train")
    train_images = _image_dir(train_dir) if os.path.isdir(train_dir) else ""
    ready = bool(
        train_images
        and resolve_annotation_file(train_images, split="train", root=path)
    )
    if not ready and _finalize_downloaded_dataset(path, name):
        st.session_state["_data_tab_next"] = "Preview"
        st.rerun()

    st.session_state["dataset_output_path"] = path
    st.session_state["dataset_name"] = name
    st.session_state["dataset_classes"] = ds_info.get("classes", ds_info.get("cls", INFER_NUM_CLASSES))
    if ds_info.get("id"):
        st.session_state["dataset_id"] = ds_info["id"]
    _clear_loaded_paths()

    t = os.path.join(path, "train")
    v = os.path.join(path, "valid")
    if not os.path.isdir(v):
        v = os.path.join(path, "val")
    if os.path.isdir(t):
        st.session_state["train_img_path"] = _image_dir(t)
    if os.path.isdir(v):
        st.session_state["val_img_path"] = _image_dir(v)
    if not st.session_state.get("val_img_path") and st.session_state.get("train_img_path"):
        found_val = find_validation_dir(
            st.session_state["train_img_path"], root=path
        )
        if found_val:
            st.session_state["val_img_path"] = _image_dir(found_val)

    # A validation folder with images but no annotations is not usable.
    val_img = st.session_state.get("val_img_path", "")
    if val_img:
        from flashstudio.utils.dataset_paths import resolve_annotation_file
        from flashstudio.pages.data.fix.annotations import _yolo_label_dir

        val_ann = resolve_annotation_file(val_img, split="val", root=path)
        val_lbl = _yolo_label_dir(
            val_img, st.session_state.get("val_label_path", ""),
        )
        has_lbl = bool(
            val_lbl and any(f.endswith(".txt") for f in os.listdir(val_lbl))
        )
        if not val_ann and not has_lbl:
            st.session_state.pop("val_img_path", None)

    if not st.session_state.get("val_img_path") and st.session_state.get("train_img_path"):
        from flashstudio.utils.dataset_paths import resolve_annotation_file
        from flashstudio.pages.data.fix.annotations import _yolo_label_dir
        from flashstudio.utils.dataset_split import maybe_auto_split

        tr = st.session_state["train_img_path"]
        ann = resolve_annotation_file(
            tr, split="train",
            hint=st.session_state.get("train_ann_path", ""),
        )
        lbl = _yolo_label_dir(
            tr, st.session_state.get("train_label_path", ""),
        )
        if maybe_auto_split(tr, "", ann_file=ann, label_dir=lbl):
            flash(
                "No validation set found — created an 80/20 split automatically",
                "info",
            )

    _ensure_dataset_annotations(path)
    if st.session_state.get("train_ann_path"):
        st.session_state["detected_format"] = "coco"

    detected_cls = _auto_detect_classes()
    if detected_cls:
        st.session_state["class_names"] = "\n".join(detected_cls)
        st.session_state["num_classes"] = len(detected_cls)

    flash(f"Dataset `{name}` loaded", "success")
    st.session_state["_data_tab_next"] = "Preview"
    st.rerun()


# Older flashdet releases hard-code dataset hosts that intermittently refuse
# connections (notably host.robots.ox.ac.uk from Google Colab).  Point those
# datasets at reachable official/mirror hosts until every supported flashdet
# version ships working URLs.  Newer flashdet releases accept a list of
# mirrors themselves, so their registry is left untouched.


def _dataset_registry_urls(dataset_id):
    """Return flashdet's URL mapping for *dataset_id*, or ``None``."""
    try:
        from flashdet.data.download import DATASET_REGISTRY
    except Exception:
        return None
    info = DATASET_REGISTRY.get(dataset_id)
    if not isinstance(info, dict):
        return None
    urls = info.get("urls")
    return urls if isinstance(urls, dict) else None


def _supports_url_lists(urls):
    """True when flashdet handles multiple mirrors per file itself."""
    return any(isinstance(value, (list, tuple)) for value in urls.values())


def _apply_dataset_mirror(dataset_id, index=0):
    """Patch a legacy flashdet registry entry to use the *index*-th mirror."""
    urls = _dataset_registry_urls(dataset_id)
    mirrors = _DATASET_URL_MIRRORS.get(dataset_id)
    if not urls or not mirrors or _supports_url_lists(urls):
        return
    for key, candidates in mirrors.items():
        if key in urls and candidates:
            urls[key] = candidates[min(index, len(candidates) - 1)]


def _download_dataset_with_mirrors(dataset_id, output_dir):
    """Download a dataset, retrying failed hosts with known mirrors."""
    urls = _dataset_registry_urls(dataset_id)
    mirrors = _DATASET_URL_MIRRORS.get(dataset_id)
    if not urls or not mirrors or _supports_url_lists(urls):
        return download_dataset(dataset_id=dataset_id, output_dir=output_dir)

    attempts = max(len(candidates) for candidates in mirrors.values())
    for index in range(attempts):
        _apply_dataset_mirror(dataset_id, index)
        try:
            return download_dataset(dataset_id=dataset_id, output_dir=output_dir)
        except RuntimeError as exc:
            # Only retry when a host failed; a post-processing error would
            # just fail again with the next mirror.
            if "Download failed" not in str(exc) or index + 1 >= attempts:
                raise
    raise RuntimeError(f"Download failed for {dataset_id}")


_DOWNLOAD_ACTIVE_STATES = {
    "starting", "downloading", "extracting", "preparing", "finalizing",
}


def _download_state():
    return st.session_state.get("_dataset_download")


def _recover_download_job():
    """Restore an active download after a page reload/session reconnect."""
    job_root = os.path.abspath(os.path.join(_get_data_root(), ".downloads"))
    if not os.path.isdir(job_root):
        return
    try:
        names = [
            name for name in os.listdir(job_root)
            if name.endswith(".status.json")
        ]
        names.sort(
            key=lambda name: os.path.getmtime(os.path.join(job_root, name)),
            reverse=True,
        )
    except OSError:
        return

    for name in names:
        status_path = os.path.join(job_root, name)
        status = _read_download_status(status_path)
        if status.get("state") not in _DOWNLOAD_ACTIVE_STATES:
            continue
        pid = status.get("pid")
        if not _pid_alive(pid):
            continue
        token = name[: -len(".status.json")]
        config_path = os.path.join(job_root, f"{token}.config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        st.session_state["_dataset_download"] = {
            "mode": config.get("mode"),
            "dataset_id": config.get("dataset_id", ""),
            "dataset_classes": config.get("dataset_classes"),
            "name": config.get("name", config.get("dataset_id", "dataset")),
            "output_dir": config.get("output_dir", ""),
            "cache_dir": config.get("cache_dir", ""),
            "status_path": status_path,
            "cancel_path": os.path.join(job_root, f"{token}.cancel"),
            "config_path": config_path,
            "log_path": os.path.join(job_root, f"{token}.log"),
            "pid": pid,
            "proc": None,
            "started": status.get("updated", time.time()),
            "state": status.get("state", "starting"),
            "message": status.get("message", "Downloading..."),
            "percent": status.get("percent"),
        }
        return


def _reap_download_process(state):
    proc = state.get("proc") if state else None
    if proc is not None:
        try:
            proc.wait(timeout=2)
        except Exception:
            pass


def _download_is_active():
    state = _download_state()
    return bool(state and state.get("state") in _DOWNLOAD_ACTIVE_STATES)


def _pid_alive(pid):
    if not pid:
        return False
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            if f.read().split()[2] == "Z":
                return False
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _read_download_status(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def _fmt_bytes(value):
    try:
        value = float(value or 0)
    except (TypeError, ValueError):
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _download_log_tail(path, lines=8):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
        return "".join(content[-lines:]).strip()
    except OSError:
        return ""


def _load_native_downloaded(dataset_id, out, display_name=None, classes=None):
    """Load a completed flashdet download into session state."""
    from flashstudio.utils import flash

    train_dir = _image_dir(os.path.join(out, "train"))
    val_dir = _image_dir(os.path.join(out, "valid"))
    if not os.path.isdir(val_dir):
        val_dir = _image_dir(os.path.join(out, "val"))
    if not train_dir or not os.path.isdir(train_dir):
        found = _dataset_already_downloaded(dataset_id, out)
        if found:
            train_dir = _image_dir(os.path.join(found, "train"))
            val_dir = _image_dir(os.path.join(found, "valid"))
    if not train_dir or not os.path.isdir(train_dir):
        flash(f"Downloaded files for `{dataset_id}` are missing a train split.", "error")
        return False

    _clear_loaded_paths()
    st.session_state["dataset_output_path"] = out
    st.session_state["dataset_name"] = display_name or dataset_id
    st.session_state["dataset_id"] = dataset_id
    if classes:
        st.session_state["dataset_classes"] = classes
    st.session_state["train_img_path"] = train_dir
    if val_dir and os.path.isdir(val_dir):
        st.session_state["val_img_path"] = val_dir
    else:
        found_val = find_validation_dir(train_dir, root=out)
        if found_val:
            st.session_state["val_img_path"] = _image_dir(found_val)
    _ensure_dataset_annotations(out)
    st.session_state["detected_format"] = "coco"
    detected = _auto_detect_classes()
    if detected:
        st.session_state["class_names"] = "\n".join(detected)
        st.session_state["num_classes"] = len(detected)
    flash(f"Downloaded `{dataset_id}` successfully", "success")
    return True


def _start_dataset_download(mode, output_dir, **extra):
    """Start a background download worker and return its state."""
    from flashstudio.utils import flash

    if _download_is_active():
        flash("A dataset download is already running. Cancel it first.", "warning")
        return None

    job_root = os.path.abspath(os.path.join(_get_data_root(), ".downloads"))
    os.makedirs(job_root, exist_ok=True)
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir) and not os.path.isdir(output_dir):
        flash(
            f"Download folder is a file, not a directory: {output_dir}",
            "error",
        )
        return None
    token = uuid.uuid4().hex
    status_path = os.path.join(job_root, f"{token}.status.json")
    cancel_path = os.path.join(job_root, f"{token}.cancel")
    config_path = os.path.join(job_root, f"{token}.config.json")
    log_path = os.path.join(job_root, f"{token}.log")

    if extra.get("cache_dir"):
        extra["cache_dir"] = os.path.abspath(extra["cache_dir"])
    config = {
        "mode": mode,
        "output_dir": output_dir,
        "status_path": status_path,
        "cancel_path": cancel_path,
    }
    config.update(extra)
    _write_json(config_path, config)
    _write_json(status_path, {
        "state": "starting",
        "message": "Preparing download...",
        "bytes": 0,
        "total": 0,
        "percent": None,
        "result": "",
        "error": "",
    })

    package_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        package_root
        + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m",
                 "flashstudio.utils.download_worker", config_path],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=package_root if os.path.isdir(package_root) else None,
                env=env,
            )
    except Exception as exc:
        flash(f"Could not start download: {exc}", "error")
        return None

    state = {
        "mode": mode,
        "dataset_id": extra.get("dataset_id", ""),
        "dataset_classes": extra.get("dataset_classes"),
        "name": extra.get("name", extra.get("dataset_id", "dataset")),
        "output_dir": output_dir,
        "cache_dir": extra.get("cache_dir", ""),
        "status_path": status_path,
        "cancel_path": cancel_path,
        "config_path": config_path,
        "log_path": log_path,
        "pid": proc.pid,
        "proc": proc,
        "started": time.time(),
        "state": "starting",
        "message": "Preparing download...",
        "percent": None,
    }
    st.session_state["_dataset_download"] = state
    return state


def _start_flashdet_download(dataset_id, display_name=None, classes=None):
    from flashstudio.utils import flash

    if not _HAS_FLASHDET:
        st.error(_FLASHDET_INSTALL_MSG)
        return
    out = os.path.join(_resolve_download_root(), dataset_id)
    existing = _dataset_already_downloaded(dataset_id, out)
    if existing:
        if _load_native_downloaded(
            dataset_id, existing, display_name=display_name, classes=classes
        ):
            st.session_state["_data_tab_next"] = "Preview"
            st.rerun()
        return
    cache_dir = st.session_state.get("download_cache_dir") or os.path.join(
        os.path.expanduser("~"), ".cache", "flashdet"
    )
    state = _start_dataset_download(
        "flashdet", out,
        dataset_id=dataset_id,
        cache_dir=cache_dir,
        name=display_name or dataset_id,
        dataset_classes=classes,
    )
    if state:
        flash(f"Started download of `{dataset_id}`", "info")
        st.rerun()


def _start_external_download(ds):
    import shutil
    from flashstudio.utils import flash

    url, name = ds.get("url", ""), ds.get("name", "dataset")
    if not _is_archive_url(url):
        st.info(f"Manual download: [{url}]({url})")
        return
    out = os.path.join(
        _resolve_download_root(),
        name.lower().replace(" ", "_").replace("(", "").replace(")", ""),
    )
    if os.path.isdir(out) and os.listdir(out):
        if _finalize_downloaded_dataset(out, name):
            st.session_state["_data_tab_next"] = "Preview"
            st.rerun()
            return
        shutil.rmtree(out, ignore_errors=True)
    state = _start_dataset_download("external", out, url=url, name=name)
    if state:
        flash(f"Started download of `{name}`", "info")
        st.rerun()


def _cleanup_download_partials(state):
    try:
        with open(state.get("config_path", ""), "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        config = {}
    mode = config.get("mode")
    if mode == "flashdet":
        cache = config.get("cache_dir") or os.path.join(
            os.path.expanduser("~"), ".cache", "flashdet"
        )
        folder = os.path.join(cache, config.get("dataset_id", ""))
        try:
            for name in os.listdir(folder):
                if name.endswith(".part"):
                    os.remove(os.path.join(folder, name))
        except OSError:
            pass
    elif mode == "external":
        out = config.get("output_dir", "")
        parent = os.path.dirname(out) or "."
        prefix = os.path.basename(out) + ".download."
        try:
            for name in os.listdir(parent):
                if name.startswith(prefix) or name.endswith(".part"):
                    os.remove(os.path.join(parent, name))
        except OSError:
            pass


def _cancel_dataset_download():
    from flashstudio.utils import flash

    state = _download_state()
    if not state:
        flash("No download to cancel", "warning")
        return
    try:
        with open(state["cancel_path"], "w", encoding="utf-8") as f:
            f.write("cancel")
    except OSError:
        pass

    proc = state.get("proc")
    pid = state.get("pid")
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
    elif pid and _pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    _cleanup_download_partials(state)
    st.session_state.pop("_dataset_download", None)
    st.session_state["_data_tab_next"] = "Download"
    flash("Dataset download canceled", "info")


def _poll_dataset_download():
    """Advance a background download and finalize it when complete."""
    from flashstudio.utils import flash

    state = _download_state()
    if not state:
        return None
    status = _read_download_status(state.get("status_path", ""))
    if status:
        for key in ("state", "message", "percent", "bytes", "total", "error",
                    "file_index", "file_count"):
            if key in status:
                state[key] = status[key]
        st.session_state["_dataset_download"] = state

    proc = state.get("proc")
    alive = proc.poll() is None if proc is not None else _pid_alive(state.get("pid"))
    current = state.get("state")

    if current == "done":
        state["state"] = "finalizing"
        st.session_state["_dataset_download"] = state
        ok = False
        if state.get("mode") == "flashdet":
            ok = _load_native_downloaded(
                state.get("dataset_id", ""),
                state.get("output_dir", ""),
                display_name=state.get("name"),
                classes=state.get("dataset_classes"),
            )
        else:
            ok = bool(_finalize_downloaded_dataset(
                state.get("output_dir", ""), state.get("name", "dataset")
            ))
        _reap_download_process(state)
        if not ok:
            if state.get("mode") == "external":
                import shutil

                shutil.rmtree(state.get("output_dir", ""), ignore_errors=True)
            st.session_state.pop("_dataset_download", None)
            flash("Download finished but the dataset could not be loaded.", "error")
            st.rerun()
            return state
        st.session_state.pop("_dataset_download", None)
        st.session_state["_data_tab_next"] = "Preview"
        st.rerun()
        return state

    if current == "cancelled":
        _reap_download_process(state)
        st.session_state.pop("_dataset_download", None)
        st.session_state["_data_tab_next"] = "Download"
        flash("Dataset download canceled", "info")
        st.rerun()
        return state

    if current == "error":
        detail = state.get("error") or state.get("message") or "Unknown error"
        tail = _download_log_tail(state.get("log_path", ""))
        if tail:
            detail += f"\n\nLast log output:\n{tail}"
        _reap_download_process(state)
        st.session_state.pop("_dataset_download", None)
        flash(f"Download failed: {detail[:1500]}", "error")
        st.rerun()
        return state

    if not alive:
        # The worker exited without a terminal status (killed, OOM, import
        # failure, ...).  Surface the tail of its log instead of hanging.
        tail = _download_log_tail(state.get("log_path", ""))
        state["state"] = "error"
        state["message"] = "Download worker stopped unexpectedly."
        state["error"] = tail[-1500:]
        st.session_state["_dataset_download"] = state
        _reap_download_process(state)
        st.session_state.pop("_dataset_download", None)
        flash(
            "Download failed: worker stopped unexpectedly."
            + (f"\n\n{tail[-1000:]}" if tail else ""),
            "error",
        )
        st.rerun()
        return state

    return state


def _render_dataset_download_status():
    """Render download progress/cancel controls, independent of the active tab."""
    if not _download_state():
        _recover_download_job()
    state = _poll_dataset_download()
    if not state:
        return
    state = _download_state()
    if not state:
        return

    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=1000, limit=None, key="dataset_download_autorefresh")
    except Exception:
        if st.button("Refresh download status", key="refresh_download_status"):
            st.rerun()

    label = state.get("dataset_id") or state.get("name") or "dataset"
    current = state.get("state", "starting")
    with st.container(border=True):
        st.markdown(f"#### Downloading {label}")
        message = str(state.get("message") or current.title())
        percent = state.get("percent")
        if isinstance(percent, (int, float)):
            st.progress(
                max(0.0, min(1.0, float(percent))),
                text=message,
            )
            total = state.get("total") or 0
            done = state.get("bytes") or 0
            if total:
                st.caption(
                    f"{_fmt_bytes(done)} of {_fmt_bytes(total)}"
                    + (
                        f"  ·  file {state.get('file_index')}/{state.get('file_count')}"
                        if state.get("file_count") else ""
                    )
                )
            else:
                st.caption(message)
        else:
            st.info(message)
        output_dir = state.get("output_dir")
        if output_dir:
            st.caption(f"Saving to `{output_dir}`")
        if st.button(
            "Cancel download", key="cancel_dataset_download",
            use_container_width=True,
        ):
            _cancel_dataset_download()
            st.rerun()


def _run_flashdet_download(dataset_id):
    if not _HAS_FLASHDET:
        st.error(_FLASHDET_INSTALL_MSG)
        return
    out = os.path.join(_resolve_download_root(), dataset_id)
    existing = _dataset_already_downloaded(dataset_id, out)
    if existing:
        _clear_loaded_paths()
        st.session_state["dataset_output_path"] = existing
        for d, k in [("train", "train_img_path"), ("valid", "val_img_path")]:
            p = os.path.join(existing, d)
            if os.path.isdir(p):
                st.session_state[k] = _image_dir(p)
        if not st.session_state.get("val_img_path") and st.session_state.get("train_img_path"):
            found_val = find_validation_dir(
                st.session_state["train_img_path"], root=existing
            )
            if found_val:
                st.session_state["val_img_path"] = _image_dir(found_val)
        _ensure_dataset_annotations(existing)
        detected_cls = _auto_detect_classes()
        if detected_cls:
            st.session_state["class_names"] = "\n".join(detected_cls)
            st.session_state["num_classes"] = len(detected_cls)
        st.session_state["_data_tab_next"] = "Preview"
        st.rerun()
        return
    with st.spinner(f"Downloading {dataset_id}..."):
        try:
            result = _download_dataset_with_mirrors(dataset_id, out)
            _clear_loaded_paths()
            st.session_state["dataset_output_path"] = result
            for d, k in [("train", "train_img_path"), ("valid", "val_img_path")]:
                p = os.path.join(result, d)
                if os.path.isdir(p):
                    st.session_state[k] = _image_dir(p)
            if not st.session_state.get("val_img_path") and st.session_state.get("train_img_path"):
                found_val = find_validation_dir(
                    st.session_state["train_img_path"], root=result
                )
                if found_val:
                    st.session_state["val_img_path"] = _image_dir(found_val)
            _ensure_dataset_annotations(result)
            detected_cls = _auto_detect_classes()
            if detected_cls:
                st.session_state["class_names"] = "\n".join(detected_cls)
                st.session_state["num_classes"] = len(detected_cls)
            from flashstudio.utils import flash
            flash(f"Downloaded `{dataset_id}` successfully", "success")
            st.session_state["_data_tab_next"] = "Preview"
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
        for root, _dirs, files in os.walk(labels_dir):
            for label in files:
                if not label.endswith(".txt"):
                    continue
                try:
                    with open(os.path.join(root, label), encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            parts = line.split()
                            if parts:
                                max_id = max(max_id, int(float(parts[0])))
                except (OSError, ValueError):
                    continue

    if (
        ("coco128" in name.lower() or ("coco" in name.lower() and max_id >= 79))
        and 0 <= max_id < len(COCO_CLASSES)
    ):
        return list(COCO_CLASSES)

    raw = st.session_state.get("class_names", "")
    if isinstance(raw, list):
        existing = [str(c).strip() for c in raw if str(c).strip()]
    else:
        existing = [c.strip() for c in str(raw).splitlines() if c.strip()]
    if len(existing) > max_id and existing:
        return existing
    return [f"class_{i}" for i in range(max_id + 1)]


def _find_kitti_dirs(root):
    for img_rel, lbl_rel in (
        (("training", "image_2"), ("training", "label_2")),
        (("image_2",), ("label_2",)),
        (("images",), ("labels",)),
    ):
        img_dir = os.path.join(root, *img_rel)
        lbl_dir = os.path.join(root, *lbl_rel)
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            return img_dir, lbl_dir
    return None, None


def _convert_kitti_layout(root, out, val_pct=20):
    """Convert a KITTI object-detection folder to COCO train/valid splits."""
    import json
    import random

    from PIL import Image

    img_dir, lbl_dir = _find_kitti_dirs(root)
    if not img_dir:
        return None

    image_names = sorted(
        name for name in os.listdir(img_dir)
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not image_names:
        return None

    entries = []
    classes = set()
    for name in image_names:
        label_path = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        boxes = []
        if os.path.isfile(label_path):
            with open(label_path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 8 or parts[0] in ("DontCare", "Misc"):
                        continue
                    try:
                        x1, y1, x2, y2 = map(float, parts[4:8])
                    except ValueError:
                        continue
                    if x2 <= x1 or y2 <= y1:
                        continue
                    boxes.append((parts[0], [x1, y1, x2 - x1, y2 - y1]))
                    classes.add(parts[0])
        try:
            with Image.open(os.path.join(img_dir, name)) as im:
                width, height = im.size
        except Exception:
            continue
        entries.append({
            "path": os.path.join(img_dir, name),
            "name": name,
            "width": width,
            "height": height,
            "boxes": boxes,
        })

    if not entries or not classes:
        return None

    class_names = sorted(classes)
    cat_ids = {name: i + 1 for i, name in enumerate(class_names)}
    rng = random.Random(0)
    rng.shuffle(entries)
    val_pct = max(5, min(50, int(val_pct or 20)))
    split_at = int(len(entries) * (100 - val_pct) / 100.0)
    split_at = min(max(1, split_at), max(1, len(entries) - 1))
    splits = {"train": entries[:split_at], "valid": entries[split_at:]}
    if not splits["valid"]:
        splits["valid"] = splits["train"][-1:]
        splits["train"] = splits["train"][:-1] or splits["valid"]

    import shutil

    shutil.rmtree(out, ignore_errors=True)
    stats = {}
    for split, split_entries in splits.items():
        img_out = os.path.join(out, split, "images")
        os.makedirs(img_out, exist_ok=True)
        images, annotations = [], []
        ann_id = 1
        for image_id, entry in enumerate(split_entries, start=1):
            _copy_or_link(entry["path"], os.path.join(img_out, entry["name"]))
            images.append({
                "id": image_id,
                "file_name": entry["name"],
                "width": entry["width"],
                "height": entry["height"],
            })
            for cls, bbox in entry["boxes"]:
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_ids[cls],
                    "bbox": bbox,
                    "area": float(bbox[2]) * float(bbox[3]),
                    "iscrowd": 0,
                })
                ann_id += 1
        with open(os.path.join(img_out, "_annotations.coco.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({
                "images": images,
                "annotations": annotations,
                "categories": [
                    {"id": cid, "name": cname, "supercategory": "object"}
                    for cname, cid in cat_ids.items()
                ],
            }, fh)
        stats[split] = len(images)
    return stats


def _finalize_downloaded_dataset(root: str, name: str) -> str | None:
    """Normalize a downloaded dataset so it becomes trainable.

    Handles COCO folders and YOLO `images/` + `labels/` archives (such as
    COCO128) by creating a train/valid split and converting YOLO to COCO.
    Returns the loaded image directory or ``None``.
    """
    from flashstudio.constants import COCO_CLASSES  # noqa: F401
    from flashstudio.utils import flash

    _clear_loaded_paths()
    val_pct = int(st.session_state.get("auto_val_pct", 20) or 20)
    root = _find_root(root) or root
    split_base = _find_split_base(root, "train")
    images_dir = _find_images(split_base) if split_base else None
    if not images_dir:
        images_dir = _find_images(root)
    from flashstudio.pages.data.fix.annotations import _yolo_label_dir

    labels_dir = _yolo_label_dir(images_dir) if images_dir else ""
    train_coco = os.path.join(root, "train")
    valid_coco = os.path.join(root, "valid")
    if not os.path.isdir(valid_coco):
        valid_coco = os.path.join(root, "val")

    # Already a COCO-style dataset.  Do not mistake a YOLO train/valid layout
    # for COCO just because the split folders exist.
    train_coco_ann = (
        resolve_annotation_file(train_coco, split="train", root=root)
        if os.path.isdir(train_coco) else ""
    )
    val_coco_ann = (
        resolve_annotation_file(valid_coco, split="val", root=root)
        if os.path.isdir(valid_coco) else ""
    )
    if (
        os.path.isdir(train_coco)
        and os.path.isdir(valid_coco)
        and (train_coco_ann or val_coco_ann)
    ):
        train_images = _image_dir(train_coco)
        val_images = _image_dir(valid_coco)
        st.session_state["dataset_output_path"] = root
        st.session_state["dataset_name"] = name
        st.session_state["train_img_path"] = train_images
        st.session_state["val_img_path"] = val_images
        _ensure_dataset_annotations(root, train_images, val_images)
        detected = _auto_detect_classes()
        if detected:
            st.session_state["class_names"] = "\n".join(detected)
            st.session_state["num_classes"] = len(detected)
        flash(f"Dataset `{name}` loaded", "success")
        return train_images

    if not images_dir:
        st.error("Downloaded archive did not contain an image folder.")
        return None

    # Official COCO layout (train2017/val2017 + annotations/) or any other
    # non-standard split names.  Detect it by the COCO JSON rather than by
    # hard-coded directory names.
    if _HAS_FLASHDET:
        # FlashDet's detect_dataset_format only recognizes
        # ``_annotations.coco.json``; official COCO uses
        # ``annotations/instances_train2017.json``, so resolve the annotation
        # directly instead of relying on hard-coded directory names.
        is_coco = bool(
            resolve_annotation_file(images_dir, split="train", root=root)
        )
        if is_coco:
            val_images = find_validation_dir(images_dir, root=root)
            st.session_state["dataset_output_path"] = root
            st.session_state["dataset_name"] = name
            st.session_state["train_img_path"] = images_dir
            if val_images:
                st.session_state["val_img_path"] = val_images
            _ensure_dataset_annotations(root, images_dir, val_images)
            detected = _auto_detect_classes()
            if detected:
                st.session_state["class_names"] = "\n".join(detected)
                st.session_state["num_classes"] = len(detected)
            if val_images:
                flash(f"Dataset `{name}` loaded", "success")
            else:
                from flashstudio.utils.dataset_split import maybe_auto_split

                ann = resolve_annotation_file(
                    images_dir, split="train", root=root,
                )
                if maybe_auto_split(images_dir, "", ann_file=ann):
                    pct = int(st.session_state.get("auto_val_pct", 20) or 20)
                    flash(
                        f"Dataset `{name}` has no validation split — "
                        f"auto-created a {100 - pct}/{pct} split.",
                        "success",
                    )
                else:
                    flash(
                        f"Dataset `{name}` loaded without a validation split; "
                        "add one before training.",
                        "warning",
                    )
            return images_dir

    # KITTI object-detection layout (training/image_2 + training/label_2).
    kitti_out = os.path.join(
        os.path.dirname(root), os.path.basename(root) + "_coco"
    )
    kitti_stats = _convert_kitti_layout(root, kitti_out, val_pct=val_pct)
    if kitti_stats:
        train_path = os.path.join(kitti_out, "train")
        valid_path = os.path.join(kitti_out, "valid")
        st.session_state["dataset_output_path"] = kitti_out
        st.session_state["dataset_name"] = name
        st.session_state["train_img_path"] = _image_dir(train_path)
        st.session_state["val_img_path"] = _image_dir(valid_path)
        _ensure_dataset_annotations(kitti_out)
        st.session_state["detected_format"] = "coco"
        st.session_state["_pending_ann_format"] = "COCO JSON"
        detected = _auto_detect_classes()
        if detected:
            st.session_state["class_names"] = "\n".join(detected)
            st.session_state["num_classes"] = len(detected)
        flash(
            f"Dataset `{name}` converted from KITTI to COCO "
            f"({kitti_stats.get('train', 0)} train · "
            f"{kitti_stats.get('valid', 0)} val)",
            "success",
        )
        return train_path

    # LabelMe / CSV / Supervisely fallbacks.
    from flashstudio.utils.universal_convert import (
        convert_extra_to_coco, detect_extra_format,
    )

    extra_fmt = detect_extra_format(root)
    if extra_fmt:
        extra_out = os.path.join(
            os.path.dirname(root), os.path.basename(root) + "_coco"
        )
        extra_stats = convert_extra_to_coco(
            root, extra_out, val_ratio=val_pct / 100.0,
        )
        if extra_stats:
            train_path = _image_dir(os.path.join(extra_out, "train"))
            valid_path = _image_dir(os.path.join(extra_out, "valid"))
            st.session_state["dataset_output_path"] = extra_out
            st.session_state["dataset_name"] = name
            st.session_state["train_img_path"] = train_path
            st.session_state["val_img_path"] = valid_path
            _ensure_dataset_annotations(extra_out)
            st.session_state["detected_format"] = "coco"
            st.session_state["_pending_ann_format"] = "COCO JSON"
            detected = _auto_detect_classes()
            if detected:
                st.session_state["class_names"] = "\n".join(detected)
                st.session_state["num_classes"] = len(detected)
            flash(
                f"Dataset `{name}` converted from {extra_fmt.upper()} to COCO",
                "success",
            )
            return train_path

    # Pascal VOC layouts (images/ + annotations/*.xml).
    if _HAS_FLASHDET:
        try:
            src_fmt = detect_dataset_format(root)
        except Exception:
            src_fmt = ""
        if src_fmt == "voc":
            import shutil

            coco_out = os.path.join(
                os.path.dirname(root), os.path.basename(root) + "_coco"
            )
            shutil.rmtree(coco_out, ignore_errors=True)
            try:
                stats = convert_dataset(
                    source_dir=root, output_dir=coco_out,
                    target_format="coco", val_ratio=val_pct / 100.0,
                )
            except Exception as exc:
                flash(f"VOC→COCO conversion failed: {exc}", "error")
                stats = None
            if stats:
                train_path = _image_dir(os.path.join(coco_out, "train"))
                valid_path = _image_dir(os.path.join(coco_out, "valid"))
                st.session_state["dataset_output_path"] = coco_out
                st.session_state["dataset_name"] = name
                st.session_state["train_img_path"] = train_path
                st.session_state["val_img_path"] = valid_path
                _ensure_dataset_annotations(coco_out)
                st.session_state["detected_format"] = "coco"
                st.session_state["_pending_ann_format"] = "COCO JSON"
                detected = _auto_detect_classes()
                if detected:
                    st.session_state["class_names"] = "\n".join(detected)
                    st.session_state["num_classes"] = len(detected)
                flash(
                    f"Dataset `{name}` converted from VOC to COCO",
                    "success",
                )
                return train_path

    # YOLO-style archive: normalize into train/valid (preserving an existing
    # validation split), then convert to COCO.
    if labels_dir and _HAS_FLASHDET:
        import shutil

        names = _class_names_for_download(name, labels_dir)
        coco_out = os.path.join(os.path.dirname(root), os.path.basename(root) + "_coco")
        shutil.rmtree(coco_out, ignore_errors=True)
        stats = None
        try:
            stats = _convert_yolo_layout(
                root, coco_out, class_names=names, val_pct=val_pct,
            )
        except Exception as exc:
            flash(f"YOLO→COCO conversion failed: {exc}", "error")

        if stats:
            train_path = os.path.join(coco_out, "train")
            valid_path = os.path.join(coco_out, "valid")
            if not os.path.isdir(valid_path):
                valid_path = os.path.join(coco_out, "val")
            st.session_state["dataset_output_path"] = coco_out
            st.session_state["dataset_name"] = name
            st.session_state["train_img_path"] = _image_dir(train_path)
            st.session_state["val_img_path"] = _image_dir(valid_path)
            _ensure_dataset_annotations(coco_out)
            st.session_state["detected_format"] = "coco"
            st.session_state["_pending_ann_format"] = "COCO JSON"
            detected = _auto_detect_classes()
            if detected:
                st.session_state["class_names"] = "\n".join(detected)
                st.session_state["num_classes"] = len(detected)
            else:
                st.session_state["class_names"] = "\n".join(names)
                st.session_state["num_classes"] = len(names)
            flash(f"Dataset `{name}` converted from YOLO to COCO", "success")
            return train_path

        st.session_state["train_img_path"] = images_dir
        st.session_state["train_label_path"] = labels_dir
        found_val = find_validation_dir(images_dir, root=root)
        if found_val:
            st.session_state["val_img_path"] = found_val
        st.session_state["dataset_name"] = name
        st.session_state["_pending_ann_format"] = "YOLO TXT"
        return images_dir

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
    import tarfile
    import tempfile
    from urllib.parse import urlparse

    url, name = ds.get("url", ""), ds.get("name", "dataset")
    if not _is_archive_url(url):
        st.info(f"Manual download: [{url}]({url})")
        return

    out = os.path.join(_get_data_root(), name.lower().replace(" ", "_").replace("(", "").replace(")", ""))
    if os.path.isdir(out) and os.listdir(out):
        if _finalize_downloaded_dataset(out, name):
            st.session_state["_data_tab_next"] = "Preview"
            st.rerun()
            return
        # The previous extraction is unusable; remove it and download again.
        import shutil

        shutil.rmtree(out, ignore_errors=True)

    path = urlparse(url).path.lower()
    if path.endswith(".zip"):
        suffix = ".zip"
    elif path.endswith((".tar.gz", ".tgz")):
        suffix = ".tar.gz"
    else:
        suffix = ".tar"

    with st.spinner(f"Downloading {name}..."):
        tmp_path = None
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp_path = tmp.name
            tmp.close()
            urllib.request.urlretrieve(url, tmp_path)
            os.makedirs(out, exist_ok=True)

            if suffix == ".zip":
                if os.path.getsize(tmp_path) < 1000 or not zipfile.is_zipfile(tmp_path):
                    st.error(f"Downloaded file is not a valid ZIP archive. Manual: [{url}]({url})")
                    return
                with zipfile.ZipFile(tmp_path, "r") as zf:
                    members = zf.namelist()
                    _validate_zip_members(members, out)
                    total = len(members)
                    progress = st.progress(0, text=f"Extracting {name}  —  0 / {total}")
                    img_count = 0
                    for i, member in enumerate(members, 1):
                        zf.extract(member, out)
                        if member.lower().endswith(IMG_EXTENSIONS):
                            img_count += 1
                        progress.progress(i / total, text=f"Extracting {name}  —  {i} / {total} files  ·  {img_count} images")
                    progress.progress(1.0, text=f"Extracted {total} files  ·  {img_count} images")
            else:
                try:
                    with tarfile.open(tmp_path, "r:*") as tf:
                        members = tf.getmembers()
                        if not members:
                            st.error(f"Archive is empty. Manual: [{url}]({url})")
                            return
                        _validate_tar_members(members, out)
                        total = len(members)
                        progress = st.progress(0, text=f"Extracting {name}  —  0 / {total}")
                        img_count = 0
                        for i, member in enumerate(members, 1):
                            _extract_tar_member(tf, member, out)
                            if member.name.lower().endswith(IMG_EXTENSIONS):
                                img_count += 1
                            progress.progress(i / total, text=f"Extracting {name}  —  {i} / {total} files  ·  {img_count} images")
                        progress.progress(1.0, text=f"Extracted {total} files  ·  {img_count} images")
                except tarfile.TarError:
                    st.error(f"Downloaded file is not a valid TAR archive. Manual: [{url}]({url})")
                    return

            st.session_state.update({"dataset_output_path": out, "dataset_name": name})
            if not _finalize_downloaded_dataset(out, name):
                import shutil

                shutil.rmtree(out, ignore_errors=True)
                st.session_state.pop("dataset_output_path", None)
                return
            st.session_state["_data_tab_next"] = "Preview"
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
