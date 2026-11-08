"""Train/validation subset controls for loaded datasets."""

import json
import os
import random
import shutil

import streamlit as st

from flashstudio.pages.data._common import _get_data_root, _load_data_prompt
from flashstudio.pages.data.helpers import _copy_or_link


def _image_files(root):
    from flashstudio.constants import IMG_EXTENSIONS_ALL

    found = []
    if not root or not os.path.isdir(root):
        return found
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(IMG_EXTENSIONS_ALL):
                found.append(os.path.join(dirpath, name))
    return found


def _slug(text):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))
    return safe.strip("_") or "dataset"


def _output_root(dataset_name, pct):
    out = os.path.join(
        _get_data_root(), "uploaded",
        f"subset_{_slug(dataset_name)}_{int(pct)}pct",
    )
    shutil.rmtree(out, ignore_errors=True)
    return out


def _unique(name, used):
    stem, ext = os.path.splitext(name)
    candidate = name
    idx = 1
    while candidate in used:
        candidate = f"{stem}_{idx}{ext}"
        idx += 1
    used.add(candidate)
    return candidate


def _annotation_index(img_dir):
    by_name = {}
    for path in _image_files(img_dir):
        by_name.setdefault(os.path.basename(path), path)
    return by_name


def _collect_coco_split(img_dir, ann_file, categories):
    """Return image entries for one COCO split."""
    with open(ann_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    cat_names = {}
    for cat in data.get("categories", []):
        name = str(cat.get("name") or cat.get("id"))
        cat_names[cat["id"]] = name
        categories.setdefault(name, None)

    anns_by_image = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(ann.get("image_id"), []).append(ann)

    index = _annotation_index(img_dir)
    entries = []
    for image in data.get("images", []):
        file_name = str(image.get("file_name") or "")
        path = os.path.join(img_dir, file_name)
        if not os.path.isfile(path):
            path = index.get(os.path.basename(file_name), "")
        if not path:
            continue
        entries.append({
            "path": path,
            "name": os.path.basename(file_name) or os.path.basename(path),
            "width": image.get("width"),
            "height": image.get("height"),
            "annotations": anns_by_image.get(image.get("id"), []),
            "cat_names": cat_names,
        })
    return entries


def _write_coco_split(out_dir, entries, category_ids):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    used = set()
    images = []
    annotations = []
    ann_id = 1

    for image_id, entry in enumerate(entries, start=1):
        name = _unique(entry["name"], used)
        _copy_or_link(entry["path"], os.path.join(img_dir, name))
        images.append({
            "id": image_id,
            "file_name": name,
            "width": entry.get("width"),
            "height": entry.get("height"),
        })
        for ann in entry["annotations"]:
            cat_name = entry["cat_names"].get(ann.get("category_id"))
            if cat_name is None or cat_name not in category_ids:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            copied = dict(ann)
            copied["id"] = ann_id
            copied["image_id"] = image_id
            copied["category_id"] = category_ids[cat_name]
            copied["bbox"] = list(bbox)
            if "area" not in copied:
                copied["area"] = float(bbox[2]) * float(bbox[3])
            copied.setdefault("iscrowd", 0)
            annotations.append(copied)
            ann_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": cid, "name": name, "supercategory": "object"}
            for name, cid in category_ids.items()
        ],
    }
    json_path = os.path.join(img_dir, "_annotations.coco.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return img_dir, json_path


def _sample(entries, pct, rng):
    """Take *pct* percent of one split (at least one image)."""
    if not entries:
        return []
    shuffled = list(entries)
    rng.shuffle(shuffled)
    count = max(1, int(round(len(shuffled) * pct / 100.0)))
    return shuffled[: min(count, len(shuffled))]


def _clear_derived_state():
    for key in (
        "_last_verify_ok", "_quality_report", "dataset_output_path",
        "_pending_ann_format", "_fixed_coco_download", "_fixed_coco_name",
    ):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if str(key).startswith(("_quality_report_", "_fix_report_")):
            st.session_state.pop(key, None)


def _render_split():
    train_path = st.session_state.get("train_img_path", "")
    val_path = st.session_state.get("val_img_path", "")
    if not train_path and not val_path:
        _load_data_prompt("Load a dataset before selecting a subset.", "split_load")
        return

    train_imgs = _image_files(train_path)
    val_imgs = _image_files(val_path)
    if not train_imgs and not val_imgs:
        st.info("No images found in the loaded dataset.")
        return

    st.caption(
        f"Current · **{len(train_imgs):,}** train · **{len(val_imgs):,}** validation"
    )
    pct = st.slider(
        "Use this share of each split", 5, 100, 100, 5,
        key="subset_pct",
        help="Samples the same percentage from the existing train and "
             "validation sets. The split boundary is preserved.",
    )
    seed = st.number_input("Random seed", 0, 9999, 42, key="subset_seed")

    train_use = max(1, int(round(len(train_imgs) * pct / 100))) if train_imgs else 0
    val_use = max(1, int(round(len(val_imgs) * pct / 100))) if val_imgs else 0
    st.caption(
        f"Will use **{train_use:,} / {len(train_imgs):,}** train images and "
        f"**{val_use:,} / {len(val_imgs):,}** validation images."
    )

    st.caption("Quick tiny dataset presets")
    p1, p2, p3 = st.columns(3)
    for col, label, value in (
        (p1, "Tiny · 10%", 10),
        (p2, "Small · 25%", 25),
        (p3, "Half · 50%", 50),
    ):
        with col:
            if st.button(
                label, key=f"subset_preset_{value}", use_container_width=True,
            ):
                try:
                    _subset_dataset(train_path, val_path, value, seed)
                except Exception as exc:
                    st.error(f"Could not build the subset: {exc}")

    if st.button("Apply subset", type="primary",
                 key="apply_subset_btn", use_container_width=True):
        try:
            _subset_dataset(train_path, val_path, pct, seed)
        except Exception as exc:
            st.error(f"Could not build the subset: {exc}")


def _subset_dataset(train_path, val_path, pct, seed):
    from flashstudio.utils.dataset_paths import resolve_annotation_file

    dataset_name = st.session_state.get("dataset_name") or "dataset"
    out = _output_root(dataset_name, pct)

    train_ann = (
        resolve_annotation_file(
            train_path, split="train",
            hint=st.session_state.get("train_ann_path", ""),
        ) if train_path else None
    )
    val_ann = (
        resolve_annotation_file(
            val_path, split="val",
            hint=st.session_state.get("val_ann_path", ""),
        ) if val_path else None
    )

    if train_ann or val_ann:
        _subset_coco(
            train_path, val_path, train_ann, val_ann, out, pct, seed,
            dataset_name,
        )
    else:
        _subset_yolo(train_path, val_path, out, pct, seed, dataset_name)

    st.success(
        f"Using {int(pct)}% of each split. "
        "Train and validation boundaries were kept."
    )
    st.rerun()


def _subset_coco(train_path, val_path, train_ann, val_ann, out,
                 pct, seed, dataset_name):
    rng = random.Random(int(seed))
    categories = {}
    train_entries = (
        _collect_coco_split(train_path, train_ann, categories)
        if train_path and train_ann else []
    )
    val_entries = (
        _collect_coco_split(val_path, val_ann, categories)
        if val_path and val_ann else []
    )
    if not train_entries and not val_entries:
        raise ValueError("No annotated images were found in the loaded splits.")

    train_entries = _sample(train_entries, pct, rng)
    val_entries = _sample(val_entries, pct, rng)
    category_ids = {name: i + 1 for i, name in enumerate(categories)}

    tr_imgs, tr_json = _write_coco_split(
        os.path.join(out, "train"), train_entries, category_ids,
    )
    va_imgs, va_json = _write_coco_split(
        os.path.join(out, "val"), val_entries, category_ids,
    )

    st.session_state.update({
        "train_img_path": tr_imgs,
        "val_img_path": va_imgs,
        "train_ann_path": tr_json,
        "val_ann_path": va_json,
        "train_label_path": "",
        "val_label_path": "",
        "class_names": "\n".join(category_ids),
        "num_classes": len(category_ids),
        "ann_format": "COCO JSON",
        "detected_format": "coco",
        "dataset_name": f"{dataset_name} ({int(pct)}% subset)",
    })
    _clear_derived_state()


def _subset_yolo(train_path, val_path, out, pct, seed, dataset_name):
    from flashstudio.pages.data.fix.annotations import _yolo_label_dir

    rng = random.Random(int(seed))
    split_pairs = {}
    for split, img_dir, hint_key in (
        ("train", train_path, "train_label_path"),
        ("val", val_path, "val_label_path"),
    ):
        pairs = []
        if img_dir:
            label_dir = _yolo_label_dir(
                img_dir, st.session_state.get(hint_key, ""),
            )
            for image in _image_files(img_dir):
                label = ""
                if label_dir:
                    candidate = os.path.join(
                        label_dir,
                        os.path.splitext(os.path.basename(image))[0] + ".txt",
                    )
                    if os.path.isfile(candidate):
                        label = candidate
                pairs.append((image, label))
        split_pairs[split] = _sample(pairs, pct, rng)

    if not split_pairs["train"] and not split_pairs["val"]:
        raise ValueError("No images were found in the loaded splits.")

    state = {}
    for split, pairs in split_pairs.items():
        img_out = os.path.join(out, split, "images")
        lbl_out = os.path.join(out, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for image, label in pairs:
            name = os.path.basename(image)
            _copy_or_link(image, os.path.join(img_out, name))
            target = os.path.join(lbl_out, os.path.splitext(name)[0] + ".txt")
            if label:
                _copy_or_link(label, target)
            else:
                open(target, "w").close()
        state[f"{split}_img_path"] = img_out
        state[f"{split}_label_path"] = lbl_out
        state[f"{split}_ann_path"] = ""

    state.update({
        "ann_format": "YOLO",
        "detected_format": "txt",
        "dataset_name": f"{dataset_name} ({int(pct)}% subset)",
    })
    st.session_state.update(state)
    _clear_derived_state()
