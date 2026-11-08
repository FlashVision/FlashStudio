"""Convert common annotation formats that the bundled converters miss.

COCO, Pascal VOC, YOLO, and KITTI already have dedicated paths. This module
adds LabelMe JSON, CSV bounding boxes, and Supervisely so every dataset the UI
offers can be normalized to COCO.
"""

import csv
import json
import os
import random
import shutil

from PIL import Image

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _images(root):
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(IMG_EXT):
                found.append(os.path.join(dirpath, name))
    return found


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _find_kitti(root):
    for img_rel, lbl_rel in (
        (("training", "image_2"), ("training", "label_2")),
        (("image_2",), ("label_2",)),
    ):
        img_dir = os.path.join(root, *img_rel)
        lbl_dir = os.path.join(root, *lbl_rel)
        if os.path.isdir(img_dir) and os.path.isdir(lbl_dir):
            return img_dir, lbl_dir
    return None, None


def _labelme_files(root):
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".json"):
                continue
            if name in ("_annotations.coco.json", "meta.json"):
                continue
            if name.startswith(("instances_", "instances-", "captions_")):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and "shapes" in data:
                found.append((path, data))
    return found


def _csv_file(root):
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".csv"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
                    header = next(csv.reader(fh), [])
            except OSError:
                continue
            lower = {h.strip().lower() for h in header}
            bbox_keys = (
                {"xmin", "ymin", "xmax", "ymax"} <= lower
                or {"x1", "y1", "x2", "y2"} <= lower
                or {"x_center", "y_center", "width", "height"} <= lower
            )
            if bbox_keys:
                return path, header
    return None, None


def detect_extra_format(root):
    """Return kitti / supervisely / labelme / csv, or ''."""
    if _find_kitti(root)[0]:
        return "kitti"
    if os.path.isfile(os.path.join(root, "meta.json")) and any(
        os.path.isdir(os.path.join(root, d, "ann"))
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ):
        return "supervisely"
    if _labelme_files(root):
        return "labelme"
    if _csv_file(root)[0]:
        return "csv"
    return ""


def _image_index(root):
    return {os.path.basename(p): p for p in _images(root)}


def _collect_labelme(root, class_names):
    entries = []
    index = _image_index(root)
    for json_path, data in _labelme_files(root):
        image_name = str(data.get("imagePath") or "")
        image_path = os.path.join(os.path.dirname(json_path), image_name)
        if not os.path.isfile(image_path):
            image_path = index.get(os.path.basename(image_name), "")
        if not image_path:
            continue
        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height:
            with Image.open(image_path) as im:
                width, height = im.size
        boxes = []
        for shape in data.get("shapes", []):
            label = str(shape.get("label") or "object")
            points = shape.get("points") or []
            if not points:
                continue
            xs = [float(p[0]) for p in points]
            ys = [float(p[1]) for p in points]
            x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
            y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
            if x2 > x1 and y2 > y1:
                boxes.append((label, [x1, y1, x2 - x1, y2 - y1]))
        entries.append({
            "path": image_path, "name": os.path.basename(image_path),
            "width": width, "height": height, "boxes": boxes,
        })
    return entries


def _collect_csv(root, header, class_names):
    path, _ = _csv_file(root)
    if not path:
        return []
    lower = {h.strip().lower(): h for h in header}
    index = _image_index(root)
    entries = {}

    def col(*names):
        for name in names:
            if name in lower:
                return lower[name]
        return ""

    name_col = col("filename", "file_name", "image", "image_path", "imageid", "image_id", "id")
    if not name_col:
        return []
    cls_col = col("class", "class_name", "label", "category", "labelname", "classid")
    xmin_c, ymin_c = col("xmin", "x_min", "x1"), col("ymin", "y_min", "y1")
    xmax_c, ymax_c = col("xmax", "x_max", "x2"), col("ymax", "y_max", "y2")
    cx_c, cy_c = col("x_center", "cx"), col("y_center", "cy")
    w_c, h_c = col("width", "w"), col("height", "h")
    openimages = bool(lower.get("xmin") and lower.get("labelname"))

    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        for row in csv.DictReader(fh):
            image_name = str(row.get(name_col, "")).strip()
            if not image_name:
                continue
            image_path = index.get(os.path.basename(image_name), "")
            if not image_path:
                candidate = os.path.join(root, image_name)
                if os.path.isfile(candidate):
                    image_path = candidate
            if not image_path:
                continue
            entry = entries.setdefault(image_path, {
                "path": image_path, "name": os.path.basename(image_path),
                "width": None, "height": None, "boxes": [],
            })
            if entry["width"] is None or entry["height"] is None:
                try:
                    with Image.open(image_path) as im:
                        entry["width"], entry["height"] = im.size
                except Exception:
                    continue
            width, height = entry["width"], entry["height"]

            label = str(row.get(cls_col, "object")).strip() or "object"
            try:
                if openimages:
                    x1 = float(row[xmin_c]) * width
                    x2 = float(row[xmax_c]) * width
                    y1 = float(row[ymin_c]) * height
                    y2 = float(row[ymax_c]) * height
                elif xmin_c:
                    x1, y1 = float(row[xmin_c]), float(row[ymin_c])
                    x2, y2 = float(row[xmax_c]), float(row[ymax_c])
                else:
                    cx, cy = float(row[cx_c]), float(row[cy_c])
                    bw, bh = float(row[w_c]), float(row[h_c])
                    if bw <= 1 and bh <= 1:
                        bw, bh = bw * width, bh * height
                    x1, y1 = cx - bw / 2, cy - bh / 2
                    x2, y2 = cx + bw / 2, cy + bh / 2
            except (KeyError, TypeError, ValueError):
                continue
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 > x1 and y2 > y1:
                entry["boxes"].append((label, [x1, y1, x2 - x1, y2 - y1]))
    return list(entries.values())


def _collect_kitti(root):
    img_dir, lbl_dir = _find_kitti(root)
    if not img_dir:
        return []
    entries = []
    for name in sorted(os.listdir(img_dir)):
        if not name.lower().endswith(IMG_EXT):
            continue
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
                    if x2 > x1 and y2 > y1:
                        boxes.append((parts[0], [x1, y1, x2 - x1, y2 - y1]))
        try:
            with Image.open(os.path.join(img_dir, name)) as im:
                width, height = im.size
        except Exception:
            continue
        entries.append({
            "path": os.path.join(img_dir, name), "name": name,
            "width": width, "height": height, "boxes": boxes,
        })
    return entries


def _write_split(out_dir, entries, cat_ids):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    images, annotations = [], []
    ann_id = 1
    used = set()
    for image_id, entry in enumerate(entries, start=1):
        name = entry["name"]
        stem, ext = os.path.splitext(name)
        candidate, idx = name, 1
        while candidate in used:
            candidate = f"{stem}_{idx}{ext}"
            idx += 1
        used.add(candidate)
        _copy(entry["path"], os.path.join(img_dir, candidate))
        images.append({
            "id": image_id, "file_name": candidate,
            "width": entry["width"], "height": entry["height"],
        })
        for label, bbox in entry["boxes"]:
            if label not in cat_ids:
                continue
            annotations.append({
                "id": ann_id, "image_id": image_id,
                "category_id": cat_ids[label],
                "bbox": bbox,
                "area": float(bbox[2]) * float(bbox[3]),
                "iscrowd": 0,
            })
            ann_id += 1
    with open(os.path.join(img_dir, "_annotations.coco.json"), "w",
              encoding="utf-8") as fh:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": [
                {"id": cid, "name": cname, "supercategory": "object"}
                for cname, cid in cat_ids.items()
            ],
        }, fh)
    return len(images)


def convert_extra_to_coco(source_dir, output_dir=None, val_ratio=0.2, seed=42):
    """Convert LabelMe/CSV/KITTI/Supervisely to COCO. Returns stats or None."""
    fmt = detect_extra_format(source_dir)
    if not fmt:
        return None
    out = output_dir or (source_dir.rstrip("/") + "_coco")
    shutil.rmtree(out, ignore_errors=True)

    if fmt == "supervisely":
        try:
            from flashdet.data import convert_supervisely_to_coco

            stats = convert_supervisely_to_coco(source_dir, out)
            return stats
        except Exception:
            return None

    if fmt == "labelme":
        entries = _collect_labelme(source_dir, None)
    elif fmt == "csv":
        _path, header = _csv_file(source_dir)
        entries = _collect_csv(source_dir, header, None)
    else:
        entries = _collect_kitti(source_dir)
    if not entries:
        return None

    classes = sorted({label for entry in entries for label, _ in entry["boxes"]})
    if not classes:
        return None
    cat_ids = {name: i + 1 for i, name in enumerate(classes)}

    rng = random.Random(seed)
    rng.shuffle(entries)
    split_at = max(1, int(len(entries) * (1.0 - val_ratio)))
    train_entries = entries[:split_at]
    val_entries = entries[split_at:] or entries[-1:]
    if not val_entries:
        val_entries = train_entries[-1:]

    train_n = _write_split(os.path.join(out, "train"), train_entries, cat_ids)
    val_n = _write_split(os.path.join(out, "valid"), val_entries, cat_ids)
    return {"train": train_n, "valid": val_n, "classes": classes}
