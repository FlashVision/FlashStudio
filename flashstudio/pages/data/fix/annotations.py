"""Annotation repair — scan and fix COCO / YOLO annotation problems."""

import json
import math
import os
import shutil
from datetime import datetime


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_coco(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("COCO annotation root must be a JSON object")
    return data


def scan_coco(ann_path: str, images_dir: str | None = None, class_names=None) -> dict:
    """Return a structured report of COCO annotation problems."""
    data = load_coco(ann_path)
    images = data.get("images", []) or []
    annotations = data.get("annotations", []) or []
    categories = data.get("categories", []) or []

    image_by_id = {img.get("id"): img for img in images}
    cat_ids = {cat.get("id") for cat in categories}
    report = {
        "format": "COCO",
        "path": ann_path,
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "missing_images": [],
        "missing_image_refs": [],
        "invalid_boxes": [],
        "zero_area": [],
        "out_of_bounds": [],
        "unknown_categories": [],
        "duplicate_boxes": [],
        "bad_categories": [],
    }

    for img in images:
        name = img.get("file_name")
        if images_dir and name and not os.path.isfile(os.path.join(images_dir, name)):
            report["missing_images"].append(name)
        for key in ("width", "height"):
            if not isinstance(img.get(key), (int, float)) or img.get(key) <= 0:
                report["bad_categories"].append(f"image {img.get('id')} has invalid {key}")

    for cat in categories:
        if not str(cat.get("name", "")).strip():
            report["bad_categories"].append(f"category {cat.get('id')} has no name")

    seen = set()
    for ann in annotations:
        iid = ann.get("image_id")
        cid = ann.get("category_id")
        if iid not in image_by_id:
            report["missing_image_refs"].append(ann)
            continue
        if cid not in cat_ids:
            report["unknown_categories"].append(ann)
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(_finite(v) for v in bbox):
            report["invalid_boxes"].append(ann)
            continue
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            report["zero_area"].append(ann)
            continue
        img = image_by_id[iid]
        iw, ih = float(img.get("width", 0) or 0), float(img.get("height", 0) or 0)
        if iw > 0 and ih > 0 and (x < -0.5 or y < -0.5 or x + w > iw + 0.5 or y + h > ih + 0.5):
            report["out_of_bounds"].append(ann)
        key = (iid, cid, round(x, 1), round(y, 1), round(w, 1), round(h, 1))
        if key in seen:
            report["duplicate_boxes"].append(ann)
        seen.add(key)

    report["issues"] = (
        len(report["missing_images"]) + len(report["missing_image_refs"])
        + len(report["invalid_boxes"]) + len(report["zero_area"])
        + len(report["out_of_bounds"]) + len(report["unknown_categories"])
        + len(report["duplicate_boxes"]) + len(report["bad_categories"])
    )
    return report


def fix_coco(ann_path: str, images_dir: str | None = None,
             output_path: str | None = None, backup: bool = True,
             class_names=None) -> dict:
    """Clamp, drop, and remap COCO annotations. Returns a change summary."""
    from PIL import Image

    data = load_coco(ann_path)
    images = data.get("images", []) or []
    annotations = data.get("annotations", []) or []
    categories = data.get("categories", []) or []

    image_by_id = {img.get("id"): img for img in images}

    # Normalise categories to contiguous 1..N ids while preserving names.
    if not categories:
        names = list(class_names or [])
        categories = [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(names)]
        old_to_new = {i: i + 1 for i in range(len(names))}
    else:
        ordered = sorted(categories, key=lambda c: (c.get("id") is None, c.get("id", 0)))
        old_to_new = {}
        cleaned = []
        used_names = {}
        for new_id, cat in enumerate(ordered, 1):
            old_id = cat.get("id")
            old_to_new[old_id] = new_id
            name = str(cat.get("name") or f"class_{new_id}").strip() or f"class_{new_id}"
            used_names[name] = used_names.get(name, 0) + 1
            if used_names[name] > 1:
                name = f"{name}_{used_names[name]}"
            cleaned.append({**cat, "id": new_id, "name": name, "supercategory": cat.get("supercategory", "object")})
        categories = cleaned
    cat_map = old_to_new

    # Repair image dimensions when possible.
    for img in images:
        name = img.get("file_name")
        path = os.path.join(images_dir, name) if images_dir and name else None
        if path and os.path.isfile(path):
            try:
                with Image.open(path) as im:
                    width, height = im.size
                if not img.get("width") or not img.get("height"):
                    img["width"], img["height"] = width, height
            except Exception:
                pass

    kept = []
    removed = clamped = duplicate = 0
    seen = set()
    for ann in annotations:
        iid = ann.get("image_id")
        cid = ann.get("category_id")
        if iid not in image_by_id or cid not in cat_map:
            removed += 1
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(_finite(v) for v in bbox):
            removed += 1
            continue
        img = image_by_id[iid]
        iw = float(img.get("width", 0) or 0)
        ih = float(img.get("height", 0) or 0)
        if iw <= 0 or ih <= 0:
            removed += 1
            continue
        x, y, w, h = [float(v) for v in bbox]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        if x2 - x1 < 0.5 or y2 - y1 < 0.5:
            removed += 1
            continue
        if (x1, y1, x2 - x1, y2 - y1) != (x, y, w, h):
            clamped += 1
        key = (iid, cid, round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1))
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        new_ann = dict(ann)
        new_ann["bbox"] = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
        new_ann["area"] = round((x2 - x1) * (y2 - y1), 2)
        new_ann["category_id"] = cat_map[cid]
        new_ann["iscrowd"] = int(ann.get("iscrowd", 0) or 0)
        kept.append(new_ann)

    # Re-number annotation ids.
    for i, ann in enumerate(kept, 1):
        ann["id"] = i

    data["images"] = images
    data["annotations"] = kept
    data["categories"] = categories

    in_place = output_path is None or os.path.abspath(output_path) == os.path.abspath(ann_path)
    if in_place:
        output_path = ann_path
        if backup:
            shutil.copy2(ann_path, ann_path + ".bak")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {
        "format": "COCO",
        "output": output_path,
        "images": len(images),
        "annotations": len(kept),
        "categories": len(categories),
        "removed": removed,
        "clamped": clamped,
        "duplicates_removed": duplicate,
    }


def _yolo_label_dir(images_dir: str, labels_dir: str | None = None) -> str:
    if labels_dir:
        return labels_dir
    candidate = images_dir.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
    if os.path.isdir(candidate):
        return candidate
    parent = os.path.dirname(os.path.normpath(images_dir))
    for name in ("labels", "labels/train", "labels/valid"):
        p = os.path.join(parent, name)
        if os.path.isdir(p):
            return p
    return images_dir


def scan_yolo(labels_dir: str, images_dir: str | None = None, num_classes: int | None = None) -> dict:
    report = {
        "format": "YOLO",
        "path": labels_dir,
        "files": 0,
        "boxes": 0,
        "invalid_lines": [],
        "out_of_range_classes": [],
        "out_of_bounds": [],
        "missing_labels": [],
        "orphan_labels": [],
    }
    image_names = set()
    if images_dir and os.path.isdir(images_dir):
        image_names = {
            os.path.splitext(f)[0] for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        }
    if not os.path.isdir(labels_dir):
        return report
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".txt"):
            continue
        report["files"] += 1
        stem = os.path.splitext(fn)[0]
        if image_names and stem not in image_names:
            report["orphan_labels"].append(fn)
        path = os.path.join(labels_dir, fn)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, 1):
                parts = line.split()
                if not parts:
                    continue
                report["boxes"] += 1
                if len(parts) != 5:
                    report["invalid_lines"].append(f"{fn}:{lineno}")
                    continue
                try:
                    cid = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    report["invalid_lines"].append(f"{fn}:{lineno}")
                    continue
                if num_classes and (cid < 0 or cid >= num_classes):
                    report["out_of_range_classes"].append(f"{fn}:{lineno}")
                if any(v < 0 or v > 1 for v in vals) or vals[2] <= 0 or vals[3] <= 0:
                    report["out_of_bounds"].append(f"{fn}:{lineno}")
    if image_names:
        label_stems = {
            os.path.splitext(f)[0] for f in os.listdir(labels_dir) if f.endswith(".txt")
        }
        report["missing_labels"] = sorted(image_names - label_stems)
    report["issues"] = sum(
        len(report[k]) for k in (
            "invalid_lines", "out_of_range_classes", "out_of_bounds",
            "missing_labels", "orphan_labels",
        )
    )
    return report


def fix_yolo(labels_dir: str, num_classes: int | None = None, backup: bool = True) -> dict:
    fixed_files = fixed_lines = removed = clamped = 0
    if not os.path.isdir(labels_dir):
        return {"format": "YOLO", "files_fixed": 0, "lines_fixed": 0, "removed": 0, "clamped": 0}
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".txt"):
            continue
        path = os.path.join(labels_dir, fn)
        out_lines = []
        changed = False
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    removed += 1
                    changed = True
                    continue
                try:
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = [float(v) for v in parts[1:]]
                except ValueError:
                    removed += 1
                    changed = True
                    continue
                if num_classes and (cid < 0 or cid >= num_classes):
                    removed += 1
                    changed = True
                    continue
                if bw <= 0 or bh <= 0:
                    removed += 1
                    changed = True
                    continue
                x1, x2 = cx - bw / 2, cx + bw / 2
                y1, y2 = cy - bh / 2, cy + bh / 2
                x1, x2 = max(0.0, x1), min(1.0, x2)
                y1, y2 = max(0.0, y1), min(1.0, y2)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    removed += 1
                    changed = True
                    continue
                ncx, ncy = (x1 + x2) / 2, (y1 + y2) / 2
                nbw, nbh = x2 - x1, y2 - y1
                if (ncx, ncy, nbw, nbh) != (cx, cy, bw, bh):
                    clamped += 1
                    changed = True
                out_lines.append(f"{cid} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}\n")
        if changed:
            if backup:
                shutil.copy2(path, path + ".bak")
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(out_lines)
            fixed_files += 1
            fixed_lines += len(out_lines)
    return {
        "format": "YOLO",
        "files_fixed": fixed_files,
        "lines_fixed": fixed_lines,
        "removed": removed,
        "clamped": clamped,
    }
