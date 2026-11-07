"""Resolve and normalize dataset paths for FlashDet.

FlashDet expects every split directory to contain both the images and an
``_annotations.coco.json`` file::

    dataset_root/
        train/_annotations.coco.json
        train/<images>
        valid/_annotations.coco.json
        valid/<images>

Uploaded datasets are frequently laid out differently.  Common examples
include ``train/images/`` plus ``train/_annotations.coco.json``, or official
COCO annotations in a sibling ``annotations/instances_train2017.json``.  The
helpers here locate the right annotation file and copy/symlink it to the
location FlashDet expects, so Preview, Edit, Fix, Quality, Verify, and the
training launcher all see the same layout.
"""

import json
import os
import shutil

ANNOTATION_NAME = "_annotations.coco.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "valid": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


def _json_files(folder: str) -> list[str]:
    if not folder or not os.path.isdir(folder):
        return []
    try:
        return [
            os.path.join(folder, name)
            for name in sorted(os.listdir(folder))
            if name.lower().endswith(".json")
        ]
    except OSError:
        return []


def split_dir(path: str) -> str:
    """Return the split directory for *path*.

    ``dataset/train/images`` -> ``dataset/train``; ``dataset/train`` stays as
    is.  This lets callers handle both common upload layouts.
    """
    path = os.path.abspath(path)
    if os.path.basename(path).lower() in ("images", "imgs", "jpegimages"):
        return os.path.dirname(path)
    return path


def has_images(folder: str) -> bool:
    """True when *folder* directly contains at least one image file."""
    if not folder or not os.path.isdir(folder):
        return False
    try:
        return any(name.lower().endswith(IMAGE_EXTENSIONS) for name in os.listdir(folder))
    except OSError:
        return False


_COCO_JSON_CACHE: dict = {}


def _is_coco_json(path: str) -> bool:
    try:
        stat = os.stat(path)
        key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return False
    if key in _COCO_JSON_CACHE:
        return _COCO_JSON_CACHE[key]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = (
            isinstance(data, dict)
            and isinstance(data.get("images"), list)
            and isinstance(data.get("annotations"), list)
        )
    except (OSError, ValueError, TypeError):
        result = False
    if len(_COCO_JSON_CACHE) > 128:
        _COCO_JSON_CACHE.clear()
    _COCO_JSON_CACHE[key] = result
    return result


def _annotation_score(path: str, split: str) -> int:
    name = os.path.basename(path).lower()
    score = 0
    if name == ANNOTATION_NAME:
        score += 1000
    if "coco" in name:
        score += 120
    if "annotation" in name:
        score += 80
    if "instance" in name:
        score += 60
    if split:
        tokens = _SPLIT_ALIASES.get(split, (split,))
        if any(token in name for token in tokens):
            score += 40
        elif any(token in name for token in ("train", "val", "valid", "test")):
            # Strongly avoid using the train annotation as the val annotation.
            score -= 100
    return score


def resolve_annotation_file(
    img_dir: str,
    split: str = "",
    hint: str = "",
    root: str = "",
) -> str:
    """Find the most likely COCO annotation file for an image directory.

    Searches the image directory, its split directory, the dataset root, and
    common ``annotations/`` folders.  Returns an empty string when nothing
    COCO-like is found, so callers never use an unrelated ``config.json``.
    """
    if not img_dir or not os.path.isdir(img_dir):
        return ""

    img_dir = os.path.abspath(img_dir)
    split_directory = split_dir(img_dir)
    parent = os.path.dirname(split_directory)

    bases = []
    for base in (img_dir, split_directory, root, parent, os.path.dirname(parent)):
        if not base:
            continue
        base = os.path.abspath(base)
        if base not in bases:
            bases.append(base)

    candidates = []
    if hint and os.path.isfile(hint):
        hint_path = os.path.abspath(hint)
        if any(
            hint_path == base or hint_path.startswith(base + os.sep)
            for base in bases
        ):
            candidates.append(hint_path)
    for base in bases:
        candidates.append(os.path.join(base, ANNOTATION_NAME))
        candidates.extend(_json_files(base))
        candidates.extend(_json_files(os.path.join(base, "annotations")))
        candidates.extend(_json_files(os.path.join(base, "Annotations")))

    seen = set()
    existing = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen or not os.path.isfile(candidate):
            continue
        seen.add(candidate)
        existing.append(candidate)

    if not existing:
        return ""

    # The exact filename FlashDet expects wins when it exists in the split.
    split_exact = os.path.join(split_directory, ANNOTATION_NAME)
    if os.path.isfile(split_exact):
        return split_exact

    scored = sorted(
        existing,
        key=lambda candidate: _annotation_score(candidate, split),
        reverse=True,
    )
    for candidate in scored:
        if _annotation_score(candidate, split) >= 0 and _is_coco_json(candidate):
            return candidate
    return ""


def ensure_coco_annotation(
    img_dir: str,
    split: str = "",
    hint: str = "",
    root: str = "",
) -> str:
    """Ensure ``img_dir/_annotations.coco.json`` exists.

    The source is symlinked when possible and copied otherwise.  An existing
    destination is never overwritten, because the user may have repaired it.
    """
    if not img_dir or not os.path.isdir(img_dir):
        return ""
    src = resolve_annotation_file(img_dir, split=split, hint=hint, root=root)
    if not src:
        return ""

    dst = os.path.join(img_dir, ANNOTATION_NAME)
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst
    if os.path.isfile(dst):
        return dst

    if os.path.lexists(dst):
        try:
            os.unlink(dst)
        except OSError:
            pass

    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        try:
            shutil.copy2(src, dst)
        except OSError:
            return ""
    return dst


def find_validation_dir(train_dir: str, root: str = "") -> str:
    """Locate the validation image directory next to *train_dir*.

    Handles both ``dataset/train`` + ``dataset/valid`` and
    ``dataset/train/images`` + ``dataset/valid/images`` layouts.
    """
    if not train_dir or not os.path.isdir(train_dir):
        return ""

    train_abs = os.path.abspath(train_dir)
    split_directory = split_dir(train_abs)
    base = os.path.abspath(root) if root else os.path.dirname(split_directory)

    split_name = os.path.basename(split_directory).lower()
    aliases = ["valid", "val", "validation", "test"]
    if "train" in split_name:
        aliases = [
            split_name.replace("train", "val", 1),
            split_name.replace("train", "valid", 1),
            split_name.replace("train", "test", 1),
        ] + aliases

    candidates = []
    for search_base in (
        base,
        os.path.dirname(split_directory),
        os.path.dirname(base),
        split_directory,
    ):
        if not search_base:
            continue
        for alias in aliases:
            candidates.append(os.path.join(search_base, alias))
            candidates.append(os.path.join(search_base, alias, "images"))
            candidates.append(os.path.join(search_base, alias, "imgs"))

    seen = set()
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen or candidate == train_abs:
            continue
        seen.add(candidate)
        if os.path.isdir(candidate) and has_images(candidate):
            return candidate
    return ""


def load_coco_classes(path: str) -> list[str]:
    """Read category names from a COCO JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        categories = data.get("categories", []) if isinstance(data, dict) else []
        ordered = sorted(categories, key=lambda c: c.get("id", 0))
        return [str(c["name"]) for c in ordered if c.get("name")]
    except (OSError, ValueError, TypeError, KeyError):
        return []


def summarize_coco_splits(train_dir: str, val_dir: str = "") -> dict:
    """Count images, annotations, and classes for a train/val COCO pair.

    Unlike FlashDet's ``summarize_coco_root`` this does not depend on the
    folders being named ``train``/``valid``, so it also handles
    ``train2017``/``val2017`` and nested ``images`` layouts.
    """
    from collections import Counter

    summary = {
        "train_images": 0,
        "val_images": 0,
        "annotations": 0,
        "class_names": [],
        "distribution": {},
    }
    id_to_name = {}
    distribution = Counter()

    for split, folder, key in (("train", train_dir, "train_images"),
                               ("val", val_dir, "val_images")):
        if not folder or not os.path.isdir(folder):
            continue
        annotation = resolve_annotation_file(folder, split=split)
        if not annotation:
            continue
        try:
            with open(annotation, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError, TypeError):
            continue
        summary[key] = len(data.get("images", []))
        summary["annotations"] += len(data.get("annotations", []))
        for category in data.get("categories", []):
            try:
                id_to_name[int(category["id"])] = str(
                    category.get("name") or f"class_{category['id']}"
                )
            except (KeyError, ValueError, TypeError):
                continue
        for annotation_record in data.get("annotations", []):
            try:
                distribution[int(annotation_record["category_id"])] += 1
            except (KeyError, ValueError, TypeError):
                continue

    if id_to_name:
        summary["class_names"] = [id_to_name[k] for k in sorted(id_to_name)]
    summary["distribution"] = {
        id_to_name.get(category_id, str(category_id)): count
        for category_id, count in sorted(distribution.items())
    }
    return summary


def prepare_annotations(
    train_dir: str,
    val_dir: str = "",
    root: str = "",
    train_hint: str = "",
    val_hint: str = "",
) -> dict:
    """Normalize annotations for a train/val pair.

    Returns a dict with ``train_images``, ``val_images``, ``train_annotation``,
    ``val_annotation``, and ``dataset_root``.
    """
    if not train_dir or not os.path.isdir(train_dir):
        return {
            "train_images": "",
            "val_images": "",
            "train_annotation": "",
            "val_annotation": "",
            "dataset_root": "",
        }

    if not val_dir or not os.path.isdir(val_dir):
        val_dir = find_validation_dir(train_dir, root=root)

    train_ann = ensure_coco_annotation(
        train_dir, split="train", hint=train_hint, root=root
    )
    val_ann = (
        ensure_coco_annotation(val_dir, split="val", hint=val_hint, root=root)
        if val_dir and os.path.isdir(val_dir)
        else ""
    )

    if not root:
        root = os.path.dirname(split_dir(train_dir))

    return {
        "train_images": train_dir,
        "val_images": val_dir,
        "train_annotation": train_ann,
        "val_annotation": val_ann,
        "dataset_root": root,
    }
