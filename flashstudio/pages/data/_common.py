"""Shared helpers and constants for the data page."""

import os
import json
import streamlit as st

from flashstudio.constants import (
    DEFAULT_DATA_DIR, NETWORK_CHECK_URL, NETWORK_CHECK_TIMEOUT,
)

try:
    from flashdet import download_dataset, list_datasets  # noqa: F401
    from flashdet.data import (  # noqa: F401
        convert_dataset, detect_dataset_format,
        verify_dataset, summarize_coco_root,
    )
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
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(NETWORK_CHECK_URL, timeout=NETWORK_CHECK_TIMEOUT)
        return True
    except (urllib.error.URLError, OSError):
        return False


def _dataset_already_downloaded(dataset_id: str, output_dir: str | None = None) -> str | None:
    if output_dir is None:
        output_dir = os.path.join(DEFAULT_DATA_DIR, dataset_id)
    if not os.path.isdir(output_dir):
        return None
    t = os.path.join(output_dir, "train")
    v = os.path.join(output_dir, "valid")
    if not os.path.isdir(v):
        v = os.path.join(output_dir, "val")
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
