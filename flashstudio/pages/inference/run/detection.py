"""Inference — Detection utilities."""

import os
import json
import tempfile
import streamlit as st
import numpy as np
from PIL import ImageDraw, ImageFont
from flashstudio.constants import (
    COCO_CLASSES, BBOX_COLORS_HEX, BBOX_COLORS_RGB,
    INFER_CONF_THRESHOLD, INFER_NMS_THRESHOLD, INFER_IMG_SIZE,
    INFER_DEFAULT_RESOLUTION, FONT_PATH_LINUX, FONT_SIZE_DEFAULT,
    DEFAULT_SAVE_DIR,
)
from flashstudio.pages.inference._common import _get_class_names, _HAS_PREDICTOR
from flashstudio.utils import get_state
from flashstudio.utils.model_loader import load_predictor

try:
    from flashdet import Predictor
except ImportError:
    pass


def _get_adapter(weights_path: str):
    """Load and cache the selected runtime adapter for the current weights."""
    framework = st.session_state.get("infer_framework", "Auto")
    family = st.session_state.get("infer_family", "FlashDet")
    key = (
        weights_path, framework, family,
        st.session_state.get("infer_conf", INFER_CONF_THRESHOLD),
        st.session_state.get("infer_nms", INFER_NMS_THRESHOLD),
        st.session_state.get("infer_img_size", INFER_IMG_SIZE),
        st.session_state.get("infer_device", "cpu"),
        tuple(_get_class_names() or []),
    )
    if st.session_state.get("_infer_adapter_key") == key:
        adapter = st.session_state.get("_infer_adapter")
        if adapter is not None:
            return adapter
    adapter = load_predictor(
        weights_path,
        framework=framework,
        device=st.session_state.get("infer_device", "cpu").split(" ")[0],
        conf=st.session_state.get("infer_conf", INFER_CONF_THRESHOLD),
        nms=st.session_state.get("infer_nms", INFER_NMS_THRESHOLD),
        input_size=st.session_state.get("infer_img_size", INFER_IMG_SIZE),
        class_names=_get_class_names(),
    )
    st.session_state["_infer_adapter"] = adapter
    st.session_state["_infer_adapter_key"] = key
    return adapter



def _resolve_weights_path() -> str:
    """Resolve the weight path according to the selected source.

    ``Upload`` deliberately ignores ``infer_weights_path`` so a stale
    "Training output" path can never shadow a freshly uploaded checkpoint.
    """
    source = st.session_state.get("weight_source")
    if source is None:
        from flashstudio.pages.inference._common import latest_inference_weight

        family = st.session_state.get("infer_family", "FlashDet")
        save_dir = get_state("save_dir") or DEFAULT_SAVE_DIR
        source = (
            "Training output"
            if latest_inference_weight(family, save_dir)
            else "Upload"
        )

    if source == "Upload":
        up = st.session_state.get("infer_weights_file")
        if not up:
            return ""
        key = (
            getattr(up, "name", ""),
            getattr(up, "size", 0),
            getattr(up, "file_id", ""),
        )
        cached = st.session_state.get("_infer_weights_tmp_path", "")
        if cached and os.path.isfile(cached) and st.session_state.get("_infer_weights_tmp_key") == key:
            return cached
        tmp = tempfile.NamedTemporaryFile(
            suffix=os.path.splitext(getattr(up, "name", ""))[1] or ".pth",
            delete=False,
        )
        tmp.write(up.read())
        tmp.flush()
        tmp.close()
        up.seek(0)
        st.session_state["_infer_weights_tmp_path"] = tmp.name
        st.session_state["_infer_weights_tmp_key"] = key
        return tmp.name

    path = str(st.session_state.get("infer_weights_path", "") or "")
    if source == "Path":
        path = os.path.abspath(path) if path else ""
        return path if os.path.isfile(path) else ""

    # "Training output"
    if path and os.path.isfile(path):
        return path
    from flashstudio.pages.inference._common import latest_inference_weight

    family = st.session_state.get("infer_family", "FlashDet")
    save_dir = get_state("save_dir") or DEFAULT_SAVE_DIR
    candidate = latest_inference_weight(family, save_dir)
    return candidate if candidate and os.path.isfile(candidate) else ""


def _detect(image):
    weights = _resolve_weights_path()
    if weights:
        try:
            return _detect_real(image, weights)
        except Exception as e:
            from flashstudio.utils import flash
            flash(f"Model error: {e}", "error")
            if st.session_state.get("infer_demo_mode"):
                return _detect_demo(image)
            return []

    if st.session_state.get("infer_demo_mode"):
        return _detect_demo(image)

    from flashstudio.utils import flash
    flash(
        "No model selected. Upload a checkpoint or enable Demo mode in Inference → Model.",
        "error",
    )
    return []


def _detect_real(image, weights_path):
    adapter = _get_adapter(weights_path)
    bgr = np.array(image.convert("RGB"))[:, :, ::-1]
    raw = adapter.predict(np.ascontiguousarray(bgr))
    class_filter = st.session_state.get("infer_class_filter", [])
    label_list = list(getattr(adapter, "class_names", None) or _get_class_names() or COCO_CLASSES)
    dets = []
    for bbox, score, cls_id in raw:
        cid = int(cls_id)
        name = label_list[cid] if cid < len(label_list) else f"class_{cid}"
        if class_filter and name not in class_filter:
            continue
        dets.append([
            name,
            f"{float(score):.2f}",
            str([int(x) for x in (bbox.tolist() if hasattr(bbox, "tolist") else list(bbox))]),
        ])
    return dets


def _detect_demo(image):
    import hashlib
    w, h = image.size
    digest = hashlib.md5(image.tobytes()[:100]).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:4], "big") % (2 ** 31))
    conf_thr = st.session_state.get("infer_conf", INFER_CONF_THRESHOLD)
    class_filter = st.session_state.get("infer_class_filter", [])
    dets = []
    for _ in range(rng.integers(3, 8)):
        cid = rng.integers(0, 10)
        name = COCO_CLASSES[cid]
        if class_filter and name not in class_filter:
            continue
        conf = rng.uniform(0.3, 0.98)
        if conf < conf_thr:
            continue
        x1 = int(rng.integers(0, max(w - 100, 1)))
        y1 = int(rng.integers(0, max(h - 100, 1)))
        dets.append([name, f"{conf:.2f}", str([x1, y1, min(x1 + int(rng.integers(60, 200)), w), min(y1 + int(rng.integers(60, 200)), h)])])
    return sorted(dets, key=lambda x: float(x[1]), reverse=True)


def _draw_boxes_cv2(frame, raw_results, class_names=None):
    """Draw bounding boxes on a BGR cv2 frame from Predictor output [(bbox, score, cls_id), ...]."""
    import cv2
    label_list = list(class_names or _get_class_names() or COCO_CLASSES)
    for bbox, score, cls_id in raw_results:
        x1, y1, x2, y2 = [int(v) for v in (bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox))]
        cid = int(cls_id)
        name = label_list[cid] if cid < len(label_list) else f"cls_{cid}"
        label = f"{name} {float(score):.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), BBOX_COLORS_RGB[cid % len(BBOX_COLORS_RGB)], 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), BBOX_COLORS_RGB[cid % len(BBOX_COLORS_RGB)], -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def _draw_boxes(image, dets):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(FONT_PATH_LINUX, FONT_SIZE_DEFAULT)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for i, (cls, conf, bbox_str) in enumerate(dets):
        bbox = json.loads(bbox_str.replace("'", '"'))
        color = BBOX_COLORS_HEX[i % len(BBOX_COLORS_HEX)]
        draw.rectangle(bbox, outline=color, width=3)
        label = f"{cls} {conf}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(bbox[1] - th - 6, 0)
        draw.rectangle([bbox[0], ly, bbox[0] + tw + 8, ly + th + 6], fill=color)
        draw.text((bbox[0] + 4, ly + 2), label, fill="white", font=font)
    return image


def _to_coco_format(results, class_names=None):
    names = list(class_names or COCO_CLASSES)
    # Preserve detections whose class is not in the configured list by adding
    # them as categories instead of writing an invalid category_id 0.
    for result in results:
        for cls, _conf, _bbox in result.get("dets", []):
            if cls not in names:
                names.append(cls)
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)],
    }
    aid = 1
    for iid, r in enumerate(results, 1):
        coco["images"].append({
            "id": iid, "file_name": r["name"],
            "width": r.get("width", INFER_DEFAULT_RESOLUTION[0]),
            "height": r.get("height", INFER_DEFAULT_RESOLUTION[1]),
        })
        for cls, conf, bbox_str in r["dets"]:
            bbox = json.loads(bbox_str.replace("'", '"'))
            if cls not in names:
                continue
            category_id = names.index(cls) + 1
            coco["annotations"].append({
                "id": aid, "image_id": iid,
                "category_id": category_id,
                "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                "score": float(conf), "iscrowd": 0,
            })
            aid += 1
    return coco
