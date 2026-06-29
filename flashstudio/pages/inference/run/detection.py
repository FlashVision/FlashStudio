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
)
from flashstudio.pages.inference._common import _get_class_names, _HAS_PREDICTOR

try:
    from flashdet import Predictor
except ImportError:
    pass


def _detect(image):
    weights = st.session_state.get("infer_weights_path", "")
    if not weights:
        cached = st.session_state.get("_infer_weights_tmp_path", "")
        if cached and os.path.isfile(cached):
            weights = cached
        else:
            up = st.session_state.get("infer_weights_file")
            if up:
                tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
                tmp.write(up.read()); tmp.flush(); up.seek(0)
                weights = tmp.name
                st.session_state["_infer_weights_tmp_path"] = weights
    if weights:
        try:
            return _detect_real(image, weights)
        except Exception:
            pass
    return _detect_demo(image)


def _detect_real(image, weights_path):
    if not _HAS_PREDICTOR:
        raise ImportError("Predictor unavailable")
    custom_cls = _get_class_names()
    predictor = Predictor(model_path=weights_path, device=st.session_state.get("infer_device", "cpu").split(" ")[0],
                          conf_thresh=st.session_state.get("infer_conf", INFER_CONF_THRESHOLD),
                          nms_thresh=st.session_state.get("infer_nms", INFER_NMS_THRESHOLD),
                          input_size=st.session_state.get("infer_img_size", INFER_IMG_SIZE),
                          class_names=custom_cls if custom_cls != COCO_CLASSES else None)
    raw = predictor(np.array(image))
    class_filter = st.session_state.get("infer_class_filter", [])
    label_list = custom_cls or COCO_CLASSES
    dets = []
    for bbox, score, cls_id in raw:
        name = label_list[int(cls_id)] if int(cls_id) < len(label_list) else f"class_{cls_id}"
        if class_filter and name not in class_filter:
            continue
        dets.append([name, f"{float(score):.2f}", str([int(x) for x in (bbox.tolist() if hasattr(bbox, 'tolist') else list(bbox))])])
    return dets


def _detect_demo(image):
    w, h = image.size
    rng = np.random.default_rng(hash(image.tobytes()[:100]) % (2 ** 31))
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


def _draw_boxes_cv2(frame, raw_results):
    """Draw bounding boxes on a BGR cv2 frame from Predictor output [(bbox, score, cls_id), ...]."""
    import cv2
    label_list = _get_class_names() or COCO_CLASSES
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


def _to_coco_format(results):
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": n} for i, n in enumerate(COCO_CLASSES)]}
    aid = 1
    for iid, r in enumerate(results, 1):
        coco["images"].append({"id": iid, "file_name": r["name"], "width": r.get("width", INFER_DEFAULT_RESOLUTION[0]), "height": r.get("height", INFER_DEFAULT_RESOLUTION[1])})
        for cls, conf, bbox_str in r["dets"]:
            bbox = json.loads(bbox_str.replace("'", '"'))
            coco["annotations"].append({"id": aid, "image_id": iid,
                                         "category_id": COCO_CLASSES.index(cls) if cls in COCO_CLASSES else 0,
                                         "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                                         "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), "score": float(conf), "iscrowd": 0})
            aid += 1
    return coco
