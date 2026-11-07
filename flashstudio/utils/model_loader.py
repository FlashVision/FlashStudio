"""Model runtimes for inference — FlashDet, Ultralytics, ONNX, TorchScript.

All adapters accept a **BGR** numpy image and return a list of
``(bbox_xyxy, score, class_id)`` tuples, matching the FlashDet predictor API.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np


class BaseAdapter:
    class_names: List[str]
    num_classes: int
    input_size: int
    framework: str = ""

    def predict(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, float, int]]:
        raise NotImplementedError

    def __call__(self, image_bgr: np.ndarray):
        return self.predict(image_bgr)


def _resolve_names(names, fallback=None) -> List[str]:
    if isinstance(names, dict):
        return [str(v) for _k, v in sorted(names.items())]
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    if fallback:
        return [str(n) for n in fallback]
    return []


class FlashDetAdapter(BaseAdapter):
    framework = "FlashDet"

    def __init__(self, model_path, device="cpu", conf=0.25, nms=0.45,
                 input_size=640, class_names=None):
        from flashdet import Predictor
        self.input_size = int(input_size)
        self._predictor = Predictor(
            model_path=model_path,
            device=str(device).split(" ")[0],
            conf_thresh=float(conf),
            nms_thresh=float(nms),
            input_size=self.input_size,
            class_names=list(class_names) if class_names else None,
        )
        self.class_names = _resolve_names(self._predictor.class_names, class_names)
        self.num_classes = int(getattr(self._predictor, "num_classes", len(self.class_names)) or len(self.class_names))

    def predict(self, image_bgr):
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        return self._predictor(rgb)


def _patch_torchvision_nms():
    """Use a portable CPU NMS when torchvision's CPU kernel is unavailable."""
    try:
        import torch
        import torchvision.ops as ops
        try:
            ops.nms(torch.zeros((0, 4)), torch.zeros((0,)), 0.5)
            return
        except Exception:
            pass

        def _nms(boxes, scores, iou_threshold):
            if boxes.numel() == 0:
                return torch.empty((0,), dtype=torch.long, device=boxes.device)
            x1, y1, x2, y2 = boxes.unbind(1)
            areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            order = scores.argsort(descending=True)
            keep = []
            while order.numel():
                i = order[0]
                keep.append(i)
                if order.numel() == 1:
                    break
                rest = order[1:]
                xx1 = torch.maximum(x1[i], x1[rest])
                yy1 = torch.maximum(y1[i], y1[rest])
                xx2 = torch.minimum(x2[i], x2[rest])
                yy2 = torch.minimum(y2[i], y2[rest])
                inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
                iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)
                order = rest[iou <= iou_threshold]
            return torch.stack(keep)

        ops.nms = _nms
    except Exception:
        pass


class UltralyticsAdapter(BaseAdapter):
    framework = "Ultralytics"

    def __init__(self, model_path, device="cpu", conf=0.25, nms=0.45,
                 input_size=640, class_names=None):
        _patch_torchvision_nms()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install with `pip install ultralytics`."
            ) from exc
        self.input_size = int(input_size)
        self.model = YOLO(model_path)
        names = _resolve_names(getattr(self.model, "names", None), class_names)
        self.class_names = names or (list(class_names) if class_names else [])
        self.num_classes = len(self.class_names) or 80
        self.device = str(device).split(" ")[0]
        self.conf = float(conf)
        self.nms = float(nms)

    def predict(self, image_bgr):
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.nms,
            imgsz=self.input_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        return [(xyxy[i], float(conf[i]), int(cls[i])) for i in range(len(xyxy))]


class _OnnxLikeAdapter(BaseAdapter):
    """Decode common YOLO-style ONNX/TorchScript outputs."""

    def _decode(self, raw, scale: float, pad_x: float, pad_y: float):
        arr = np.asarray(raw, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise RuntimeError(f"Unsupported model output shape: {arr.shape}")
        if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 256 and arr.shape[1] > 256:
            arr = arr.T

        nc = self.num_classes if self.num_classes > 0 else max(arr.shape[1] - 5, 1)
        cols = arr.shape[1]
        boxes = None
        scores = None
        classes = None

        if cols == 5 + nc:
            boxes = arr[:, :4]
            obj = arr[:, 4]
            cls_scores = arr[:, 5:5 + nc]
            classes = cls_scores.argmax(axis=1)
            scores = obj * cls_scores.max(axis=1)
        elif cols == 4 + nc:
            boxes = arr[:, :4]
            cls_scores = arr[:, 4:4 + nc]
            classes = cls_scores.argmax(axis=1)
            scores = cls_scores.max(axis=1)
        elif cols == 6:
            boxes = arr[:, :4]
            scores = arr[:, 4]
            classes = arr[:, 5].astype(int)
        else:
            raise RuntimeError(
                f"Unsupported model output shape {arr.shape}; "
                f"expected 4+classes, 5+classes, or 6 columns."
            )

        if boxes is None:
            return []
        boxes = boxes.astype(np.float32)
        # YOLO-style outputs are cx, cy, w, h; the 6-column form is xyxy.
        if cols != 6:
            cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / max(scale, 1e-6)
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / max(scale, 1e-6)

        keep = scores >= self.conf
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        if len(boxes) == 0:
            return []

        # Per-class NMS.
        out = []
        for cid in np.unique(classes):
            idx = np.where(classes == cid)[0]
            keep_idx = _nms(boxes[idx], scores[idx], self.nms)
            for j in keep_idx:
                out.append((boxes[idx][j], float(scores[idx][j]), int(cid)))
        out.sort(key=lambda item: item[1], reverse=True)
        return out


class OnnxAdapter(_OnnxLikeAdapter):
    framework = "ONNX Runtime"

    def __init__(self, model_path, device="cpu", conf=0.25, nms=0.45,
                 input_size=640, class_names=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime is not installed. Install with `pip install onnxruntime`."
            ) from exc
        self.class_names = list(class_names) if class_names else []
        self.num_classes = len(self.class_names) or 80
        self.conf = float(conf)
        self.nms = float(nms)
        self.input_size = int(input_size)
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        shape = inp.shape
        if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_size = int(shape[2])
        self.input_name = inp.name
        self.nhwc = len(shape) == 4 and isinstance(shape[-1], int) and shape[-1] == 3

    def predict(self, image_bgr):
        import cv2
        h, w = image_bgr.shape[:2]
        size = self.input_size
        scale = min(size / max(w, 1), size / max(h, 1))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(image_bgr, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[None] if not self.nhwc else rgb[None]
        raw = self.session.run(None, {self.input_name: blob.astype(np.float32)})[0]
        # Standard YOLO ONNX exports map boxes by the resize gain only; the
        # padding offset is already baked into the exported predictions.
        return self._decode(raw, scale, 0.0, 0.0)


class TorchScriptAdapter(_OnnxLikeAdapter):
    framework = "TorchScript"

    def __init__(self, model_path, device="cpu", conf=0.25, nms=0.45,
                 input_size=640, class_names=None):
        import torch
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device).eval()
        self.class_names = list(class_names) if class_names else []
        self.num_classes = len(self.class_names) or 80
        self.conf = float(conf)
        self.nms = float(nms)
        self.input_size = int(input_size)

    def predict(self, image_bgr):
        import cv2
        h, w = image_bgr.shape[:2]
        size = self.input_size
        scale = min(size / max(w, 1), size / max(h, 1))
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        resized = cv2.resize(image_bgr, (nw, nh))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = self.torch.from_numpy(rgb.transpose(2, 0, 1)[None]).to(self.device)
        with self.torch.no_grad():
            out = self.model(tensor)
        if isinstance(out, dict):
            out = out.get("preds", next(iter(out.values()), None))
        if isinstance(out, (list, tuple)):
            out = out[0]
        if out is None:
            return []
        raw = out.detach().cpu().numpy()
        return self._decode(raw, scale, 0.0, 0.0)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def detect_framework(model_path: str) -> str:
    ext = os.path.splitext(str(model_path))[1].lower()
    if ext == ".onnx":
        try:
            __import__("ultralytics")
            return "Ultralytics"
        except Exception:
            return "ONNX Runtime"
    if ext == ".engine":
        return "TensorRT"
    if ext == ".pth":
        return "FlashDet"
    if ext == ".pt":
        try:
            import torch
            try:
                torch.jit.load(model_path, map_location="cpu")
                return "TorchScript"
            except Exception:
                pass
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and ("model_state_dict" in ckpt or "config" in ckpt):
                return "FlashDet"
            if isinstance(ckpt, dict) and "model" in ckpt and "names" in ckpt:
                return "Ultralytics"
        except Exception:
            pass
        return "Ultralytics"
    return "Auto"


def available_frameworks() -> List[dict]:
    out = []
    for name, label, module in (
        ("FlashDet", "FlashDet", "flashdet"),
        ("Ultralytics", "Ultralytics", "ultralytics"),
        ("ONNX Runtime", "ONNX Runtime", "onnxruntime"),
        ("TorchScript", "TorchScript", "torch"),
        ("TensorRT", "TensorRT", "tensorrt"),
    ):
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "")
            out.append({"id": name, "label": label, "installed": True, "version": str(version)})
        except Exception as exc:
            out.append({"id": name, "label": label, "installed": False, "version": str(exc)})
    return out


def load_predictor(model_path: str, framework: str = "Auto", device="cpu",
                   conf=0.25, nms=0.45, input_size=640, class_names=None):
    framework = framework or "Auto"
    if framework == "Auto":
        framework = detect_framework(model_path)
    common = dict(device=device, conf=conf, nms=nms,
                  input_size=input_size, class_names=class_names)
    if framework == "TensorRT":
        try:
            import tensorrt  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "TensorRT engines need the TensorRT runtime. Install `tensorrt` "
                "or export an ONNX model and use the ONNX Runtime."
            ) from exc
        return UltralyticsAdapter(model_path, **common)
    if framework == "FlashDet":
        return FlashDetAdapter(model_path, **common)
    if framework == "Ultralytics":
        return UltralyticsAdapter(model_path, **common)
    if framework == "ONNX Runtime":
        return OnnxAdapter(model_path, **common)
    if framework == "TorchScript":
        return TorchScriptAdapter(model_path, **common)
    raise RuntimeError(f"Unknown inference framework: {framework}")
