# FlashStudio

**Interactive Training & Inference UI for FlashDet**

---

## Install

```bash
pip install flashstudio
```

FlashStudio installs FlashDet automatically and uses its `Trainer` API for training, so no repository checkout is required.

## Quick Start

```bash
# Launch locally
flashstudio

# Or with Python
from flashstudio import launch
launch()
```

## Google Colab

```python
!pip install flashstudio

from flashstudio import launch
launch()
```

FlashStudio detects Colab and uses the built-in proxy — no token, ngrok, or signup required. Pass `ngrok_token="YOUR_TOKEN"` only if you want a stable public URL.

Get a free ngrok token at: https://dashboard.ngrok.com/get-started/your-authtoken

## Features

- **Guided Workflow** — Dashboard → Data → Model → Training → Export → Inference with pipeline readiness and next-step hints
- **Project Workspaces** — isolated datasets, runs, checkpoints, and saved UI state per project
- **Data** — Upload, download, preview with annotation overlays, verify, repair, edit boxes, and run quality reports
- **Framework-first Model Config** — FlashDet Pico-Zoo backbones, Ultralytics YOLO, YOLOX export
- **Training** — FlashDet or Ultralytics backends with preflight checks, live loss curves, visualizations, logs, and run comparison
- **Weight Hub** — download FlashDet registry and Ultralytics checkpoints from training or inference
- **Export** — ONNX and TorchScript export for FlashDet and Ultralytics weights, TensorRT where available
- **Optimize** — layer/kernel fusion, ONNX parity, INT8 quantization, and FPS/latency benchmarking
- **Inference** — FlashDet, Ultralytics, ONNX Runtime, TorchScript, and TensorRT models
- **Live Inference** — RTSP/webcam streaming, multi-camera grid, live FPS and detections
- **Zones & Solutions** — polygon, rectangle, and line zones with 16 built-in solutions
- **Active Learning** — Export low-confidence detections for review
- **Dependency Hints** — in-app guidance and one-click installs for optional runtimes

## Supported Models

| Model | Params | Use Case |
|-------|--------|----------|
| FlashDet-Pico | ~298K | Edge / MCU |
| FlashDet-Nano | ~790K | Embedded / IoT |
| FlashDet-Small | ~1.8M | General purpose |
| FlashDet-Medium | ~3.6M | High accuracy |
| FlashDet-Large | ~5.8M | High accuracy |
| FlashDet-X | ~9.0M | Server / Max accuracy |

### Training and inference backends

- **Training** — FlashDet Trainer (default) or Ultralytics (YOLOv8/v9/v10/v11)
- **Inference** — FlashDet `.pth`, Ultralytics `.pt`, ONNX Runtime, TorchScript, TensorRT `.engine`

## Requirements

- Python >= 3.10
- FlashDet (for FlashDet training/inference)
- Ultralytics + ONNX Runtime for external YOLO/ONNX models
- TensorRT for `.engine` inference (optional)
- ngrok account (optional, for a stable Colab URL)
- GPU recommended for training

## Links

- GitHub: https://github.com/FlashVision/FlashStudio
- FlashDet: https://github.com/FlashVision/FlashDet
- Organization: https://flashvision.github.io

## License

Apache-2.0
