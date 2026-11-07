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
launch(ngrok_token="YOUR_TOKEN")
```

Get a free ngrok token at: https://dashboard.ngrok.com/get-started/your-authtoken

## Features

- **Training** — FlashDet or Ultralytics backends with live loss curves and run comparison
- **Framework-specific Model Config** — FlashDet Pico-Zoo backbones, Ultralytics YOLO, YOLOX export
- **Weight Hub** — download FlashDet registry and Ultralytics checkpoints from training or inference
- **Inference** — FlashDet, Ultralytics, ONNX Runtime, TorchScript, and TensorRT models
- **Live Inference** — RTSP/webcam streaming, multi-camera grid, live FPS and detections
- **Data** — Upload, download, preview, repair annotations, edit boxes, and run quality reports
- **Optimize** — layer/kernel fusion, ONNX parity, INT8 quantization, and FPS benchmarking
- **Model Zoo** — One-click pretrained YOLO checkpoints
- **Active Learning** — Export low-confidence detections for review
- **Export** — ONNX and TorchScript export for FlashDet and Ultralytics weights
- **Dashboard** — Project overview with recent runs, workspace, and environment details

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
- TensorRT for `.engine` inference
- ngrok account (free, for Colab)
- GPU recommended for training

## Links

- GitHub: https://github.com/FlashVision/FlashStudio
- FlashDet: https://github.com/FlashVision/FlashDet
- Organization: https://flashvision.github.io

## License

Apache-2.0
