# FlashStudio

**A visual studio for training, fine-tuning, and deploying detection models — without the script spaghetti.**

FlashStudio is an interactive Streamlit workspace built on top of [FlashDet](https://github.com/FlashVision/FlashDet). It gives you one place to manage datasets, configure models, run training with live feedback, optimize and export weights, and try out inference on images, video, or live streams.

## Install

```bash
pip install flashstudio
```

That's it — FlashDet and the core dependencies come along automatically, so there's no repository checkout or manual setup required.

## Quick Start

Launch the app from your terminal:

```bash
flashstudio
```

Or from Python:

```python
from flashstudio import launch
launch()
```

Then open **http://localhost:8501** in your browser.

## Google Colab

FlashStudio works in Colab with no tokens, tunnels, or signups:

```python
!pip install -q flashstudio

from flashstudio import launch
launch()
```

Colab is detected automatically and the built-in proxy gives you a clickable, working URL. If you'd rather use a stable public address, pass an ngrok token instead:

```python
launch(ngrok_token="YOUR_TOKEN")
```

Grab a free token from the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).

## What You Get

- **Guided workflow** — move from Dashboard → Data → Model → Training → Export → Inference, with pipeline readiness checks and hints about what to do next.
- **Project workspaces** — each project keeps its own datasets, runs, checkpoints, and saved UI state. Switch projects without losing your place.
- **Data tools** — upload ZIP/TAR archives or pull datasets from the FlashDet registry, preview annotations, verify splits, repair broken COCO/YOLO labels, edit boxes by hand, and run a quality report.
- **Framework-first model setup** — FlashDet (Pico through X, including the Pico-Zoo backbones), Ultralytics YOLOv8/v9/v10/v11, and YOLOX for inference and export.
- **Training cockpit** — preflight checks, launch/pause/resume/stop, live loss curves, epoch visualizations, ground-truth verification, logs, checkpoints, and cross-run comparison.
- **Weight Hub** — download FlashDet registry and Ultralytics checkpoints directly from the training or inference pages.
- **Export & optimize** — ONNX and TorchScript export, TensorRT when available, layer/kernel fusion, ONNX parity checks, INT8 quantization, and FPS/latency benchmarks.
- **Inference** — FlashDet, Ultralytics, ONNX Runtime, TorchScript, and TensorRT engines on images, video, or RTSP/webcam streams.
- **Live inference** — real-time FPS and detections, MP4 capture, and a multi-camera grid.
- **Zones & solutions** — polygon, rectangle, and line zones with 16 built-in solutions.
- **Active learning** — export low-confidence detections for review and retraining.
- **Dependency hints** — in-app guidance (and one-click installs) for optional runtimes and frameworks.

## Supported Models

| Model | Params | Best For |
|-------|--------|----------|
| FlashDet-Pico | ~298K | Edge / MCU |
| FlashDet-Nano | ~790K | Embedded / IoT |
| FlashDet-Small | ~1.8M | General purpose |
| FlashDet-Medium | ~3.6M | High accuracy |
| FlashDet-Large | ~5.8M | High accuracy |
| FlashDet-X | ~9.0M | Server / max accuracy |

**Training backends:** FlashDet Trainer (default) or Ultralytics (YOLOv8/v9/v10/v11).

**Inference backends:** FlashDet `.pth`, Ultralytics `.pt`, ONNX Runtime, TorchScript, and TensorRT `.engine`.

## Requirements

- Python 3.10 or newer
- FlashDet (installed automatically)
- Ultralytics + ONNX Runtime for external YOLO/ONNX models
- TensorRT for `.engine` inference (optional)
- An ngrok account only if you want a stable public Colab URL (optional)
- A GPU is strongly recommended for training

## Links

- GitHub: https://github.com/FlashVision/FlashStudio
- FlashDet: https://github.com/FlashVision/FlashDet
- Organization: https://flashvision.github.io

## License

Apache 2.0 — see [LICENSE](https://github.com/FlashVision/FlashStudio/blob/main/LICENSE).
