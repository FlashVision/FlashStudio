# FlashStudio

**Interactive Training & Inference UI for FlashDet**

---

## Install

```bash
pip install flashstudio
pip install git+https://github.com/FlashVision/FlashDet.git
```

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
!pip install git+https://github.com/FlashVision/FlashDet.git

from flashstudio import launch
launch(ngrok_token="YOUR_TOKEN")
```

Get a free ngrok token at: https://dashboard.ngrok.com/get-started/your-authtoken

## Features

- **Training** — Launch and monitor FlashDet training with live loss curves
- **Inference** — Run detection on images/video with 17 built-in solutions
- **Model Config** — FlashDet (Pico → X) + YOLOv8/v9/v10/v11/YOLOX
- **Data** — Upload datasets or download from FlashDet registry
- **Export** — ONNX export with FP16 support
- **Dashboard** — Project overview with recent runs and metrics

## Supported Models

| Model | Params | Use Case |
|-------|--------|----------|
| FlashDet-Pico | ~298K | Edge / MCU |
| FlashDet-Nano | ~790K | Embedded / IoT |
| FlashDet-Small | ~1.8M | General purpose |
| FlashDet-Medium | ~3.6M | High accuracy |
| FlashDet-Large | ~5.8M | High accuracy |
| FlashDet-X | ~9.0M | Server / Max accuracy |

## Requirements

- Python >= 3.10
- FlashDet (for training/inference)
- ngrok account (free, for Colab)
- GPU recommended for training

## Links

- GitHub: https://github.com/FlashVision/FlashStudio
- FlashDet: https://github.com/FlashVision/FlashDet
- Organization: https://flashvision.github.io

## License

Apache-2.0
