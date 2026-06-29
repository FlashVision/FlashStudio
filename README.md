# FlashStudio

[![PyPI](https://img.shields.io/pypi/v/flashstudio)](https://pypi.org/project/flashstudio/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Training & inference UI for [FlashDet](https://github.com/FlashVision/FlashDet). Runs locally or on Google Colab via Streamlit + ngrok.

---

## Features

- **Training** — Launch FlashDet training, monitor live loss curves, view epoch visualizations
- **Inference** — 17 built-in solutions (counting, speed estimation, heatmaps, etc.) with zone drawing
- **Model Config** — FlashDet (Pico → X) + YOLOv8/v9/v10/v11/YOLOX, LoRA/QLoRA fine-tuning
- **Data** — Upload custom datasets or download from FlashDet registry (COCO/VOC/YOLO formats)
- **Export** — ONNX with FP16 support
- **Dashboard** — Project overview, recent runs, system status

---

## Install

```bash
pip install flashstudio
pip install git+https://github.com/FlashVision/FlashDet.git
```

---

## Usage

### Local

```bash
flashstudio
```

Opens at `http://localhost:8501`.

### Google Colab

```python
from flashstudio import launch
launch(ngrok_token="YOUR_TOKEN")
```

Get a free ngrok token at https://dashboard.ngrok.com/get-started/your-authtoken

---

## Supported Models

| Model | Params | Use Case |
|-------|--------|----------|
| FlashDet-Pico | ~298K | Edge / MCU |
| FlashDet-Nano | ~790K | Embedded / IoT |
| FlashDet-Small | ~1.8M | General |
| FlashDet-Medium | ~3.6M | Accuracy |
| FlashDet-Large | ~5.8M | High accuracy |
| FlashDet-X | ~9.0M | Server |
| YOLOv8/v9/v10/v11 | Varies | General YOLO |

---

## Development

```bash
git clone https://github.com/FlashVision/FlashStudio.git
cd FlashStudio
pip install -e ".[dev]"
pytest tests/
```

---

## Requirements

- Python >= 3.10
- FlashDet (for training/inference)
- ngrok account (free, for Colab only)
- GPU recommended for training

## License

Apache-2.0
