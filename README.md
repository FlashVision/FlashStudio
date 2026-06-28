<p align="center">
  <img src="https://raw.githubusercontent.com/FlashVision/FlashStudio/main/assets/logo.png" width="200" alt="FlashStudio Logo">
</p>

<h1 align="center">FlashStudio</h1>

<p align="center">
  <a href="https://pypi.org/project/flashstudio/"><img src="https://badge.fury.io/py/flashstudio.svg" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FlashDet-Powered-00CCFF?logo=lightning&logoColor=white" alt="FlashDet">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
</p>

<p align="center">
  <b>Interactive Training & Inference UI for FlashDet — runs locally or on Google Colab with a Streamlit interface.</b>
</p>

<p align="center">
  <img src="docs/mockups/flashstudio_streamlit_mockup.png" width="800" alt="FlashStudio UI"/>
</p>

## Features

- 🏋️ **Training Dashboard** — Real-time monitoring with live loss curves, per-epoch visualizations, GT verification
- 🧠 **Model Config** — All 6 FlashDet sizes + YOLOv8/v9/v10/v11/YOLOX with accurate params
- 🔍 **Inference Pipeline** — 4-step wizard: Model → Data → Zone → Run (17 solutions, 6 trackers)
- 📤 **Export** — ONNX export with FP16 auto-generated weights
- 📦 **Data** — Native `flashdet download` datasets + custom upload (COCO/VOC/YOLO formats)
- 📊 **Dashboard** — Overview with recent training runs from workspace
- 🚀 **Colab Support** — ngrok tunneling for remote access

---

## UI Tour

### Dashboard
Project overview with key metrics (GPU, Dataset, Model, mAP), pipeline status, quick actions, and recent training runs.

<p align="center">
  <img src="demos/01_dashboard.gif" width="800" alt="Dashboard"/>
</p>

### Data Management
Upload datasets (ZIP/TAR), download from `flashdet` registry, preview images with navigation, and verify annotations.

<p align="center">
  <img src="demos/02_data.gif" width="800" alt="Data Management"/>
</p>

### Model Configuration
Choose from 6 FlashDet sizes + YOLO variants. Configure hyperparameters, augmentations, and advanced settings (memory optimization, distributed training).

<p align="center">
  <img src="demos/03_model.gif" width="800" alt="Model Configuration"/>
</p>

### Training
Create experiments, launch training with preflight checks, and monitor runs with real-time curves, visualizations, ground truth, logs, and file browser.

<p align="center">
  <img src="demos/04_training.gif" width="800" alt="Training"/>
</p>

### Export
Export trained models to ONNX format with configurable opset, image size, and dynamic batching. Auto-detects saved weights from training.

<p align="center">
  <img src="demos/05_export.gif" width="800" alt="Export"/>
</p>

### Inference
Full inference pipeline with 17 built-in solutions (object counting, speed estimation, heatmaps, security alarms, etc.). Interactive zone drawing with Polygon, Rectangle, and Line tools.

<p align="center">
  <img src="demos/06_inference.gif" width="800" alt="Inference"/>
</p>

---

## Installation

### Step 1: Install FlashStudio

```bash
pip install flashstudio
```

### Step 2: Install FlashDet (required for training/inference)

```bash
pip install git+https://github.com/FlashVision/FlashDet.git
```

### Development install (from source)

```bash
git clone https://github.com/FlashVision/FlashStudio.git
cd FlashStudio
pip install -e .
```

---

## Usage — Local Machine

### Option 1: CLI

```bash
flashstudio --port 8501
```

### Option 2: Streamlit directly

```bash
cd FlashStudio
streamlit run flashstudio/app.py
```

Then open **http://localhost:8501** in your browser.

---

## Usage — Google Colab

### Step 1: Install packages

```python
!pip install flashstudio
!pip install git+https://github.com/FlashVision/FlashDet.git
```

### Step 2: Get ngrok token (free, one-time setup)

FlashStudio uses [ngrok](https://ngrok.com) to create a public URL for the Streamlit UI in Colab.

1. **Sign up** (free): https://dashboard.ngrok.com/signup
2. **Get your auth token**: https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy the token (looks like `2xAbC1234_something...`)

### Step 3: Launch

```python
from flashstudio import launch

# Pass your ngrok token
launch(ngrok_token="YOUR_NGROK_TOKEN_HERE")
```

Or set it as an environment variable:

```python
import os
os.environ["NGROK_TOKEN"] = "YOUR_NGROK_TOKEN_HERE"

from flashstudio import launch
launch()
```

### Step 4: Open the URL

After launching, you'll see output like:

```
============================================================
  FlashStudio is running!
  Local:  http://localhost:8501
  Public: https://abc123.ngrok-free.app
============================================================
```

Click the **Public URL** to open FlashStudio in a new tab.

---

## Google Colab Notebooks (Ready to Use)

| Notebook | Description | Link |
|----------|-------------|------|
| Training | Train FlashDet models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FlashVision/FlashStudio/blob/main/notebooks/FlashStudio_Train.ipynb) |
| Inference | Run detection on images/video | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FlashVision/FlashStudio/blob/main/notebooks/FlashStudio_Inference.ipynb) |

---

## Supported Models

| Model | Params | Best For |
|-------|--------|----------|
| FlashDet-Pico | ~298K | Edge / MCU |
| FlashDet-Nano | ~790K | Embedded / IoT |
| FlashDet-Small | ~1.8M | General purpose |
| FlashDet-Medium | ~3.6M | High accuracy |
| FlashDet-Large | ~5.8M | High accuracy |
| FlashDet-X | ~9.0M | Max accuracy / Server |
| YOLOv8/v9/v10/v11/YOLOX | Varies | General YOLO |

---

## Architecture

```
FlashStudio/
├── flashstudio/
│   ├── __init__.py              # Package init + launch() export
│   ├── app.py                   # Main Streamlit app (wizard flow)
│   ├── launcher.py              # Colab/local launcher with ngrok
│   ├── cli.py                   # CLI entrypoint
│   ├── pages/
│   │   ├── dashboard.py         # Overview + recent training runs
│   │   ├── data.py              # Dataset upload/download
│   │   ├── model.py             # Architecture & hyperparameter config
│   │   ├── training.py          # Training monitor (reads real workspace)
│   │   ├── export.py            # ONNX export
│   │   └── inference.py         # 4-step inference pipeline
│   ├── components/
│   │   ├── sidebar.py           # Navigation sidebar
│   │   ├── styles.py            # Custom CSS
│   │   └── wizard.py            # Step indicator & navigation
│   └── utils/
│       └── device.py            # GPU/environment detection
├── notebooks/
│   ├── FlashStudio_Train.ipynb
│   └── FlashStudio_Inference.ipynb
├── .streamlit/config.toml
├── pyproject.toml
└── README.md
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'pyngrok'`

```bash
pip install --upgrade flashstudio
```

### ngrok authentication error (`ERR_NGROK_4018`)

You need an ngrok auth token. Get one free at:
https://dashboard.ngrok.com/get-started/your-authtoken

Then pass it to `launch(ngrok_token="your_token")`.

### Streamlit port already in use

```bash
# Kill existing Streamlit processes
pkill -f "streamlit run"

# Then restart
flashstudio --port 8501
```

---

## Requirements

- Python >= 3.9
- FlashDet (`pip install git+https://github.com/FlashVision/FlashDet.git`)
- ngrok account (free) for Google Colab usage
- GPU recommended for training (T4 or better)

## License

Apache-2.0
