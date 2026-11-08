<p align="center">
  <img src="https://raw.githubusercontent.com/FlashVision/FlashStudio/main/assets/logo.png" width="180" alt="FlashStudio Logo">
</p>

<h1 align="center">FlashStudio</h1>

<p align="center">
  <a href="https://pypi.org/project/flashstudio/"><img src="https://badge.fury.io/py/flashstudio.svg" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FlashDet-Powered-00CCFF?logo=lightning&logoColor=white" alt="FlashDet">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
</p>

<p align="center">
  <b>The modern, lightning-fast visual studio for training, fine-tuning, and deploying computer vision models. Built for developers, researchers, and engineers who want power without the configuration headache.</b>
</p>

<p align="center">
  <img src="docs/mockups/flashstudio_streamlit_mockup.png" width="850" alt="FlashStudio UI"/>
</p>

---

## What is FlashStudio? 🤔

Whether you're training a detector for a tiny edge device, fine-tuning a model for a specific industry, or wiring up a multi-camera live stream, getting a vision pipeline off the ground usually means gluing together dozens of one-off scripts, TensorBoard windows, and half-finished inference notebooks.

**FlashStudio** puts everything into a clean, interactive Streamlit workspace. It wraps **FlashDet** and Ultralytics YOLO models into a unified control center where you can manage data, configure architectures, train with real-time feedback, optimize weights, and deploy inference with zero boilerplate.

---

## ✨ Key Features

- **🗂️ Isolated Project Workspaces** — Switch between different vision projects effortlessly. Every project keeps its own datasets, checkpoints, runs, and settings cleanly separated.
- **🧭 Step-by-Step Guided Workflow** — Never lose track of where you are in the pipeline. Move smoothly from **Dashboard ➔ Data ➔ Model ➔ Training ➔ Export ➔ Inference** with smart preflight checks.
- **🔍 Intelligent Data Quality & Repair** — Instantly spot class imbalances, corrupted images, missing files, duplicate bounding boxes, and automatically fix broken COCO or YOLO labels.
- **🧠 Framework-First Architectures** — Choose your weapon: FlashDet (Pico through X, optimized for edge and servers) or Ultralytics (YOLOv8, v9, v10, v11, YOLOX).
- **🏋️ Live Training Cockpit** — Launch runs, watch live loss curves and epoch visualizations in real time, inspect ground-truth bounding box verifications, and compare multiple runs side-by-side.
- **⚡ Advanced Optimization & Export** — Export clean ONNX, TorchScript, or TensorRT weights, apply kernel fusion and INT8 quantization, and benchmark exact FPS/latency out of the box.
- **🎯 Zones & Solutions** — Define custom polygons, rectangles, or tripwire lines with 16 built-in smart solutions (counting, intrusion detection, etc.) for live video streams.
- **🚀 Run Anywhere** — Run smoothly on your local machine via CLI or right inside Google Colab with zero configuration tokens or tunnel setups required.

---

## 🚀 Quick Start

### 1. Installation
Install the package straight from PyPI:
```bash
pip install flashstudio
```

### 2. Launch
Fire up the studio instantly:
```bash
flashstudio
```
Or run it directly through Streamlit:
```bash
streamlit run flashstudio/app.py
```
Open **http://localhost:8501** in your browser and start building!

---

## ☁️ Running in Google Colab

Want free GPU power without local setup? FlashStudio works out-of-the-box in Google Colab with zero tokens or signups needed.

Open your notebook and run:

```python
!pip install -q flashstudio
```

```python
from flashstudio import launch
launch()
```

Click the generated **"Open FlashStudio"** link in your cell output and you're good to go!

| Ready-to-Use Notebooks | Link |
|------------------------|------|
| **Train FlashDet** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FlashVision/FlashStudio/blob/main/notebooks/FlashStudio_Train.ipynb) |
| **Run Inference** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FlashVision/FlashStudio/blob/main/notebooks/FlashStudio_Inference.ipynb) |

---

## 📸 A Quick Look Around the UI

### 📊 Dashboard
Your command center. Check pipeline readiness, system compute status, current dataset stats, best mAP scores, and jump straight into recent runs.
<p align="center"><img src="demos/01_dashboard.gif" width="850" alt="Dashboard"/></p>

### 🗂️ Data Management
Upload your dataset archives, verify splits, browse images with live bounding box overlays, inspect class balance charts, and repair bad annotations in seconds.
<p align="center"><img src="demos/02_data.gif" width="850" alt="Data Management"/></p>

### ⚙️ Model Setup
Select your framework and architecture, fine-tune hyperparameters, configure robust data augmentations, and save your setup as clean YAML configs.
<p align="center"><img src="demos/03_model.gif" width="850" alt="Model Configuration"/></p>

### 📈 Training & Monitoring
Run preflight checks, launch training with a single click, monitor live loss curves, verify ground-truth alignments, and check saved model checkpoints.
<p align="center"><img src="demos/04_training.gif" width="850" alt="Training"/></p>

### ⚡ Export & Optimization
Export your trained weights to ONNX, TorchScript, or TensorRT, apply INT8 quantization, and benchmark model speed on your hardware.
<p align="center"><img src="demos/05_export.gif" width="850" alt="Export"/></p>

### 🎯 Live Inference & Solutions
Test models on images, video files, or live RTSP streams. Draw custom zones and count objects or trigger alerts instantly.
<p align="center"><img src="demos/06_inference.gif" width="850" alt="Inference"/></p>

---

## 💡 Supported Models

| Family | Variants | Recommended Use Case |
|--------|----------|----------------------|
| **FlashDet** | Pico, Nano, Small, Medium, Large, X | Edge microcontrollers to high-throughput servers (~298K to 9M params) |
| **Ultralytics YOLO** | YOLOv8, YOLOv9, YOLOv10, YOLOv11, YOLOX | Standardized deployment and transfer learning |

---

## 🛠️ Development Setup

If you want to run the app from source or contribute to the project:

```bash
git clone https://github.com/FlashVision/FlashStudio.git
cd FlashStudio
pip install -e .
```

Need every optional backend (Ultralytics, ONNX Runtime, TensorRT export, ngrok, …)? Grab the full extra:

```bash
pip install -e ".[full]"
```

### Optional extras

| Extra | What it adds |
|-------|--------------|
| `full` | Torch, FlashDet export tools, Ultralytics, ONNX Runtime, ONNX Slim, ngrok |
| `inference` | Ultralytics + ONNX Runtime for the inference page |
| `ngrok` | Public tunnel support as a Colab fallback |
| `dev` | Ruff, pytest, and pytest-cov |

### Tests

```bash
pytest
```

---

## 🤝 Contributing

Found a bug, have a feature idea, or want to add support for another framework? Issues and pull requests are very welcome. If you're planning something bigger, feel free to open an issue first so we can talk it through before you sink time into it.

---

## 📜 License

Released under the **Apache 2.0 License**. See [LICENSE](LICENSE) for the full text.

