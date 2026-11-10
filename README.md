<p align="center">
  <img src="https://raw.githubusercontent.com/FlashVision/FlashStudio/main/assets/logo.png" width="200" alt="FlashStudio Logo">
</p>

<h1 align="center">FlashStudio</h1>

<p align="center">
  <b>Visual training, export and inference studio for FlashDet and YOLO models.</b><br>
  Run it in your browser with Streamlit, in Google Colab, or as a standalone Windows app.
</p>

<p align="center">
  <a href="https://pypi.org/project/flashstudio/"><img src="https://badge.fury.io/py/flashstudio.svg" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/FlashDet-Powered-00CCFF?logo=lightning&logoColor=white" alt="FlashDet">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  <a href="https://github.com/FlashVision/FlashStudio/releases/tag/desktop-v0.2.1"><img src="https://img.shields.io/badge/Windows-Download%20.exe-0078D6?logo=windows&logoColor=white" alt="Download for Windows"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#features">Features</a> ·
  <a href="#ui-tour">UI tour</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#usage">Usage</a> ·
  <a href="#supported-models">Models</a> ·
  <a href="#troubleshooting">Troubleshooting</a>
</p>

<p align="center">
  <img src="docs/mockups/flashstudio_streamlit_mockup.png" width="800" alt="FlashStudio UI"/>
</p>

---

<a id="quick-start"></a>

## Quick Start

| I want to… | Do this |
|---|---|
| **Use the Windows app** | [Download `FlashStudio.exe`](https://github.com/FlashVision/FlashStudio/releases/tag/desktop-v0.2.1) — no Python or installer required |
| **Run it locally with Python** | `pip install flashstudio` then `flashstudio` |
| **Run it on Google Colab** | `!pip install -q flashstudio`, then `from flashstudio import launch; launch()` |

> The Windows build is distributed as a **binary release only**. The desktop source code is not published in this repository.

---

<a id="why-flashstudio"></a>

## Why FlashStudio

Object-detection projects usually mean juggling scripts, notebooks, YAML files and a dozen terminal commands. FlashStudio puts the whole workflow in one place:

**Data → Model → Training → Export → Inference**

- No code required for the common path, but every setting stays visible and exportable as YAML.
- Works with **FlashDet** (Pico → X), **Ultralytics YOLOv8/v9/v10/v11** and **YOLOX**.
- Keeps each project isolated: its own datasets, runs, checkpoints and UI state.
- Runs locally, in Colab, or as a portable Windows executable.
- Export to **ONNX**, **TorchScript** or **TensorRT**, with INT8 quantization and benchmarking.

---

<a id="features"></a>

## Features

### Projects and workflow

- 🗂️ **Project Workspaces** — every project keeps its own datasets, runs, checkpoints and UI state; switch projects without losing context.
- 🧭 **Guided Workflow** — Dashboard → Data → Model → Training → Export → Inference, with pipeline readiness and "what to do next" hints.
- 🔌 **Dependency Hints** — in-app guidance for optional runtimes and frameworks, with one-click installs where possible.
- 🚀 **Colab Support** — works out of the box; no token or signup needed.

### Data and annotations

- 📦 **Data Ingest** — native `flashdet download` datasets, ZIP/TAR upload, and COCO/VOC/YOLO conversion.
- 📊 **Dataset Quality** — class balance, empty/corrupt/duplicate images, box-size distribution and annotation counts.
- 🖱️ **Annotation Editor** — edit boxes and classes per image with live preview.
- ✏️ **Annotation Repair** — scan and fix COCO and YOLO boxes, class IDs, duplicates and missing files.

### Models and training

- 🧠 **Framework-first Model Config** — FlashDet (Pico → X, 42 Pico-Zoo backbones), Ultralytics YOLOv8/v9/v10/v11, and YOLOX inference/export.
- 🏋️ **Training Dashboard** — preflight checks, launch/pause/resume/stop, live loss curves, epoch visualizations, ground-truth verification, logs and checkpoints.
- 📈 **Run Comparison** — compare metrics and curves across training runs.
- 🧪 **Active Learning** — export low-confidence detections for review and retraining.

### Export and inference

- ⚡ **Optimize** — layer/kernel fusion, ONNX export with parity checks, INT8 quantization, and FPS/latency benchmarking.
- 📤 **Export** — ONNX and TorchScript for FlashDet and Ultralytics weights, plus TensorRT when available.
- 🔍 **Inference Pipeline** — images, video and RTSP/webcam input with FlashDet, Ultralytics, ONNX Runtime, TorchScript and TensorRT engines.
- 🎯 **Zones & Solutions** — polygon, rectangle and line zones with 16 built-in solutions.
- 📹 **Live Inference** — live FPS and detections, MP4 capture and a multi-camera grid.

---

<a id="windows-app"></a>

## Windows App

A portable, single-file Windows build is available from the releases page:

**➡️ [Download FlashStudio for Windows](https://github.com/FlashVision/FlashStudio/releases/tag/desktop-v0.2.1)**

- Windows 10/11 (64-bit), no installer and no Python required.
- Includes model configuration, dataset handling, training, export and inference.
- Data and runs are stored in `%USERPROFILE%\FlashStudio`.
- Training runs in a separate process so pause, resume and stop work.
- The first launch is slower because the single-file executable unpacks itself to a temporary folder.

The release contains the `.exe` only. If Windows SmartScreen shows a warning, choose **More info → Run anyway** — the build is unsigned.

---

<a id="ui-tour"></a>

## UI Tour

### Dashboard

Project overview with compute, dataset, model and best-mAP metrics, a four-stage pipeline status (Data → Model → Training → Export), guided next steps and recent runs. Expand **Workspace** for environment details.

<p align="center">
  <img src="demos/01_dashboard.gif" width="800" alt="Dashboard"/>
</p>

### Data Management

Upload ZIP/TAR datasets or download from the `flashdet` registry, preview images with annotation overlays, verify splits, repair annotations, edit boxes, and inspect a quality report with class balance and duplicate/empty-image checks.

<p align="center">
  <img src="demos/02_data.gif" width="800" alt="Data Management"/>
</p>

### Model Configuration

Pick a framework first — FlashDet, Ultralytics or YOLOX — then choose an architecture and tune hyperparameters, augmentations and advanced options. The config summary bar updates live and can be saved/loaded as YAML.

<p align="center">
  <img src="demos/03_model.gif" width="800" alt="Model Configuration"/>
</p>

### Training

Create a run, pass preflight checks, then launch and control training. Monitor live curves, epoch visualizations, ground truth, logs and checkpoints — or compare metrics across runs.

<p align="center">
  <img src="demos/04_training.gif" width="800" alt="Training"/>
</p>

### Export & Optimize

Framework-first export with dependency hints: select a checkpoint, choose ONNX/TorchScript (plus TensorRT for Ultralytics), then fuse layers, quantize to INT8, and benchmark FPS and latency.

<p align="center">
  <img src="demos/05_export.gif" width="800" alt="Export"/>
</p>

### Inference

Framework-first inference with Model → Data → Solution → Run, including Batch and Live modes. Upload images, video or an RTSP stream, draw polygon/rectangle/line zones, run detection, and review per-image detections and timings.

<p align="center">
  <img src="demos/06_inference.gif" width="800" alt="Inference"/>
</p>

---

<a id="installation"></a>

## Installation

### From PyPI

```bash
pip install flashstudio
```

This installs FlashStudio, FlashDet and the base dependencies.

Optional extras:

```bash
pip install "flashstudio[full]"       # torch, export and optimization stack
pip install "flashstudio[inference]"  # Ultralytics + ONNX runtimes
pip install "flashstudio[ngrok]"      # ngrok fallback for Colab
pip install "flashstudio[dev]"        # ruff, pytest, pytest-cov
```

<details>
<summary><b>Development install (from source)</b></summary>

```bash
git clone https://github.com/FlashVision/FlashStudio.git
cd FlashStudio
pip install -e ".[dev]"
```

</details>

---

<a id="usage"></a>

## Usage

### Local machine

**Option 1: CLI**

```bash
flashstudio --port 8501
```

**Option 2: Streamlit directly**

```bash
git clone https://github.com/FlashVision/FlashStudio.git
cd FlashStudio
streamlit run flashstudio/app.py
```

Then open **http://localhost:8501**.

### First run

1. Create or select a project in the sidebar.
2. Open **Data** and upload a ZIP/TAR dataset, download one from the `flashdet` registry, or convert an existing dataset.
3. Pick a model in **Model** — start with `FlashDet-Small` for a good speed/accuracy balance.
4. Launch training in **Training** and watch the live metrics.
5. Export to ONNX or TorchScript in **Export**, then try it in **Inference**.

Datasets and runs live inside the active project folder. Shared weights are cached in `~/.flashstudio/weights`, and storage locations can be changed in the app.

### Google Colab

No signup or token needed.

**Step 1: Install**

```python
!pip install -q flashstudio
```

**Step 2: Launch**

```python
from flashstudio import launch
launch()
```

A clickable **"Open FlashStudio"** button appears in the cell output. Click it to open the UI.

FlashStudio auto-detects Colab and uses the built-in proxy to create a working URL — no `localhost`, no ngrok, no token.

<details>
<summary><b>Optional: ngrok for a stable public URL</b></summary>

If the built-in proxy doesn't work for your setup, you can use ngrok as a fallback:

1. Get a free token at <https://dashboard.ngrok.com/get-started/your-authtoken>
2. Pass it when launching:

```python
launch(ngrok_token="YOUR_TOKEN")
```

Or set it as an environment variable:

```python
import os
os.environ["NGROK_TOKEN"] = "YOUR_TOKEN"
launch()
```

</details>

---

## Google Colab Notebooks

| Notebook | Description | Link |
|----------|-------------|------|
| Training | Train FlashDet models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FlashVision/FlashStudio/blob/main/notebooks/FlashStudio_Train.ipynb) |
| Inference | Run detection on images/video | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FlashVision/FlashStudio/blob/main/notebooks/FlashStudio_Inference.ipynb) |

---

<a id="supported-models"></a>

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

## Project Structure

```
FlashStudio/
├── flashstudio/
│   ├── __init__.py                  # Package init + launch() export
│   ├── app.py                       # Main Streamlit app (shell, routing, autosave)
│   ├── launcher.py                  # Colab/local launcher (auto-tunnel)
│   ├── cli.py                       # CLI entrypoint
│   ├── constants.py                 # Centralized constants (paths, defaults, models)
│   ├── core/
│   │   └── constants.py             # Re-export of canonical constants
│   │
│   ├── pages/                       # Each page is a sub-package
│   │   ├── dashboard/
│   │   │   ├── page.py              # Dashboard render
│   │   │   └── overview/            # Project banner, metrics, pipeline status,
│   │   │                            # quick actions, recent runs
│   │   ├── data/
│   │   │   ├── page.py              # Data page shell + tab routing
│   │   │   ├── helpers.py           # Extract, convert, download runners
│   │   │   ├── upload/tab.py        # ZIP/TAR upload + class configuration
│   │   │   ├── download/tab.py      # Quick Start, native, and external datasets
│   │   │   ├── preview/tab.py       # Image grid with COCO/YOLO overlays
│   │   │   ├── verify/tab.py        # Dataset validation and split stats
│   │   │   ├── edit/tab.py          # Manual annotation editor
│   │   │   ├── fix/
│   │   │   │   ├── tab.py           # Repair tab
│   │   │   │   └── annotations.py   # COCO/YOLO repair engine
│   │   │   └── quality/tab.py       # Quality report and class balance
│   │   ├── model/
│   │   │   ├── page.py              # Model page shell + summary bar
│   │   │   ├── architecture/tab.py  # Framework + architecture selection
│   │   │   ├── hyperparams/tab.py   # Training schedule and optimizer
│   │   │   ├── augment/tab.py       # Augmentation settings
│   │   │   ├── advanced/tab.py      # Memory/distributed/advanced options
│   │   │   └── utils/config.py      # Config build/apply helpers
│   │   ├── training/
│   │   │   ├── page.py              # Training page shell
│   │   │   ├── launch/              # Run creation, preflight, runner,
│   │   │   │                        # controls, dialogs
│   │   │   ├── monitor/             # Curves, visualizations, ground-truth
│   │   │   │                        # verification, logs, checkpoints, parsers
│   │   │   └── compare/tab.py       # Cross-run comparison
│   │   ├── export/
│   │   │   ├── page.py              # Framework-first Export page
│   │   │   ├── weights/             # Exporter engine + shared settings UI
│   │   │   └── optimization/tab.py  # Fusion, export, INT8, benchmark
│   │   └── inference/
│   │       ├── page.py              # Framework-first Inference page
│   │       ├── model/tab.py         # Runtime + weights selection
│   │       ├── data/tab.py          # Images / video / RTSP input
│   │       ├── solution/tab.py      # Solutions + zone drawing
│   │       └── run/                 # Run tab, detection, video, live stream
│   │
│   ├── components/
│   │   ├── sidebar.py               # Workspace navigation + status
│   │   ├── styles.py                # Shared visual system and page chrome
│   │   ├── wizard.py                # Step indicator + navigation footer
│   │   ├── project_manager.py       # Project CRUD + state persistence
│   │   ├── dependencies.py          # Dependency checks + in-app install
│   │   └── zone_drawer/             # Interactive canvas zone drawing
│   │
│   └── utils/
│       ├── config.py                # Training config build/save/load
│       ├── dataset_paths.py         # Dataset path resolution and normalization
│       ├── dataset_sources.py       # Dataset source URLs for UI + downloader
│       ├── dataset_split.py         # Automatic train/validation splitting
│       ├── dataset_state.py         # Repair inconsistent saved dataset state
│       ├── device.py                # GPU / environment detection
│       ├── download_worker.py       # Background dataset download worker
│       ├── filesystem.py            # Directory size/listing helpers
│       ├── jobs.py                  # Background process tracking
│       ├── model_loader.py          # FlashDet / Ultralytics / ONNX / TorchScript runtimes
│       ├── storage.py               # User-configurable storage locations
│       ├── subprocess_env.py        # Environment setup for child processes
│       ├── trainer_launcher.py      # Standalone FlashDet training launcher
│       ├── training_hooks.py        # FlashDet training callbacks
│       ├── universal_convert.py     # Extra annotation-format converters
│       └── weight_hub.py            # Weight discovery + downloads
│
├── tests/                           # Pytest test suite
│   ├── conftest.py                  # Shared fixtures
│   ├── test_app.py                  # App shell tests
│   ├── test_cli.py                  # CLI tests
│   ├── test_constants.py            # Constants validation
│   ├── test_launcher.py             # Launcher tests
│   ├── test_pipeline_flow.py        # End-to-end workflow state
│   ├── test_save_dir_compat.py      # Run directory compatibility
│   ├── test_components/             # Sidebar, wizard, styles, projects, zones
│   ├── test_core/                   # Core constants
│   ├── test_pages/                  # Per-page and per-tab tests
│   └── test_utils/                  # Config, device, jobs, loaders, weight hub
│
├── demos/                           # README GIFs
├── docs/mockups/                    # Hero screenshot
├── notebooks/                       # Ready-to-run Colab notebooks
├── .github/workflows/ci.yml         # GitHub Actions CI
├── .streamlit/config.toml
├── pyproject.toml
└── README.md
```

---

<a id="troubleshooting"></a>

## Troubleshooting

### Windows: SmartScreen or antivirus warning

The executable is unsigned, so Windows may show a SmartScreen prompt. Choose **More info → Run anyway**. Some antivirus tools flag one-file PyInstaller builds; adding an exclusion for the executable resolves it.

### Windows: first launch is slow

The single-file executable extracts itself to a temporary folder on first run. Later launches are faster.

### Colab: "localhost can't be reached" or ngrok error

`launch()` uses Colab's built-in proxy automatically. If it doesn't work, run this in a **new cell**:

```python
import subprocess, sys, time

proc = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run",
     __import__('flashstudio.launcher', fromlist=['_get_app_path'])._get_app_path(),
     "--server.port", "8501", "--server.headless", "true",
     "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(5)

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(8501)"))
```

Open the printed URL in a new tab. No token or signup needed.

### Streamlit: port already in use

```bash
pkill -f "streamlit run"
flashstudio --port 8501
```

### Training is slow or CUDA is not detected

Install a CUDA-enabled PyTorch build for your driver, then restart FlashStudio. FlashStudio detects the available device on the Dashboard; CPU training works but is much slower.

### Missing optional runtime

Open the in-app dependency hints (sidebar or the relevant page). Every optional runtime — Ultralytics, ONNX Runtime, TensorRT, pyngrok — is listed with its install command.

---

## Requirements

- Python >= 3.10
- FlashDet >= 1.3.0 (installed automatically with `pip install flashstudio`)
- GPU recommended for training (T4 or better)
- Google Colab works out of the box

## Contributing

Issues and pull requests are welcome at <https://github.com/FlashVision/FlashStudio>.

```bash
pip install -e ".[dev]"
ruff check flashstudio/ tests/
pytest
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
