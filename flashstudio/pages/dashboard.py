"""FlashStudio — Dashboard Page."""

import os
import json
import glob as glob_module

import streamlit as st
from flashstudio.components.styles import render_page_header
from flashstudio.utils.device import get_gpu_info, get_colab_runtime_type


def render_dashboard():
    """Render the main dashboard with overview and quick-start."""
    from flashstudio.components.project_manager import get_active_project, get_project_stats, get_project_dir

    render_page_header("⚡", "FlashStudio Dashboard",
                       "End-to-end object detection: Data → Model → Train → Export → Inference")

    # Active project banner
    project = get_active_project()
    if project:
        with st.container(border=True):
            col_proj, col_stats = st.columns([2, 3])
            with col_proj:
                st.markdown(f"### 📋 {project['name']}")
                if project.get("description"):
                    st.caption(project["description"])
            with col_stats:
                stats = get_project_stats(project["id"])
                sc1, sc2, sc3, sc4 = st.columns(4)
                with sc1:
                    st.metric("Runs", stats["runs"])
                with sc2:
                    map_str = f"{stats['best_map']:.4f}" if stats["best_map"] else "—"
                    st.metric("Best mAP", map_str)
                with sc3:
                    st.metric("Size", stats["total_size"])
                with sc4:
                    st.metric("Export", "✅" if stats["has_export"] else "—")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    gpu = get_gpu_info()
    with col1:
        st.metric("🖥️ GPU", gpu["name"] if gpu["available"] else "CPU")
    with col2:
        st.metric("🌐 Runtime", get_colab_runtime_type())
    with col3:
        mem = f"{gpu['memory_total']:.1f} GB" if gpu["available"] else "—"
        st.metric("💾 VRAM", mem)
    with col4:
        st.metric("📦 FlashDet", _get_flashdet_version())

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # Main content
    col_left, col_right = st.columns([2, 1])

    with col_left:
        with st.container(border=True):
            st.markdown("### 🚀 Quick Start Guide")
            st.markdown("""
| Step | Action | Description |
|:----:|--------|-------------|
| 1 | **📦 Data** | Upload your dataset or download an open-source one |
| 2 | **🧠 Model** | Choose architecture & configure training parameters |
| 3 | **🏋️ Train** | Start training and monitor real-time progress |
| 4 | **📤 Export** | Convert to ONNX for deployment |
| 5 | **🔍 Inference** | Test your model on images, video, or RTSP |
            """)
            st.caption("Click **Next** below or use the sidebar to jump to any step.")

    with col_right:
        with st.container(border=True):
            st.markdown("### 📊 Session Info")
            dataset = st.session_state.get("dataset_name", "Not selected")
            model = st.session_state.get("model_arch", "FlashDet-Pico")
            status = st.session_state.get("training_status", "Not started")

            st.markdown(f"**Dataset:** {dataset}")
            st.markdown(f"**Model:** {model}")
            st.markdown(f"**Training:** {status}")

            if st.session_state.get("best_map"):
                st.metric("Best mAP", f"{st.session_state['best_map']:.4f}")
            if st.session_state.get("exported_model"):
                st.markdown(f"**Exported:** {st.session_state['exported_model']}")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # Recent training runs
    _render_recent_runs()

    # Supported models table (correct data)
    with st.container(border=True):
        st.markdown("### 🔗 Supported Architectures")
        models_data = [
            ["FlashDet-Pico", "LiteBackbone (0.5x)", "~298K", "Ultra-fast", "Edge / MCU"],
            ["FlashDet-Nano", "FlashBackbone (stem=32)", "~790K", "Very fast", "Embedded / IoT"],
            ["FlashDet-Small", "FlashBackbone (stem=48)", "~1.8M", "Fast", "General purpose"],
            ["FlashDet-Medium", "FlashBackbone (stem=64)", "~3.6M", "Balanced", "High accuracy"],
            ["FlashDet-Large", "FlashBackbone (stem=80)", "~5.8M", "Accurate", "High accuracy"],
            ["FlashDet-X", "FlashBackbone (stem=96)", "~9.0M", "Max accuracy", "Server"],
            ["YOLOv8", "YOLOv8Backbone", "Varies", "Fast", "General YOLO"],
            ["YOLOv9", "YOLOv9Backbone", "Varies", "Fast", "PGI-based"],
            ["YOLOv10", "YOLOv10Backbone", "Varies", "Fast", "PSA-enhanced"],
            ["YOLOv11", "YOLOv11Backbone", "Varies", "Fast", "C2PSA-based"],
            ["YOLOX", "YOLOXBackbone", "Varies", "Fast", "Anchor-free"],
        ]
        st.dataframe(
            {
                "Model": [m[0] for m in models_data],
                "Backbone": [m[1] for m in models_data],
                "Params": [m[2] for m in models_data],
                "Speed": [m[3] for m in models_data],
                "Best For": [m[4] for m in models_data],
            },
            use_container_width=True,
            hide_index=True,
        )
        st.caption("All architectures use fixed backbone/neck/head — only FlashDet-Pico allows backbone choice (lite vs pico_v2).")


def _render_recent_runs():
    """Show recent training runs from workspace."""
    workspace_candidates = [
        os.path.join(os.getcwd(), "workspace"),
        os.path.join(os.getcwd(), "..", "FlashDet", "workspace"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "FlashDet", "workspace"),
    ]

    workspace = None
    for c in workspace_candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c):
            workspace = c
            break

    if not workspace:
        return

    runs = sorted(
        [d for d in os.listdir(workspace) if os.path.isdir(os.path.join(workspace, d))],
        key=lambda d: os.path.getmtime(os.path.join(workspace, d)),
        reverse=True,
    )[:5]

    if not runs:
        return

    with st.container(border=True):
        st.markdown("### 🕐 Recent Training Runs")

        for run_name in runs:
            run_dir = os.path.join(workspace, run_name)
            # Try to get results summary
            results_path = os.path.join(run_dir, "results.json")
            log_files = glob_module.glob(os.path.join(run_dir, "train_*.log"))

            col_name, col_info, col_status = st.columns([2, 3, 1])
            with col_name:
                st.markdown(f"**{run_name}**")
            with col_info:
                if os.path.isfile(results_path):
                    with open(results_path) as f:
                        results = json.load(f)
                    mAP = results.get("best_mAP50", 0)
                    epochs = results.get("epochs_trained", "?")
                    st.caption(f"mAP: {mAP:.4f} · {epochs} epochs")
                elif log_files:
                    st.caption("Training in progress / incomplete")
                else:
                    st.caption("No data")
            with col_status:
                has_best = os.path.isfile(os.path.join(run_dir, "checkpoint_best.pth"))
                if has_best:
                    st.caption("✅ Done")
                elif log_files:
                    st.caption("🔄 Active")
                else:
                    st.caption("—")


def _get_flashdet_version() -> str:
    try:
        import flashdet
        return getattr(flashdet, "__version__", "installed")
    except Exception:
        return "not installed"
