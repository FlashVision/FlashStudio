"""FlashStudio — Training Dashboard Page (reads real FlashDet workspace output)."""

import os
import re
import json
import glob as glob_module

import yaml
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from flashstudio.utils.device import has_cuda, get_device
from flashstudio.utils.filesystem import dir_size_str, list_subdirs, count_files, safe_rmtree
from flashstudio.utils.config import (
    build_training_config, apply_training_config, config_to_yaml_str, load_config_yaml, save_config_yaml,
)


def _get_default_workspace():
    """Auto-detect workspace: use save_dir from config or scan common locations."""
    save_dir = st.session_state.get("save_dir", "")
    if save_dir and os.path.isdir(save_dir):
        return save_dir

    # Try to find FlashDet workspace relative to this package
    candidates = [
        os.path.join(os.getcwd(), "workspace"),
        os.path.join(os.getcwd(), "..", "FlashDet", "workspace"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "FlashDet", "workspace"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c):
            return c

    return os.getcwd()


def render_training_page():
    """Render training dashboard — ultra-compact."""
    from flashstudio.components.styles import render_page_header
    render_page_header("", "Training")

    if "training_active" not in st.session_state:
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Not started"

    # Inline status
    is_active = st.session_state.get("training_active", False)
    pid = st.session_state.get("training_pid")
    if is_active:
        if st.session_state.get("training_paused"):
            st.warning(f"Paused (PID {pid})")
        else:
            st.success(f"Running (PID {pid})")

    tab_start, tab_monitor = st.tabs(["Launch", "Monitor"])

    with tab_start:
        _render_start_tab()
    with tab_monitor:
        _render_monitor_tab()



def _render_monitor_tab():
    """Tab: monitor an existing training run from workspace — with rename & delete."""
    # Auto-detect workspace from project or save_dir
    from flashstudio.components.project_manager import get_active_project, get_project_dir
    active_proj = get_active_project()
    if active_proj:
        default_ws = os.path.join(get_project_dir(active_proj["id"]), "runs")
    else:
        default_ws = _get_default_workspace()

    wc1, wc2 = st.columns([5, 1])
    with wc1:
        workspace = st.text_input("Workspace", value=default_ws, key="workspace_path", label_visibility="collapsed")
    with wc2:
        st.caption("Workspace")

    if not os.path.isdir(workspace):
        st.warning("Workspace not found")
        return

    runs = sorted(
        [d for d in os.listdir(workspace) if os.path.isdir(os.path.join(workspace, d))],
        key=lambda d: os.path.getmtime(os.path.join(workspace, d)),
        reverse=True,
    )
    if not runs:
        st.info("No runs found. Start training first.")
        return

    # Run selector — compact labels
    labels = []
    for r in runs:
        meta = _get_run_meta(os.path.join(workspace, r))
        parts = [meta["status"].split(" ")[0], r]
        if meta["mAP"]:
            parts.append(f"mAP={meta['mAP']:.3f}")
        if meta["size"]:
            parts.append(meta["size"])
        labels.append(" ".join(parts))

    sc1, sc2 = st.columns([7, 1])
    with sc1:
        selected_idx = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i],
                                    key="selected_run_idx", label_visibility="collapsed")
    selected_run = runs[selected_idx]
    run_dir = os.path.join(workspace, selected_run)
    with sc2:
        if st.button("Del", key="delete_run_btn", help="Delete this run"):
            st.session_state["show_delete_run"] = selected_run

    if st.session_state.get("show_delete_run"):
        dc1, dc2 = st.columns([1, 1])
        with dc1:
            if st.button(f"Confirm delete {st.session_state['show_delete_run']}", type="primary", key="confirm_delete_run"):
                import shutil
                shutil.rmtree(os.path.join(workspace, st.session_state["show_delete_run"]))
                st.session_state.pop("show_delete_run", None)
                st.rerun()
        with dc2:
            if st.button("Cancel", key="cancel_delete_run"):
                st.session_state.pop("show_delete_run", None)
                st.rerun()

    # Run info — single compact row of metrics
    meta = _get_run_meta(run_dir)
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Status", meta["status"][:10])
    with m2:
        st.metric("Model", (meta["model"] or "—")[:12])
    with m3:
        st.metric("Epochs", meta["epochs"])
    with m4:
        st.metric("mAP@50", f"{meta['mAP']:.3f}" if meta["mAP"] else "—")
    with m5:
        st.metric("Size", meta["size"] or "—")
    with m6:
        st.metric("Date", meta["date"][:8])

    _render_run_dashboard(run_dir)


def _get_run_meta(run_dir: str) -> dict:
    """Extract metadata from a training run for display."""
    import time

    name = os.path.basename(run_dir)
    meta = {
        "display_name": name,
        "status": "unknown",
        "date": "",
        "epochs": "?",
        "mAP": None,
        "model": "",
        "dataset": "",
        "size": "",
    }

    # Date from folder modification time
    try:
        mtime = os.path.getmtime(run_dir)
        meta["date"] = time.strftime("%b %d %H:%M", time.localtime(mtime))
    except OSError:
        pass

    # Folder size
    try:
        total = 0
        for dirpath, _dirs, files in os.walk(run_dir):
            for f in files:
                total += os.path.getsize(os.path.join(dirpath, f))
        if total > 1_073_741_824:
            meta["size"] = f"{total / 1_073_741_824:.1f} GB"
        elif total > 1_048_576:
            meta["size"] = f"{total / 1_048_576:.0f} MB"
        else:
            meta["size"] = f"{total / 1024:.0f} KB"
    except OSError:
        pass

    # Status: check for checkpoints
    has_best = os.path.isfile(os.path.join(run_dir, "checkpoint_best.pth"))
    has_last = os.path.isfile(os.path.join(run_dir, "checkpoint_last.pth"))
    log_files = glob_module.glob(os.path.join(run_dir, "train_*.log"))
    has_log = bool(log_files)

    if has_best:
        meta["status"] = "Complete"
    elif has_last and has_log:
        meta["status"] = "In Progress"
    elif has_log:
        meta["status"] = "Incomplete"
    else:
        meta["status"] = "Empty"

    # Try to extract model/dataset from log file header
    if log_files:
        try:
            with open(log_files[0], "r") as f:
                header_lines = f.readlines()[:30]
            for line in header_lines:
                line_clean = line.strip()
                # Skip timestamp prefixed lines for model extraction
                if "Model Size" in line_clean or "model_arch" in line_clean:
                    # Extract value after last colon
                    parts = line_clean.rsplit(":", 1)
                    if len(parts) == 2:
                        val = parts[1].strip()
                        if val and len(val) < 30 and not val.startswith("["):
                            meta["model"] = val
                if "dataset" in line_clean.lower() and "path" not in line_clean.lower():
                    parts = line_clean.rsplit(":", 1)
                    if len(parts) == 2:
                        val = parts[1].strip()
                        if val and len(val) < 40 and not val.startswith("["):
                            meta["dataset"] = val
                if "epoch" in line_clean.lower() and "/" in line_clean:
                    match = re.search(r"(\d+)/(\d+)", line_clean)
                    if match:
                        meta["epochs"] = match.group(2)
        except OSError:
            pass

    # Try config.yaml inside run folder
    config_path = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(config_path):
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f.read())
            if cfg:
                if "model" in cfg:
                    meta["model"] = cfg["model"].get("model_size", cfg["model"].get("variant", ""))
                if "dataset" in cfg:
                    meta["dataset"] = cfg["dataset"].get("name", "")
                if "training" in cfg:
                    meta["epochs"] = cfg["training"].get("epochs", "?")
        except Exception:
            pass

    # Try to get mAP from results.json
    results_file = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_file):
        try:
            with open(results_file) as f:
                results = json.load(f)
            mAP = results.get("best_mAP50", 0)
            if mAP > 0:
                meta["mAP"] = mAP
            epochs = results.get("epochs_trained", meta["epochs"])
            meta["epochs"] = epochs
        except (json.JSONDecodeError, OSError):
            pass

    # Build enriched display name
    parts = [name]
    if meta["model"]:
        parts.append(meta["model"])
    if meta["mAP"]:
        parts.append(f"mAP={meta['mAP']:.3f}")
    meta["display_name"] = " | ".join(parts) if len(parts) > 1 else name

    return meta


def _render_run_dashboard(run_dir: str):
    """Render full dashboard for a single training run."""
    log_file = _find_log_file(run_dir)
    history = _parse_training_log(log_file) if log_file else None

    _render_metrics_from_history(history, run_dir)

    tab_curves, tab_viz, tab_gt, tab_log, tab_files = st.tabs([
        "Curves", "Visualizations", "Ground Truth", "Log", "Files"
    ])

    with tab_curves:
        _render_curves(history, run_dir)

    with tab_viz:
        _render_visualizations(run_dir)

    with tab_gt:
        _render_gt_verification(run_dir)

    with tab_log:
        _render_full_log(log_file)

    with tab_files:
        _render_checkpoints(run_dir)


def _find_log_file(run_dir: str):
    """Find the training log file in a run directory."""
    logs = glob_module.glob(os.path.join(run_dir, "train_*.log"))
    if logs:
        return max(logs, key=os.path.getmtime)
    return None


def _parse_training_log(log_path: str):
    """Parse FlashDet training log and extract metrics per epoch."""
    if not log_path or not os.path.isfile(log_path):
        return None

    history = {
        "epochs": [], "lr": [], "train_loss": [],
        "val_loss": [], "mAP50": [],
        "o2m_cls": [], "o2m_box": [], "o2o_cls": [], "o2o_box": [],
        "ema_decay": [], "epoch_time": [],
        "model_info": "", "device": "", "classes": [],
        "total_epochs": 0, "batch_size": 0,
    }

    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Parse header info
        if "Model Size:" in line:
            history["model_info"] = line.split("Model Size:")[-1].strip()
        if "Device:" in line:
            history["device"] = line.split("Device:")[-1].strip()
        if "Classes" in line and ":" in line:
            m = re.search(r"Classes \((\d+)\): \[(.+)\]", line)
            if m:
                history["classes"] = [c.strip().strip("'") for c in m.group(2).split(",")]
        if "Epochs:" in line and "Batch" not in line:
            m = re.search(r"Epochs: (\d+)", line)
            if m:
                history["total_epochs"] = int(m.group(1))
        if "Batch Size:" in line:
            m = re.search(r"Batch Size: (\d+)", line)
            if m:
                history["batch_size"] = int(m.group(1))

        # Parse epoch header: Epoch 1/10 (lr=0.000010, ema_decay=0.000500)
        epoch_m = re.search(r"Epoch (\d+)/(\d+) \(lr=([\d.e+-]+),\s*ema_decay=([\d.e+-]+)\)", line)
        if epoch_m:
            if history["total_epochs"] == 0:
                history["total_epochs"] = int(epoch_m.group(2))
            history["lr"].append(float(epoch_m.group(3)))
            history["ema_decay"].append(float(epoch_m.group(4)))

        # Parse batch loss: Epoch [1] Batch [10/16] Loss: 1156.5105 (...)
        batch_m = re.search(
            r"Epoch \[(\d+)\] Batch \[\d+/\d+\] Loss: ([\d.]+) "
            r"\(.*?o2m_cls: ([\d.]+), o2m_box: ([\d.]+).*?o2o_cls: ([\d.]+), o2o_box: ([\d.]+)",
            line
        )
        if batch_m:
            epoch_num = int(batch_m.group(1))
            total_loss = float(batch_m.group(2))
            # Keep only the last batch of each epoch as the "epoch loss"
            while len(history["train_loss"]) < epoch_num:
                history["train_loss"].append(None)
            history["train_loss"][epoch_num - 1] = total_loss

            while len(history["o2m_cls"]) < epoch_num:
                history["o2m_cls"].append(None)
                history["o2m_box"].append(None)
                history["o2o_cls"].append(None)
                history["o2o_box"].append(None)
            history["o2m_cls"][epoch_num - 1] = float(batch_m.group(3))
            history["o2m_box"][epoch_num - 1] = float(batch_m.group(4))
            history["o2o_cls"][epoch_num - 1] = float(batch_m.group(5))
            history["o2o_box"][epoch_num - 1] = float(batch_m.group(6))

        # Parse epoch time: Epoch time: 366.7s
        time_m = re.search(r"Epoch time: ([\d.]+)s", line)
        if time_m:
            history["epoch_time"].append(float(time_m.group(1)))

        # Parse validation: Validation - Loss: X.XXXX | mAP@0.5: X.XXXX
        val_m = re.search(r"Validation.*Loss: ([\d.]+).*mAP@0.5: ([\d.]+)", line)
        if val_m:
            history["val_loss"].append(float(val_m.group(1)))
            history["mAP50"].append(float(val_m.group(2)))

    # Fill in epochs list
    n = max(len(history["train_loss"]), len(history["lr"]))
    history["epochs"] = list(range(1, n + 1))

    return history


def _render_metrics_from_history(history, run_dir):
    """Render top metric cards from parsed history."""
    cols = st.columns(6)

    if history:
        n_epochs = len(history["epochs"])
        total = history["total_epochs"] or n_epochs

        with cols[0]:
            st.metric("Epoch", f"{n_epochs}/{total}")
        with cols[1]:
            losses = [x for x in history["train_loss"] if x is not None]
            val = f"{losses[-1]:.1f}" if losses else "—"
            st.metric("Train Loss", val)
        with cols[2]:
            val = f"{history['val_loss'][-1]:.4f}" if history["val_loss"] else "—"
            st.metric("Val Loss", val)
        with cols[3]:
            val = f"{history['mAP50'][-1]:.4f}" if history["mAP50"] else "—"
            st.metric("mAP@0.5", val)
        with cols[4]:
            best = f"{max(history['mAP50']):.4f}" if history["mAP50"] else "—"
            st.metric("Best mAP", best)
        with cols[5]:
            lr_val = f"{history['lr'][-1]:.2e}" if history["lr"] else "—"
            st.metric("Current LR", lr_val)

        if n_epochs < total:
            st.progress(n_epochs / max(total, 1))
    else:
        for col in cols:
            with col:
                st.metric("—", "No data")


def _render_curves(history, run_dir):
    """Render training curves — from parsed log or saved plots."""
    # Check for pre-generated plot images
    plots_dir = os.path.join(run_dir, "plots")
    training_curves_img = os.path.join(plots_dir, "training_curves.png")
    map_curve_img = os.path.join(plots_dir, "mAP_curve.png")

    if os.path.isfile(training_curves_img):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.image(training_curves_img, use_container_width=True)
        if os.path.isfile(map_curve_img):
            with cc2:
                st.image(map_curve_img, use_container_width=True)

    if not history or not history["train_loss"]:
        st.info("No data yet.")
        return

    losses = [x for x in history["train_loss"] if x is not None]
    epochs_for_loss = [i + 1 for i, x in enumerate(history["train_loss"]) if x is not None]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Total Loss", "mAP@0.5", "Sub-Losses (o2m/o2o)", "Learning Rate"),
    )

    # Total loss
    fig.add_trace(
        go.Scatter(x=epochs_for_loss, y=losses, mode="lines+markers",
                   name="Train Loss", line=dict(color="#7C3AED", width=2), marker=dict(size=4)),
        row=1, col=1,
    )
    if history["val_loss"]:
        val_epochs = list(range(1, len(history["val_loss"]) + 1))
        fig.add_trace(
            go.Scatter(x=val_epochs, y=history["val_loss"], mode="lines+markers",
                       name="Val Loss", line=dict(color="#F59E0B", width=2), marker=dict(size=4)),
            row=1, col=1,
        )

    # mAP
    if history["mAP50"]:
        map_epochs = list(range(1, len(history["mAP50"]) + 1))
        fig.add_trace(
            go.Scatter(x=map_epochs, y=history["mAP50"], mode="lines+markers",
                       name="mAP@0.5", line=dict(color="#10B981", width=2), marker=dict(size=5)),
            row=1, col=2,
        )

    # Sub-losses
    o2m_cls = [x for x in history["o2m_cls"] if x is not None]
    o2o_cls = [x for x in history["o2o_cls"] if x is not None]
    o2m_box = [x for x in history["o2m_box"] if x is not None]
    if o2m_cls:
        ep = list(range(1, len(o2m_cls) + 1))
        fig.add_trace(
            go.Scatter(x=ep, y=o2m_cls, mode="lines", name="o2m_cls",
                       line=dict(color="#EF4444", width=1.5)),
            row=2, col=1,
        )
    if o2o_cls:
        ep = list(range(1, len(o2o_cls) + 1))
        fig.add_trace(
            go.Scatter(x=ep, y=o2o_cls, mode="lines", name="o2o_cls",
                       line=dict(color="#3B82F6", width=1.5)),
            row=2, col=1,
        )
    if o2m_box:
        ep = list(range(1, len(o2m_box) + 1))
        fig.add_trace(
            go.Scatter(x=ep, y=o2m_box, mode="lines", name="o2m_box",
                       line=dict(color="#F97316", width=1.5, dash="dash")),
            row=2, col=1,
        )

    # LR
    if history["lr"]:
        lr_epochs = list(range(1, len(history["lr"]) + 1))
        fig.add_trace(
            go.Scatter(x=lr_epochs, y=history["lr"], mode="lines",
                       name="LR", line=dict(color="#6366F1", width=2)),
            row=2, col=2,
        )

    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=30, r=10, t=30, b=25),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_visualizations(run_dir):
    """Compact visualization grid."""
    vis_dir = os.path.join(run_dir, "visualizations")
    if not os.path.isdir(vis_dir):
        st.info("No visualizations yet.")
        return

    images = sorted([f for f in os.listdir(vis_dir) if f.endswith(".jpg") and f != "latest_visualization.jpg"])
    if not images:
        st.info("No images yet.")
        return

    latest = os.path.join(vis_dir, "latest_visualization.jpg")
    if os.path.isfile(latest):
        st.image(latest, caption="Latest", use_container_width=True)

    with st.expander(f"All Epochs ({len(images)})", expanded=False):
        for i in range(0, len(images), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(images):
                    with col:
                        st.image(os.path.join(vis_dir, images[idx]), caption=images[idx][:20], use_container_width=True)


def _render_gt_verification(run_dir):
    """Compact GT verification."""
    gt_dir = os.path.join(run_dir, "gt_verification")
    if not os.path.isdir(gt_dir):
        st.info("No GT verification data.")
        return

    report_path = os.path.join(gt_dir, "verification_report.json")
    if os.path.isfile(report_path):
        with open(report_path) as f:
            report = json.load(f)
        st.success("PASSED") if report.get("passed") else st.error("FAILED")
        tc = report.get("splits", {}).get("train", {}).get("coco", {})
        vc = report.get("splits", {}).get("val", {}).get("coco", {})
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Train Imgs", tc.get("num_images", 0))
        with m2:
            st.metric("Train Ann", tc.get("num_annotations", 0))
        with m3:
            st.metric("Val Imgs", vc.get("num_images", 0))
        with m4:
            st.metric("Val Ann", vc.get("num_annotations", 0))

    raw_dir = os.path.join(gt_dir, "images", "raw")
    dl_dir = os.path.join(gt_dir, "images", "dataloader")
    gt_tab_raw, gt_tab_dl = st.tabs(["Raw GT", "Dataloader GT"])
    with gt_tab_raw:
        _render_image_grid(raw_dir, "Raw GT")
    with gt_tab_dl:
        _render_image_grid(dl_dir, "After transforms")


def _render_image_grid(img_dir, description):
    """Render images from a directory in a grid."""
    if not os.path.isdir(img_dir):
        st.info(f"Directory not found: {img_dir}")
        return

    images = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))])
    if not images:
        st.info("No images found.")
        return

    st.caption(f"{description} — {len(images)} images")

    for i in range(0, len(images), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(images):
                img_path = os.path.join(img_dir, images[idx])
                with col:
                    st.image(img_path, caption=images[idx][:30], use_container_width=True)


def _render_full_log(log_path):
    """Compact log viewer."""
    if not log_path or not os.path.isfile(log_path):
        st.info("No log found.")
        return

    with open(log_path) as f:
        content = f.read()
    lines = content.strip().split("\n")
    lc1, lc2 = st.columns([4, 1])
    with lc1:
        st.caption(f"`{os.path.basename(log_path)}` — {len(lines)} lines")
    with lc2:
        show_all = st.checkbox("Full", value=False, key="show_full_log")
    st.code("\n".join(lines if show_all else lines[-30:]), language="bash")


def _render_checkpoints(run_dir):
    """Compact checkpoints list."""
    files_info = []
    for f in sorted(os.listdir(run_dir)):
        fpath = os.path.join(run_dir, f)
        if os.path.isfile(fpath) and f.endswith((".pth", ".json", ".csv", ".log")):
            size = os.path.getsize(fpath)
            size_str = f"{size / (1024*1024):.1f}MB" if size > 1048576 else f"{size / 1024:.0f}KB"
            files_info.append({"File": f, "Size": size_str, "Type": _file_type(f)})

    if files_info:
        st.dataframe(files_info, use_container_width=True, hide_index=True, height=200)
    else:
        st.info("No files.")

    results_path = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Epochs", results.get("epochs_trained", "?"))
        with r2:
            st.metric("Best mAP", f"{results.get('best_mAP50', 0):.4f}")
        with r3:
            st.metric("Val Loss", f"{results.get('best_val_loss', 0):.4f}")
        with r4:
            st.metric("Params", f"{results.get('model_params_M', 0):.2f}M")


def _file_type(filename):
    """Categorize a file by its name."""
    if "best" in filename:
        return "Best checkpoint"
    if "last" in filename:
        return "Latest checkpoint"
    if "inference" in filename:
        return "Inference weights"
    if "fp16" in filename:
        return "FP16 weights"
    if filename.endswith(".json"):
        return "Results/Report"
    if filename.endswith(".csv"):
        return "Training log CSV"
    if filename.endswith(".log"):
        return "Training log"
    return "Other"


# ---- Start New Training Tab ----

def _render_start_tab():
    """Tab: launch a new training run — professional step-based layout."""

    # ─── STEP 1: Experiment Folder ───
    from flashstudio.components.project_manager import get_active_project, get_project_dir
    active_proj = get_active_project()
    if active_proj:
        project_root = get_project_dir(active_proj["id"])
        runs_root = os.path.join(project_root, "runs")
    else:
        runs_root = st.session_state.get("save_dir", os.path.join(os.getcwd(), "flashstudio_runs"))

    os.makedirs(runs_root, exist_ok=True)

    existing_folders = sorted(
        [d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))],
        key=lambda d: os.path.getmtime(os.path.join(runs_root, d)),
        reverse=True,
    )

    with st.container(border=True):
        st.markdown("#### 1. Experiment")
        nc1, nc2, nc3 = st.columns([3, 3, 1])
        with nc1:
            default_name = _generate_run_name()
            new_folder_name = st.text_input("New", value=st.session_state.get("run_name", default_name),
                                            key="new_workflow_folder_name", label_visibility="collapsed")
        with nc3:
            if st.button("Create", key="btn_create_wf", use_container_width=True, type="primary"):
                folder_path = os.path.join(runs_root, new_folder_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                    st.session_state["run_name"] = new_folder_name
                    st.session_state["active_workflow_folder"] = new_folder_name
                    st.rerun()

        with nc2:
            if existing_folders:
                active_folder = st.session_state.get("active_workflow_folder", "")
                folder_labels = [f"{f} ({_get_folder_size_str(os.path.join(runs_root, f))})"
                                 + (" ←" if f == active_folder else "") for f in existing_folders]
                sel_idx = st.selectbox("Existing", range(len(existing_folders)),
                                       format_func=lambda i: folder_labels[i], key="sel_existing_wf",
                                       label_visibility="collapsed")
                bc1, bc2, bc3 = st.columns(3)
                with bc1:
                    if st.button("Use", key="btn_use_wf", help="Use this folder", use_container_width=True):
                        st.session_state["active_workflow_folder"] = existing_folders[sel_idx]
                        st.session_state["run_name"] = existing_folders[sel_idx]
                        st.rerun()
                with bc2:
                    if st.button("Rename", key="btn_rename_wf", help="Rename folder", use_container_width=True):
                        st.session_state["wf_rename_target"] = existing_folders[sel_idx]
                with bc3:
                    if st.button("Delete", key="btn_del_wf", help="Delete folder", use_container_width=True):
                        st.session_state["wf_delete_target"] = existing_folders[sel_idx]

        _render_folder_dialogs(runs_root, existing_folders)

    # Determine active path
    active_folder = st.session_state.get("active_workflow_folder", "")
    if not active_folder and existing_folders:
        active_folder = existing_folders[0]
        st.session_state["active_workflow_folder"] = active_folder
    run_name = active_folder or st.session_state.get("run_name", "untitled_run")
    st.session_state["run_name"] = run_name
    st.session_state["save_dir"] = runs_root
    full_run_path = os.path.join(runs_root, run_name)

    # Config overview — single row
    with st.container(border=True):
        st.markdown("#### 2. Config")
        cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8 = st.columns(8)
        with cc1:
            st.metric("Model", st.session_state.get("model_arch", "Pico").replace("FlashDet-", ""))
        with cc2:
            st.metric("Dataset", (st.session_state.get("dataset_name") or "—")[:8])
        with cc3:
            st.metric("Epochs", st.session_state.get("epochs", 100))
        with cc4:
            st.metric("Batch", st.session_state.get("batch_size", 16))
        with cc5:
            st.metric("LR", f"{st.session_state.get('lr', 0.001):.1e}")
        with cc6:
            st.metric("Img", f"{st.session_state.get('img_size', 320)}")
        with cc7:
            st.metric("Device", "GPU" if has_cuda() else "CPU")
        with cc8:
            st.metric("Output", run_name[:8])

    # Pre-flight + Launch — single row
    with st.container(border=True):
        st.markdown("#### 3. Launch")
        checks = _run_preflight_checks()
        all_ok = True
        check_html = []
        for label, ok, msg in checks:
            status = "OK" if ok else "Failed"
            check_html.append(f'{label}: {status}')
            if not ok:
                all_ok = False
        st.markdown(
            f'<div class="ds-card-stats">' + ' '.join(f'<span>{c}</span>' for c in check_html) + '</div>',
            unsafe_allow_html=True,
        )

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if not st.session_state.get("training_active"):
                if st.button("Start", use_container_width=True, type="primary",
                             key="btn_start_training", disabled=not all_ok):
                    st.session_state.update({"training_active": True, "training_status": "Running",
                                             "training_paused": False, "save_dir": full_run_path})
                    _start_training()
            else:
                if st.button("Stop", use_container_width=True, key="btn_stop_training"):
                    _stop_training()
        with b2:
            if st.session_state.get("training_active"):
                if st.session_state.get("training_paused"):
                    if st.button("Resume", use_container_width=True, key="btn_resume_active"):
                        _resume_active_training()
                else:
                    if st.button("Pause", use_container_width=True, key="btn_pause_training"):
                        _pause_training()
            else:
                st.button("Pause", use_container_width=True, key="btn_pause_disabled", disabled=True)
        with b3:
            if st.button("Resume Ckpt", use_container_width=True, key="btn_resume_training", help="Resume from checkpoint"):
                st.session_state["show_resume_dialog"] = True
        with b4:
            if st.button("Clean", use_container_width=True, key="btn_clean_workspace"):
                st.session_state["show_clean_dialog"] = True

    if st.session_state.get("show_clean_dialog"):
        _render_clean_workspace_dialog()

    if st.session_state.get("show_resume_dialog"):
        _render_resume_dialog()


def _render_folder_dialogs(runs_root: str, existing_folders: list):
    """Render rename/delete dialogs for folder management."""
    import shutil

    if st.session_state.get("wf_rename_target"):
        target = st.session_state["wf_rename_target"]
        with st.container(border=True):
            rename_val = st.text_input("New name", value=target, key="rename_wf_input")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Rename", type="primary", key="do_rename_wf"):
                    if rename_val and rename_val != target:
                        old_path = os.path.join(runs_root, target)
                        new_path = os.path.join(runs_root, rename_val)
                        if os.path.exists(new_path):
                            st.error(f"`{rename_val}` already exists.")
                        else:
                            os.rename(old_path, new_path)
                            if st.session_state.get("active_workflow_folder") == target:
                                st.session_state["active_workflow_folder"] = rename_val
                                st.session_state["run_name"] = rename_val
                            st.session_state.pop("wf_rename_target", None)
                            st.rerun()
            with rc2:
                if st.button("Cancel", key="cancel_rename_wf"):
                    st.session_state.pop("wf_rename_target", None)
                    st.rerun()

    if st.session_state.get("wf_delete_target"):
        target = st.session_state["wf_delete_target"]
        with st.container(border=True):
            st.error(f"Delete **{target}** and all contents?")
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("Delete", type="primary", key="confirm_delete_wf"):
                    shutil.rmtree(os.path.join(runs_root, target), ignore_errors=True)
                    if st.session_state.get("active_workflow_folder") == target:
                        st.session_state.pop("active_workflow_folder", None)
                    st.session_state.pop("wf_delete_target", None)
                    st.rerun()
            with dc2:
                if st.button("Cancel", key="cancel_delete_wf"):
                    st.session_state.pop("wf_delete_target", None)
                    st.rerun()

    if st.session_state.get("show_delete_all_wf_folders"):
        with st.container(border=True):
            st.error(f"Delete ALL {len(existing_folders)} folders? This is permanent!")
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("DELETE ALL", type="primary", key="confirm_delete_all_wf"):
                    for f in existing_folders:
                        shutil.rmtree(os.path.join(runs_root, f), ignore_errors=True)
                    st.session_state.pop("active_workflow_folder", None)
                    st.session_state.pop("show_delete_all_wf_folders", None)
                    st.rerun()
            with dc2:
                if st.button("Cancel", key="cancel_delete_all_wf"):
                    st.session_state.pop("show_delete_all_wf_folders", None)
                    st.rerun()


def _render_config_file_dialog(run_path: str):
    """Compact save/load config dialog."""
    with st.container(border=True):
        tab_save, tab_load = st.tabs(["Save", "Load"])
        with tab_save:
            config = _build_config_dict()
            yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
            with st.expander("YAML Preview"):
                st.code(yaml_str, language="yaml")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.download_button("Download", yaml_str, file_name="config.yaml", mime="text/yaml",
                                   use_container_width=True, type="primary")
            with sc2:
                if st.button("Save to Run", use_container_width=True, key="save_config_to_run"):
                    os.makedirs(run_path, exist_ok=True)
                    with open(os.path.join(run_path, "config.yaml"), "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    st.success("Saved")
        with tab_load:
            load_method = st.radio("From", ["Upload", "Path", "Runs"], key="config_load_method", horizontal=True)
            config_to_load = None
            if load_method == "Upload":
                uploaded = st.file_uploader("YAML", type=["yaml", "yml"], key="config_file_upload")
                if uploaded:
                    try:
                        config_to_load = yaml.safe_load(uploaded.read().decode("utf-8"))
                    except Exception as e:
                        st.error(str(e)[:40])
            elif load_method == "Path":
                cp = st.text_input("Path", placeholder="/path/config.yaml", key="config_path_input")
                if cp and os.path.isfile(cp):
                    try:
                        with open(cp) as f:
                            config_to_load = yaml.safe_load(f.read())
                    except Exception as e:
                        st.error(str(e)[:40])
            else:
                save_dir = st.session_state.get("save_dir", "")
                parent = os.path.dirname(save_dir) if save_dir else ""
                found = [(d, os.path.join(parent, d, "config.yaml"))
                         for d in (os.listdir(parent) if parent and os.path.isdir(parent) else [])
                         if os.path.isfile(os.path.join(parent, d, "config.yaml"))]
                if found:
                    sel = st.selectbox("Run", range(len(found)), format_func=lambda i: found[i][0], key="config_from_run_select")
                    with open(found[sel][1]) as f:
                        config_to_load = yaml.safe_load(f.read())
            if config_to_load:
                if st.button("Apply", type="primary", key="apply_loaded_config", use_container_width=True):
                    _apply_config_dict(config_to_load)
                    st.rerun()


_build_config_dict = build_training_config
_apply_config_dict = apply_training_config


_get_folder_size_str = dir_size_str


def _generate_run_name() -> str:
    """Generate a descriptive default run name."""
    from datetime import datetime
    arch = st.session_state.get("model_arch", "FlashDet-Pico")
    size_code = arch.split("-")[-1].lower()[:4] if "-" in arch else "det"
    dataset = st.session_state.get("dataset_name", "")
    ds_code = dataset.split("(")[0].strip().replace(" ", "")[:8].lower() if dataset else "custom"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return f"{size_code}_{ds_code}_{timestamp}"


def _run_preflight_checks() -> list:
    """Run pre-flight validation checks before training. Returns list of (label, ok, msg)."""
    checks = []

    # 1. Dataset
    train_path = st.session_state.get("train_img_path", "")
    if train_path and os.path.isdir(train_path):
        checks.append(("Dataset", True, ""))
    else:
        checks.append(("Dataset", False, "No train data path"))

    # 2. Annotations
    if train_path and os.path.isdir(train_path):
        ann_file = os.path.join(train_path, "_annotations.coco.json")
        json_files = [f for f in os.listdir(train_path) if f.endswith(".json")] if os.path.isdir(train_path) else []
        if os.path.isfile(ann_file) or json_files:
            checks.append(("Annotations", True, ""))
        else:
            checks.append(("Annotations", False, "No COCO JSON found"))
    else:
        checks.append(("Annotations", False, "No data path"))

    # 3. Model config
    arch = st.session_state.get("model_arch", "")
    if arch:
        checks.append(("Model", True, ""))
    else:
        checks.append(("Model", False, "No model selected"))

    # 4. Device
    if has_cuda():
        checks.append(("GPU", True, ""))
    else:
        checks.append(("GPU", False, "CPU only (slow)"))
        checks[-1] = ("GPU", True, "CPU mode")

    # 5. Disk space
    save_dir = st.session_state.get("save_dir", os.getcwd())
    try:
        stat = os.statvfs(os.path.dirname(save_dir) if not os.path.isdir(save_dir) else save_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        if free_gb > 1:
            checks.append(("Disk", True, ""))
        else:
            checks.append(("Disk", False, f"Only {free_gb:.1f} GB free"))
    except (OSError, AttributeError):
        checks.append(("Disk", True, ""))

    return checks


def _render_clean_workspace_dialog():
    """Compact workspace cleanup."""
    with st.container(border=True):
        workspace = st.session_state.get("save_dir", os.path.join(os.getcwd(), "flashstudio_runs"))
        parent_dir = os.path.dirname(workspace) if not os.path.isdir(workspace) else workspace

        if not os.path.isdir(parent_dir):
            st.info("Nothing to clean.")
            if st.button("Close", key="close_clean"):
                st.session_state["show_clean_dialog"] = False
                st.rerun()
            return

        runs = [{"name": d, "path": os.path.join(parent_dir, d), "size": _get_dir_size(os.path.join(parent_dir, d)),
                 "has_best": os.path.isfile(os.path.join(parent_dir, d, "checkpoint_best.pth"))}
                for d in sorted(os.listdir(parent_dir)) if os.path.isdir(os.path.join(parent_dir, d))]
        if not runs:
            st.info("No runs."); return

        clean_mode = st.radio("Mode", ["Selected", "Keep best only", "Delete incomplete", "Full reset"],
                              key="clean_mode", horizontal=True)

        if clean_mode == "Selected":
            selected = st.multiselect("Runs", [r["name"] for r in runs], key="runs_to_delete")
            if selected and st.button("Delete", type="primary", key="do_delete_selected"):
                import shutil
                for n in selected:
                    shutil.rmtree(os.path.join(parent_dir, n), ignore_errors=True)
                st.session_state["show_clean_dialog"] = False; st.rerun()
        elif clean_mode == "Keep best only":
            if st.button("Clean", type="primary", key="do_keep_best"):
                for r in runs:
                    _cleanup_run_keep_best(r["path"])
                st.session_state["show_clean_dialog"] = False; st.rerun()
        elif clean_mode == "Delete incomplete":
            incomplete = [r for r in runs if not r["has_best"]]
            st.caption(f"{len(incomplete)} incomplete")
            if incomplete and st.button("Delete", type="primary", key="do_delete_incomplete"):
                import shutil
                for r in incomplete:
                    shutil.rmtree(r["path"])
                st.session_state["show_clean_dialog"] = False; st.rerun()
        elif clean_mode == "Full reset":
            if st.button("DELETE ALL", type="primary", key="do_delete_all"):
                import shutil
                shutil.rmtree(parent_dir); os.makedirs(parent_dir, exist_ok=True)
                st.session_state["show_clean_dialog"] = False; st.rerun()

        if st.button("Cancel", key="cancel_clean"):
            st.session_state["show_clean_dialog"] = False; st.rerun()


def _render_resume_dialog():
    """Compact resume dialog."""
    with st.container(border=True):
        workspace = st.session_state.get("save_dir", os.path.join(os.getcwd(), "flashstudio_runs"))
        parent_dir = os.path.dirname(workspace) if not os.path.isdir(workspace) else workspace
        resumable = []
        if os.path.isdir(parent_dir):
            for d in sorted(os.listdir(parent_dir), reverse=True):
                full = os.path.join(parent_dir, d)
                ckpt = os.path.join(full, "checkpoint_last.pth")
                if os.path.isdir(full) and os.path.isfile(ckpt):
                    resumable.append({"name": d, "path": full, "ckpt": ckpt})
        if not resumable:
            st.info("No resumable runs.")
            if st.button("Close", key="close_resume"):
                st.session_state["show_resume_dialog"] = False; st.rerun()
            return
        rc1, rc2, rc3 = st.columns([4, 1, 1])
        with rc1:
            selected = st.selectbox("Run", [r["name"] for r in resumable], key="resume_run_select", label_visibility="collapsed")
        with rc2:
            if st.button("Resume", type="primary", key="do_resume"):
                for r in resumable:
                    if r["name"] == selected:
                        st.session_state.update({"resume_training": True, "resume_path": r["ckpt"],
                                                 "save_dir": r["path"], "training_active": True,
                                                 "training_status": "Resuming", "show_resume_dialog": False})
                        _start_training(); break
        with rc3:
            if st.button("Cancel", key="cancel_resume"):
                st.session_state["show_resume_dialog"] = False; st.rerun()


_get_dir_size = dir_size_str


def _cleanup_run_keep_best(run_dir: str) -> int:
    """Remove non-essential files from a run, keeping only best checkpoint and results."""
    keep = {"checkpoint_best.pth", "results.json", "model_best_inference.pth", "model_best_fp16.pth"}
    removed = 0
    for f in os.listdir(run_dir):
        fpath = os.path.join(run_dir, f)
        if os.path.isfile(fpath) and f not in keep:
            os.remove(fpath)
            removed += 1
        elif os.path.isdir(fpath) and f in ("visualizations", "gt_verification", "plots"):
            import shutil
            shutil.rmtree(fpath)
            removed += 1
    return removed


def _stop_training():
    """Stop the training subprocess by killing its process tree."""
    import signal

    pid = st.session_state.get("training_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            st.success(f"Training process (PID {pid}) terminated.")
        except ProcessLookupError:
            st.info("Training process already finished.")
        except PermissionError:
            try:
                os.kill(pid, signal.SIGKILL)
                st.success(f"Training process (PID {pid}) force-killed.")
            except Exception as e:
                st.error(f"Could not kill process {pid}: {e}")
        except Exception as e:
            st.error(f"Error stopping training: {e}")

    st.session_state["training_active"] = False
    st.session_state["training_status"] = "Stopped"
    st.session_state["training_paused"] = False
    st.session_state["training_pid"] = None
    st.rerun()


def _pause_training():
    """Pause the training subprocess by sending SIGSTOP."""
    import signal

    pid = st.session_state.get("training_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGSTOP)
            st.session_state["training_paused"] = True
            st.session_state["training_status"] = "Paused"
            st.success(f"Training paused (PID {pid}). GPU memory is still held.")
        except ProcessLookupError:
            st.warning("Training process already finished.")
            st.session_state["training_active"] = False
            st.session_state["training_paused"] = False
        except Exception as e:
            st.error(f"Could not pause training: {e}")
    else:
        st.warning("No training process found to pause.")
    st.rerun()


def _resume_active_training():
    """Resume a paused training subprocess by sending SIGCONT."""
    import signal

    pid = st.session_state.get("training_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGCONT)
            st.session_state["training_paused"] = False
            st.session_state["training_status"] = "Running"
            st.success(f"Training resumed (PID {pid}).")
        except ProcessLookupError:
            st.warning("Training process already finished.")
            st.session_state["training_active"] = False
            st.session_state["training_paused"] = False
        except Exception as e:
            st.error(f"Could not resume training: {e}")
    else:
        st.warning("No training process found to resume.")
    st.rerun()


def _start_training():
    """Start training using flashdet.Trainer Python API with pre-flight validation."""
    from flashdet.data import detect_dataset_format, convert_dataset, verify_dataset

    train_images = st.session_state.get("train_img_path", "")
    val_images = st.session_state.get("val_img_path", "")

    # Validate dataset paths exist
    if not train_images or not os.path.isdir(train_images):
        dataset_name = st.session_state.get("dataset_name", "")
        dataset_id = dataset_name.lower().replace(" ", "").replace("(demo)", "sample")
        for candidate_id in ["sample", "coco2017", "coco2017-val", "voc2007", "voc2012"]:
            if candidate_id in dataset_id or dataset_id in candidate_id:
                candidate_dir = os.path.join("data", candidate_id)
                if os.path.isdir(os.path.join(candidate_dir, "train")):
                    train_images = os.path.join(candidate_dir, "train")
                    val_images = os.path.join(candidate_dir, "valid")
                    break

    if not train_images or not val_images:
        st.error(
            "**No dataset paths configured.** Please go to the **Data** page and either:\n"
            "- Enter train/val image paths manually, or\n"
            "- Download a dataset first"
        )
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "No dataset"
        return

    # Auto-detect and convert format if not COCO
    parent_dir = os.path.dirname(train_images) if os.path.basename(train_images) in ("train", "images") else train_images
    detected_fmt = detect_dataset_format(parent_dir)

    if detected_fmt in ("txt", "voc"):
        st.warning(f"Dataset format detected: **{detected_fmt}**. Converting to COCO JSON...")
        try:
            output_dir = parent_dir + "_coco"
            class_names = None
            raw_names = st.session_state.get("class_names", "")
            if raw_names.strip():
                class_names = [c.strip() for c in raw_names.strip().split("\n") if c.strip()]

            convert_dataset(source_dir=parent_dir, output_dir=output_dir,
                            target_format="coco", class_names=class_names)

            train_images = os.path.join(output_dir, "train")
            val_dir = os.path.join(output_dir, "valid")
            if not os.path.isdir(val_dir):
                val_dir = os.path.join(output_dir, "val")
            val_images = val_dir
            st.session_state["train_img_path"] = train_images
            st.session_state["val_img_path"] = val_images
            st.success("Dataset converted to COCO format.")
        except Exception as e:
            st.error(f"**Format conversion failed:** {e}\n\nPlease convert your dataset to COCO JSON format manually.")
            st.session_state["training_active"] = False
            st.session_state["training_status"] = "Conversion failed"
            return

    # Verify COCO annotation file exists
    train_ann = os.path.join(train_images, "_annotations.coco.json")
    if not os.path.isfile(train_ann):
        ann_candidates = [f for f in os.listdir(train_images) if f.endswith(".json")] if os.path.isdir(train_images) else []
        if ann_candidates:
            st.info(f"Found annotation file: `{ann_candidates[0]}` (expected `_annotations.coco.json`)")
        else:
            st.error(
                f"**Missing annotations!** FlashDet expects `_annotations.coco.json` at:\n"
                f"`{train_ann}`\n\n"
                "Please ensure your dataset has COCO-format annotations in the train directory."
            )
            st.session_state["training_active"] = False
            st.session_state["training_status"] = "Missing annotations"
            return

    # Run training
    _run_flashdet_training()


def _run_flashdet_training():
    """Run actual FlashDet training via Python API (non-blocking subprocess)."""
    import subprocess
    import sys
    import textwrap

    size_map = {
        "FlashDet-Pico": "p", "FlashDet-Nano": "n", "FlashDet-Small": "s",
        "FlashDet-Medium": "m", "FlashDet-Large": "l", "FlashDet-X": "x",
    }

    arch_family = st.session_state.get("arch_family", "FlashDet (recommended)")
    model_arch = st.session_state.get("model_arch", "FlashDet-Pico")
    save_dir = st.session_state.get("save_dir", os.path.join(_get_default_workspace(), "flashstudio_run"))

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    train_images = st.session_state.get("train_img_path", "")
    val_images = st.session_state.get("val_img_path", "")

    architecture = "flashdet"
    if "YOLOv8" in arch_family:
        architecture = "yolov8"
    elif "YOLOv9" in arch_family:
        architecture = "yolov9"
    elif "YOLOv10" in arch_family:
        architecture = "yolov10"
    elif "YOLOv11" in arch_family:
        architecture = "yolov11"
    elif "YOLOX" in arch_family:
        architecture = "yolox"

    model_size = size_map.get(model_arch, "n")
    epochs = st.session_state.get("epochs", 100)
    batch_size = st.session_state.get("batch_size", 16)
    lr = st.session_state.get("lr", 1e-3)
    workers = st.session_state.get("num_workers", 4)
    amp = st.session_state.get("amp", True) and device == "cuda"
    mosaic = st.session_state.get("aug_mosaic", True)
    mixup = st.session_state.get("aug_mixup", False)
    copy_paste = st.session_state.get("aug_copypaste", False)
    warmup_epochs = st.session_state.get("warmup_epochs", 3)
    lora = st.session_state.get("finetune_strategy", "").startswith("LoRA")
    lora_rank = st.session_state.get("lora_rank", 8)
    lora_alpha = st.session_state.get("lora_alpha", 16.0)
    lora_dropout = st.session_state.get("lora_dropout", 0.05)
    lora_variant = st.session_state.get("lora_variant", "standard")
    lora_targets = st.session_state.get("lora_targets", ["backbone", "fpn"])
    grad_accum = st.session_state.get("grad_accum", 1)
    patience = st.session_state.get("patience", 50)
    input_size = st.session_state.get("img_size", 320)
    activation_checkpointing = st.session_state.get("activation_checkpointing", False)
    activation_offloading = st.session_state.get("activation_offloading", False)
    optimizer_in_bwd = st.session_state.get("optimizer_in_bwd", False)
    use_8bit_optimizer = st.session_state.get("use_8bit_optimizer", False)
    compile_model = st.session_state.get("compile_model", False)
    multi_gpu = st.session_state.get("ddp", False)

    # Handle pretrained weights — use pretrained_ckpt for COCO/custom,
    # and finetune for resume from a checkpoint
    pretrain_option = st.session_state.get("pretrain_option", "COCO pretrained (recommended)")
    pretrained_ckpt = None
    if pretrain_option == "Custom weights":
        pretrained_ckpt = st.session_state.get("custom_weights", "")

    resume_ckpt = None
    if st.session_state.get("resume_training", False):
        resume_ckpt = st.session_state.get("resume_path", "")

    # QLoRA support
    qlora = st.session_state.get("qlora", False)
    qlora_dtype = st.session_state.get("qlora_dtype", "int8")

    # Chunked loss support
    chunked_loss = st.session_state.get("chunked_loss", False)
    chunk_size = st.session_state.get("chunk_size", 1024)

    # Class file support
    class_file = st.session_state.get("class_file", None)
    class_names_raw = st.session_state.get("class_names", "")
    if class_names_raw.strip() and not class_file:
        # Write class names to a temp file for Trainer
        import tempfile
        cls_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="classes_")
        for name in class_names_raw.strip().split("\n"):
            if name.strip():
                cls_tmp.write(name.strip() + "\n")
        cls_tmp.close()
        class_file = cls_tmp.name

    backbone_type = "lite"
    if model_arch == "FlashDet-Pico":
        pico_bb = st.session_state.get("pico_backbone", "")
        if "PicoBackbone" in pico_bb or "RepNeXt" in pico_bb:
            backbone_type = "pico_v2"

    pretrained_arg = f'"{pretrained_ckpt}"' if pretrained_ckpt else "None"
    resume_arg = f'"{resume_ckpt}"' if resume_ckpt else "None"
    class_file_arg = f'"{class_file}"' if class_file else "None"

    train_script = textwrap.dedent(f"""\
        from flashdet import Trainer
        trainer = Trainer(
            model_size="{model_size}",
            architecture="{architecture}",
            epochs={epochs},
            batch_size={batch_size},
            lr={lr},
            workers={workers},
            device="{device}",
            save_dir="{save_dir}",
            train_images="{train_images}",
            val_images="{val_images}",
            warmup_epochs={warmup_epochs},
            amp={amp},
            mosaic={mosaic},
            mixup={mixup},
            copy_paste={copy_paste},
            lora={lora},
            lora_variant="{lora_variant}",
            lora_rank={lora_rank},
            lora_alpha={lora_alpha},
            lora_dropout={lora_dropout},
            lora_targets={lora_targets},
            qlora={qlora},
            qlora_dtype="{qlora_dtype}",
            chunked_loss={chunked_loss},
            chunk_size={chunk_size},
            grad_accum={grad_accum},
            patience={patience},
            input_size={input_size},
            activation_checkpointing={activation_checkpointing},
            activation_offloading={activation_offloading},
            optimizer_in_bwd={optimizer_in_bwd},
            use_8bit_optimizer={use_8bit_optimizer},
            compile={compile_model},
            multi_gpu={multi_gpu},
            backbone_type="{backbone_type}",
            pretrained_ckpt={pretrained_arg},
            resume={resume_arg},
            class_file={class_file_arg},
        )
        trainer.train()
    """)

    process = subprocess.Popen(
        [sys.executable, "-c", train_script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    st.session_state["training_pid"] = process.pid

    st.info(f"Training started via `flashdet.Trainer` (PID {process.pid})")
    st.caption(f"Model: `{architecture}-{model_size}` · Epochs: {epochs} · BS: {batch_size} · LR: {lr}")
    st.caption(f"Output: `{save_dir}`")
    st.caption("Switch to 'Monitor Run' tab to see live progress.")
