"""FlashStudio — Training Dashboard Page (reads real FlashDet workspace output)."""

import os
import re
import json
import glob as glob_module

import yaml
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    """Render training dashboard — monitor real FlashDet runs or start new training."""
    from flashstudio.components.styles import render_page_header
    render_page_header("🏋️", "Training Dashboard",
                       "Monitor real FlashDet training runs — logs, curves, visualizations.")

    if "training_active" not in st.session_state:
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Not started"

    tab_start, tab_monitor = st.tabs(["▶️ Start Training", "📊 Monitor Run"])

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

    workspace = st.text_input(
        "Workspace path",
        value=default_ws,
        key="workspace_path",
        help="Path containing training run folders",
    )

    if not os.path.isdir(workspace):
        st.warning(f"Workspace not found: `{workspace}`")
        return

    runs = sorted(
        [d for d in os.listdir(workspace) if os.path.isdir(os.path.join(workspace, d))],
        key=lambda d: os.path.getmtime(os.path.join(workspace, d)),
        reverse=True,
    )

    if not runs:
        st.info("No training runs found in workspace. Start a training first.")
        return

    # ─── Run Selector with metadata ───
    run_display_names = []
    run_metas = []
    for r in runs:
        run_path = os.path.join(workspace, r)
        meta = _get_run_meta(run_path)
        run_metas.append(meta)
        # Rich label: status icon + name + model + date + size
        parts = [meta["status"].split(" ")[0], r]  # status icon + folder name
        if meta["model"]:
            parts.append(f"[{meta['model']}]")
        if meta["mAP"]:
            parts.append(f"mAP={meta['mAP']:.3f}")
        parts.append(f"({meta['date']})")
        if meta["size"]:
            parts.append(f"— {meta['size']}")
        label = " ".join(parts)
        run_display_names.append(label)

    col_select, col_actions = st.columns([3, 1])

    with col_select:
        selected_idx = st.selectbox(
            "Select Training Run",
            range(len(runs)),
            format_func=lambda i: run_display_names[i],
            key="selected_run_idx",
        )

    selected_run = runs[selected_idx]
    run_dir = os.path.join(workspace, selected_run)

    with col_actions:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        act_col1, act_col2, act_col3 = st.columns(3)
        with act_col1:
            if st.button("✏️", key="rename_run_btn", help="Rename this run"):
                st.session_state["show_rename_run"] = True
        with act_col2:
            if st.button("🗑️", key="delete_run_btn", help="Delete this run"):
                st.session_state["show_delete_run"] = selected_run
        with act_col3:
            if st.button("🧹", key="delete_all_runs_btn", help="Delete ALL runs"):
                st.session_state["show_delete_all_runs"] = True

    # ─── Rename Dialog ───
    if st.session_state.get("show_rename_run"):
        with st.container(border=True):
            st.markdown("#### ✏️ Rename Run")
            new_name = st.text_input(
                "New name",
                value=selected_run,
                key="rename_run_input",
                help="Folder will be renamed. Use descriptive names like 'pico_coco_best_run'",
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Rename", type="primary", key="do_rename_run"):
                    if new_name and new_name != selected_run:
                        new_path = os.path.join(workspace, new_name)
                        if os.path.exists(new_path):
                            st.error(f"Name `{new_name}` already exists.")
                        else:
                            os.rename(run_dir, new_path)
                            st.success(f"Renamed to `{new_name}`")
                            st.session_state.pop("show_rename_run", None)
                            st.rerun()
            with rc2:
                if st.button("Cancel", key="cancel_rename_run"):
                    st.session_state.pop("show_rename_run", None)
                    st.rerun()

    # ─── Delete Single Run ───
    if st.session_state.get("show_delete_run"):
        run_to_delete = st.session_state["show_delete_run"]
        with st.container(border=True):
            st.error(f"⚠️ Delete run **{run_to_delete}**? This removes all checkpoints, logs, and visualizations.")
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("Yes, Delete", type="primary", key="confirm_delete_run"):
                    import shutil
                    shutil.rmtree(os.path.join(workspace, run_to_delete))
                    st.success(f"Deleted `{run_to_delete}`")
                    st.session_state.pop("show_delete_run", None)
                    st.rerun()
            with dc2:
                if st.button("Cancel", key="cancel_delete_run"):
                    st.session_state.pop("show_delete_run", None)
                    st.rerun()

    # ─── Delete ALL Runs ───
    if st.session_state.get("show_delete_all_runs"):
        with st.container(border=True):
            st.error(f"⚠️ Delete ALL **{len(runs)} training runs**? This is permanent!")
            st.caption("This frees up disk space but removes all checkpoints and logs.")
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("DELETE ALL RUNS", type="primary", key="confirm_delete_all_runs"):
                    import shutil
                    for r in runs:
                        shutil.rmtree(os.path.join(workspace, r), ignore_errors=True)
                    st.success(f"Deleted {len(runs)} runs.")
                    st.session_state.pop("show_delete_all_runs", None)
                    st.rerun()
            with dc2:
                if st.button("Cancel", key="cancel_delete_all_runs"):
                    st.session_state.pop("show_delete_all_runs", None)
                    st.rerun()

    st.divider()

    # ─── Run Info Card ───
    meta = _get_run_meta(run_dir)
    with st.container(border=True):
        st.markdown(f"#### 📂 {selected_run}")
        info_cols = st.columns(5)
        with info_cols[0]:
            st.markdown(f"**Status:** {meta['status']}")
        with info_cols[1]:
            st.markdown(f"**Model:** {meta['model'] or '—'}")
        with info_cols[2]:
            st.markdown(f"**Epochs:** {meta['epochs']}")
        with info_cols[3]:
            mAP_str = f"{meta['mAP']:.3f}" if meta['mAP'] else "—"
            st.markdown(f"**mAP@50:** {mAP_str}")
        with info_cols[4]:
            st.markdown(f"**Size:** {meta['size'] or '—'}")

        info_cols2 = st.columns(3)
        with info_cols2[0]:
            st.markdown(f"**Last modified:** {meta['date']}")
        with info_cols2[1]:
            st.markdown(f"**Dataset:** {meta['dataset'] or '—'}")
        with info_cols2[2]:
            st.markdown(f"**Path:** `{run_dir}`")

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
        meta["status"] = "✅ Complete"
    elif has_last and has_log:
        meta["status"] = "🔄 In Progress"
    elif has_log:
        meta["status"] = "⚠️ Incomplete"
    else:
        meta["status"] = "📁 Empty"

    # Try to extract model/dataset from log file header
    if log_files:
        try:
            with open(log_files[0], "r") as f:
                header_lines = f.readlines()[:30]
            for line in header_lines:
                if "model" in line.lower() and ":" in line:
                    val = line.split(":", 1)[1].strip()
                    if val and len(val) < 40:
                        meta["model"] = val
                if "dataset" in line.lower() and ":" in line:
                    val = line.split(":", 1)[1].strip()
                    if val and len(val) < 60:
                        meta["dataset"] = val
                if "epoch" in line.lower() and "/" in line:
                    match = re.search(r"(\d+)/(\d+)", line)
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

    # Top metrics row
    _render_metrics_from_history(history, run_dir)

    # Main content in tabs
    tab_curves, tab_viz, tab_gt, tab_log, tab_files = st.tabs([
        "📈 Training Curves", "🖼️ Visualizations", "✅ GT Verification",
        "📜 Full Log", "📁 Checkpoints"
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
        st.markdown("#### Saved Training Curves (matplotlib)")
        st.image(training_curves_img, caption="training_curves.png")
        if os.path.isfile(map_curve_img):
            st.image(map_curve_img, caption="mAP_curve.png")
        st.divider()

    # Interactive plotly charts from parsed log
    if not history or not history["train_loss"]:
        st.info("No training data parsed yet. Training may still be starting.")
        return

    st.markdown("#### Interactive Charts (parsed from log)")

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
        height=500,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_visualizations(run_dir):
    """Show per-epoch GT vs Prediction visualizations."""
    vis_dir = os.path.join(run_dir, "visualizations")

    if not os.path.isdir(vis_dir):
        st.info("No visualizations directory found. Training may not have generated any yet.")
        return

    images = sorted([
        f for f in os.listdir(vis_dir)
        if f.endswith(".jpg") and f != "latest_visualization.jpg"
    ])

    if not images:
        st.info("No visualization images found yet.")
        return

    st.markdown(f"**{len(images)} epoch visualizations** (GT vs Predictions)")

    # Show latest first
    latest = os.path.join(vis_dir, "latest_visualization.jpg")
    if os.path.isfile(latest):
        st.markdown("#### Latest Visualization")
        st.image(latest, caption="Latest (GT left | Predictions right)")

    st.divider()
    st.markdown("#### All Epoch Visualizations")

    # Display in a grid (2 per row)
    for i in range(0, len(images), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(images):
                img_path = os.path.join(vis_dir, images[idx])
                with col:
                    st.image(img_path, caption=images[idx], use_container_width=True)


def _render_gt_verification(run_dir):
    """Show GT verification images and report."""
    gt_dir = os.path.join(run_dir, "gt_verification")

    if not os.path.isdir(gt_dir):
        st.info("No GT verification data found.")
        return

    # Summary
    summary_path = os.path.join(gt_dir, "verification_summary.txt")
    if os.path.isfile(summary_path):
        with open(summary_path) as f:
            st.code(f.read(), language="text")

    # Report JSON highlights
    report_path = os.path.join(gt_dir, "verification_report.json")
    if os.path.isfile(report_path):
        with open(report_path) as f:
            report = json.load(f)

        passed = report.get("passed", False)
        if passed:
            st.success("Annotation verification: PASSED")
        else:
            st.error("Annotation verification: FAILED")

        cols = st.columns(4)
        train_coco = report.get("splits", {}).get("train", {}).get("coco", {})
        val_coco = report.get("splits", {}).get("val", {}).get("coco", {})
        with cols[0]:
            st.metric("Train Images", train_coco.get("num_images", 0))
        with cols[1]:
            st.metric("Train Annotations", train_coco.get("num_annotations", 0))
        with cols[2]:
            st.metric("Val Images", val_coco.get("num_images", 0))
        with cols[3]:
            st.metric("Val Annotations", val_coco.get("num_annotations", 0))

    # GT Images
    st.divider()
    raw_dir = os.path.join(gt_dir, "images", "raw")
    dl_dir = os.path.join(gt_dir, "images", "dataloader")

    gt_tab_raw, gt_tab_dl = st.tabs(["Raw GT Images", "Dataloader GT Images"])

    with gt_tab_raw:
        _render_image_grid(raw_dir, "Raw ground truth with bounding boxes")

    with gt_tab_dl:
        _render_image_grid(dl_dir, "After dataloader transforms (letterbox, normalize)")


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
    """Show full training log file."""
    if not log_path or not os.path.isfile(log_path):
        st.info("No training log found.")
        return

    with open(log_path) as f:
        content = f.read()

    st.markdown(f"**Log file:** `{os.path.basename(log_path)}`")

    # Show last 100 lines by default, with option to expand
    lines = content.strip().split("\n")
    show_all = st.checkbox("Show full log", value=False, key="show_full_log")

    if show_all:
        st.code(content, language="bash")
    else:
        st.code("\n".join(lines[-50:]), language="bash")
        if len(lines) > 50:
            st.caption(f"Showing last 50 of {len(lines)} lines. Check 'Show full log' to see all.")


def _render_checkpoints(run_dir):
    """Show checkpoint files and their sizes."""
    st.markdown("#### Saved Checkpoints & Weights")

    files_info = []
    for f in sorted(os.listdir(run_dir)):
        fpath = os.path.join(run_dir, f)
        if os.path.isfile(fpath) and f.endswith((".pth", ".json", ".csv", ".log")):
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            files_info.append({"File": f, "Size": size_str, "Type": _file_type(f)})

    if files_info:
        st.dataframe(files_info, use_container_width=True, hide_index=True)
    else:
        st.info("No checkpoint files found.")

    # results.json
    results_path = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_path):
        st.divider()
        st.markdown("#### Training Results Summary")
        with open(results_path) as f:
            results = json.load(f)

        cols = st.columns(4)
        with cols[0]:
            st.metric("Epochs Trained", results.get("epochs_trained", "?"))
        with cols[1]:
            st.metric("Best mAP@0.5", f"{results.get('best_mAP50', 0):.4f}")
        with cols[2]:
            st.metric("Best Val Loss", f"{results.get('best_val_loss', 0):.4f}")
        with cols[3]:
            st.metric("Model Params", f"{results.get('model_params_M', 0):.2f}M")


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
    """Tab: start a new training run with full configuration."""
    st.markdown("### Start New Training")

    # ─── Workflow Folder Manager ───
    from flashstudio.components.project_manager import get_active_project, get_project_dir
    active_proj = get_active_project()
    if active_proj:
        project_root = get_project_dir(active_proj["id"])
        runs_root = os.path.join(project_root, "runs")
    else:
        runs_root = st.session_state.get("save_dir", os.path.join(os.getcwd(), "flashstudio_runs"))

    os.makedirs(runs_root, exist_ok=True)

    with st.container(border=True):
        st.markdown("### 📁 Workflow Folder Manager")
        st.caption(f"Root: `{runs_root}` — Each experiment gets its own folder for logs, checkpoints & images.")

        # ─── Create New Folder (always visible at top) ───
        create_col1, create_col2 = st.columns([3, 1])
        with create_col1:
            default_name = _generate_run_name()
            new_folder_name = st.text_input(
                "New folder name",
                value=st.session_state.get("run_name", default_name),
                key="new_workflow_folder_name",
                placeholder="e.g. pico_coco_v1, nano_custom_aug",
            )
        with create_col2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("➕ Create", key="btn_create_workflow_folder", use_container_width=True, type="primary"):
                folder_path = os.path.join(runs_root, new_folder_name)
                if os.path.exists(folder_path):
                    st.warning(f"Folder `{new_folder_name}` already exists!")
                else:
                    os.makedirs(folder_path, exist_ok=True)
                    st.session_state["run_name"] = new_folder_name
                    st.session_state["active_workflow_folder"] = new_folder_name
                    st.success(f"Created: `{new_folder_name}`")
                    st.rerun()

        st.divider()

        # ─── Existing Folders List (always visible) ───
        existing_folders = sorted(
            [d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d))],
            key=lambda d: os.path.getmtime(os.path.join(runs_root, d)),
            reverse=True,
        )

        st.markdown(f"**Existing Folders** ({len(existing_folders)})")

        if not existing_folders:
            st.info("No workflow folders yet. Create one above to get started.")
        else:
            # Show each folder as a row with info + action buttons
            active_folder = st.session_state.get("active_workflow_folder", "")

            for idx, folder_name in enumerate(existing_folders):
                folder_path = os.path.join(runs_root, folder_name)
                size = _get_folder_size_str(folder_path)
                n_files = sum(1 for _ in os.scandir(folder_path) if _.is_file())

                is_active = (folder_name == active_folder)
                prefix = "▶️" if is_active else "📁"

                row_c1, row_c2, row_c3, row_c4, row_c5 = st.columns([3, 1, 1, 1, 1])

                with row_c1:
                    st.markdown(f"{prefix} **{folder_name}**{'  ← active' if is_active else ''}")
                    st.caption(f"{size} · {n_files} files")

                with row_c2:
                    if st.button("✅ Use", key=f"use_folder_{idx}", use_container_width=True):
                        st.session_state["active_workflow_folder"] = folder_name
                        st.session_state["run_name"] = folder_name
                        st.rerun()

                with row_c3:
                    if st.button("✏️", key=f"rename_folder_{idx}", use_container_width=True,
                                 help="Rename"):
                        st.session_state["wf_rename_target"] = folder_name

                with row_c4:
                    if st.button("🗑️", key=f"delete_folder_{idx}", use_container_width=True,
                                 help="Delete"):
                        st.session_state["wf_delete_target"] = folder_name

                with row_c5:
                    if st.button("📂", key=f"open_folder_{idx}", use_container_width=True,
                                 help="View contents"):
                        st.session_state["wf_view_target"] = folder_name

            # ─── Bulk action: Delete ALL ───
            st.divider()
            del_all_c1, del_all_c2 = st.columns([3, 1])
            with del_all_c2:
                if st.button("🧹 Delete ALL Folders", key="btn_delete_all_wf", use_container_width=True):
                    st.session_state["show_delete_all_wf_folders"] = True

        # ─── Rename Dialog (visible when triggered) ───
        if st.session_state.get("wf_rename_target"):
            target = st.session_state["wf_rename_target"]
            with st.container(border=True):
                st.markdown(f"#### ✏️ Rename `{target}`")
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

        # ─── Delete Single Folder Dialog (visible when triggered) ───
        if st.session_state.get("wf_delete_target"):
            target = st.session_state["wf_delete_target"]
            with st.container(border=True):
                st.error(f"⚠️ Delete **{target}** and ALL its contents (logs, checkpoints, images)?")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("Yes, Delete", type="primary", key="confirm_delete_wf"):
                        import shutil
                        shutil.rmtree(os.path.join(runs_root, target), ignore_errors=True)
                        if st.session_state.get("active_workflow_folder") == target:
                            st.session_state.pop("active_workflow_folder", None)
                        st.session_state.pop("wf_delete_target", None)
                        st.success(f"Deleted `{target}`")
                        st.rerun()
                with dc2:
                    if st.button("Cancel", key="cancel_delete_wf"):
                        st.session_state.pop("wf_delete_target", None)
                        st.rerun()

        # ─── Delete ALL Dialog (visible when triggered) ───
        if st.session_state.get("show_delete_all_wf_folders"):
            with st.container(border=True):
                st.error(f"⚠️ Delete ALL **{len(existing_folders)}** workflow folders? This is permanent!")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("DELETE ALL", type="primary", key="confirm_delete_all_wf"):
                        import shutil
                        for f in existing_folders:
                            shutil.rmtree(os.path.join(runs_root, f), ignore_errors=True)
                        st.session_state.pop("active_workflow_folder", None)
                        st.session_state.pop("show_delete_all_wf_folders", None)
                        st.success("All workflow folders deleted.")
                        st.rerun()
                with dc2:
                    if st.button("Cancel", key="cancel_delete_all_wf"):
                        st.session_state.pop("show_delete_all_wf_folders", None)
                        st.rerun()

        # ─── View Folder Contents (visible when triggered) ───
        if st.session_state.get("wf_view_target"):
            target = st.session_state["wf_view_target"]
            target_path = os.path.join(runs_root, target)
            with st.container(border=True):
                st.markdown(f"#### 📂 Contents of `{target}`")
                if os.path.isdir(target_path):
                    contents = sorted(os.listdir(target_path))
                    if contents:
                        file_data = []
                        for f in contents:
                            fp = os.path.join(target_path, f)
                            if os.path.isfile(fp):
                                sz = os.path.getsize(fp)
                                if sz > 1_048_576:
                                    sz_str = f"{sz / 1_048_576:.1f} MB"
                                elif sz > 1024:
                                    sz_str = f"{sz / 1024:.0f} KB"
                                else:
                                    sz_str = f"{sz} B"
                                file_data.append({"File": f, "Size": sz_str})
                            else:
                                file_data.append({"File": f + "/", "Size": "DIR"})
                        st.dataframe(file_data, use_container_width=True, hide_index=True)
                    else:
                        st.info("Folder is empty.")
                if st.button("Close", key="close_view_wf"):
                    st.session_state.pop("wf_view_target", None)
                    st.rerun()

    # Determine final run path from active folder
    active_folder = st.session_state.get("active_workflow_folder", "")
    if not active_folder and existing_folders:
        active_folder = existing_folders[0]
        st.session_state["active_workflow_folder"] = active_folder
    run_name = active_folder or st.session_state.get("run_name", "untitled_run")
    st.session_state["run_name"] = run_name
    save_dir = runs_root
    st.session_state["save_dir"] = save_dir
    full_run_path = os.path.join(save_dir, run_name)

    # ─── Config Summary ───
    with st.container(border=True):
        st.markdown("**Training Summary**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            arch = st.session_state.get("model_arch", "FlashDet-Pico")
            st.metric("Model", arch.replace("FlashDet-", "FD-"))
        with col2:
            dataset = st.session_state.get("dataset_name", "Not selected")
            st.metric("Dataset", dataset[:15] if dataset else "—")
        with col3:
            st.metric("Epochs", st.session_state.get("epochs", 100))
        with col4:
            st.metric("Batch Size", st.session_state.get("batch_size", 16))
        with col5:
            st.metric("LR", f"{st.session_state.get('lr', 0.001):.1e}")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Image Size", st.session_state.get("img_size", 320))
        with col_b:
            strategy = st.session_state.get("finetune_strategy", "Full fine-tune")
            st.metric("Strategy", strategy[:12])
        with col_c:
            device = "GPU" if _has_cuda() else "CPU"
            st.metric("Device", device)
        with col_d:
            train_path = st.session_state.get("train_img_path", "")
            status = "✅ Ready" if train_path and os.path.isdir(train_path) else "❌ Missing"
            st.metric("Data", status)

    # ─── Pre-flight Checks ───
    with st.container(border=True):
        st.markdown("**Pre-flight Checks**")
        checks = _run_preflight_checks()
        cols = st.columns(len(checks))
        all_ok = True
        for i, (label, ok, msg) in enumerate(checks):
            with cols[i]:
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} **{label}**")
                if not ok:
                    st.caption(msg)
                    all_ok = False

    # ─── Action Buttons ───
    col_start, col_clean, col_resume, col_config = st.columns(4)

    with col_start:
        if not st.session_state.get("training_active", False):
            start_disabled = not all_ok
            if st.button("▶️ Start Training", use_container_width=True, type="primary",
                         key="btn_start_training", disabled=start_disabled):
                st.session_state["training_active"] = True
                st.session_state["training_status"] = "Running"
                # Update save_dir to include run name
                st.session_state["save_dir"] = full_run_path
                _start_training()
        else:
            if st.button("⏹️ Stop Training", use_container_width=True, type="secondary",
                         key="btn_stop_training"):
                _stop_training()

    with col_clean:
        if st.button("🧹 Clean Workspace", use_container_width=True, key="btn_clean_workspace"):
            st.session_state["show_clean_dialog"] = True

    with col_resume:
        if st.button("🔄 Resume Training", use_container_width=True, key="btn_resume_training"):
            st.session_state["show_resume_dialog"] = True

    with col_config:
        if st.button("📄 Save/Load Config", use_container_width=True, key="btn_config_file"):
            st.session_state["show_config_dialog"] = not st.session_state.get("show_config_dialog", False)

    # ─── Status ───
    if st.session_state.get("training_active"):
        st.info("Training is running... Switch to **Monitor Run** tab and select latest run to see progress.")

    # ─── Config File Dialog ───
    if st.session_state.get("show_config_dialog"):
        _render_config_file_dialog(full_run_path)

    # ─── Clean Workspace Dialog ───
    if st.session_state.get("show_clean_dialog"):
        _render_clean_workspace_dialog()

    # ─── Resume Dialog ───
    if st.session_state.get("show_resume_dialog"):
        _render_resume_dialog()


def _render_config_file_dialog(run_path: str):
    """Save/load training configuration as YAML file."""
    st.divider()
    with st.container(border=True):
        st.markdown("### 📄 Training Config File")
        st.caption("Save your workflow as a config file to reproduce or share training runs.")

        tab_save, tab_load = st.tabs(["💾 Save Config", "📂 Load Config"])

        with tab_save:
            config = _build_config_dict()

            # Show YAML preview
            yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
            st.code(yaml_str, language="yaml")

            col_dl, col_save = st.columns(2)
            with col_dl:
                st.download_button(
                    "📥 Download config.yaml",
                    yaml_str,
                    file_name="flashstudio_config.yaml",
                    mime="text/yaml",
                    use_container_width=True,
                    type="primary",
                )
            with col_save:
                if st.button("💾 Save to Run Folder", use_container_width=True, key="save_config_to_run"):
                    os.makedirs(run_path, exist_ok=True)
                    config_path = os.path.join(run_path, "config.yaml")
                    with open(config_path, "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    st.success(f"Config saved to `{config_path}`")

        with tab_load:
            st.markdown("Load a previously saved config to restore all settings.")

            load_method = st.radio(
                "Load from",
                ["Upload file", "Enter path", "Select from runs"],
                key="config_load_method",
                horizontal=True,
            )

            config_to_load = None

            if load_method == "Upload file":
                uploaded = st.file_uploader(
                    "Upload config.yaml",
                    type=["yaml", "yml", "json"],
                    key="config_file_upload",
                )
                if uploaded:
                    content = uploaded.read().decode("utf-8")
                    try:
                        config_to_load = yaml.safe_load(content)
                        st.success(f"Loaded: {uploaded.name}")
                    except Exception as e:
                        st.error(f"Failed to parse: {e}")

            elif load_method == "Enter path":
                config_path = st.text_input(
                    "Config file path",
                    placeholder="/path/to/config.yaml",
                    key="config_path_input",
                )
                if config_path and os.path.isfile(config_path):
                    with open(config_path) as f:
                        try:
                            config_to_load = yaml.safe_load(f.read())
                            st.success(f"Loaded from: `{config_path}`")
                        except Exception as e:
                            st.error(f"Failed to parse: {e}")

            else:
                # Find configs from existing runs
                save_dir = st.session_state.get("save_dir", "")
                parent = os.path.dirname(save_dir) if save_dir else ""
                found_configs = []
                if parent and os.path.isdir(parent):
                    for d in os.listdir(parent):
                        cfg_path = os.path.join(parent, d, "config.yaml")
                        if os.path.isfile(cfg_path):
                            found_configs.append((d, cfg_path))

                if found_configs:
                    selected = st.selectbox(
                        "Select config from run",
                        range(len(found_configs)),
                        format_func=lambda i: found_configs[i][0],
                        key="config_from_run_select",
                    )
                    cfg_path = found_configs[selected][1]
                    with open(cfg_path) as f:
                        config_to_load = yaml.safe_load(f.read())
                    st.success(f"Loaded config from run: `{found_configs[selected][0]}`")
                else:
                    st.info("No saved configs found in workspace runs.")

            if config_to_load:
                st.divider()
                st.markdown("**Preview:**")
                st.json(config_to_load)

                if st.button("✅ Apply Config", type="primary", key="apply_loaded_config",
                             use_container_width=True):
                    _apply_config_dict(config_to_load)
                    st.success("Config applied! All settings updated.")
                    st.rerun()


def _build_config_dict() -> dict:
    """Build a complete training config dictionary from current session state."""
    config = {
        "project": {
            "name": st.session_state.get("run_name", "untitled"),
            "description": "",
        },
        "dataset": {
            "name": st.session_state.get("dataset_name", ""),
            "train_images": st.session_state.get("train_img_path", ""),
            "val_images": st.session_state.get("val_img_path", ""),
            "format": st.session_state.get("ann_format", "COCO JSON"),
            "num_classes": st.session_state.get("upload_num_classes", 80),
            "class_names": st.session_state.get("class_names", ""),
        },
        "model": {
            "architecture": st.session_state.get("arch_family", "FlashDet (recommended)"),
            "model_size": st.session_state.get("model_arch", "FlashDet-Pico"),
            "backbone_type": "pico_v2" if "PicoBackbone" in st.session_state.get("pico_backbone", "") else "lite",
            "pretrained": st.session_state.get("pretrain_option", "COCO pretrained (recommended)"),
            "custom_weights": st.session_state.get("custom_weights", ""),
        },
        "training": {
            "epochs": st.session_state.get("epochs", 100),
            "batch_size": st.session_state.get("batch_size", 16),
            "learning_rate": st.session_state.get("lr", 0.001),
            "image_size": st.session_state.get("img_size", 320),
            "warmup_epochs": st.session_state.get("warmup_epochs", 3),
            "patience": st.session_state.get("patience", 50),
            "grad_accum": st.session_state.get("grad_accum", 1),
            "num_workers": st.session_state.get("num_workers", 4),
        },
        "optimizer": {
            "name": st.session_state.get("optimizer", "AdamW"),
            "weight_decay": st.session_state.get("weight_decay", 0.05),
            "use_8bit": st.session_state.get("use_8bit_optimizer", False),
        },
        "augmentation": {
            "mosaic": st.session_state.get("aug_mosaic", True),
            "mixup": st.session_state.get("aug_mixup", False),
            "copy_paste": st.session_state.get("aug_copypaste", False),
        },
        "finetune": {
            "strategy": st.session_state.get("finetune_strategy", "Full fine-tune (all layers trainable)"),
            "lora": st.session_state.get("finetune_strategy", "").startswith("LoRA"),
            "lora_variant": st.session_state.get("lora_variant", "standard"),
            "lora_rank": st.session_state.get("lora_rank", 16),
            "lora_alpha": st.session_state.get("lora_alpha", 32),
            "lora_dropout": st.session_state.get("lora_dropout", 0.0),
            "lora_targets": st.session_state.get("lora_targets", ["backbone", "fpn"]),
            "qlora": st.session_state.get("qlora", False),
            "qlora_dtype": st.session_state.get("qlora_dtype", "int8"),
        },
        "advanced": {
            "amp": st.session_state.get("amp", True),
            "compile": st.session_state.get("compile_model", False),
            "multi_gpu": st.session_state.get("ddp", False),
            "activation_checkpointing": st.session_state.get("activation_checkpointing", False),
            "activation_offloading": st.session_state.get("activation_offloading", False),
            "optimizer_in_bwd": st.session_state.get("optimizer_in_bwd", False),
            "chunked_loss": st.session_state.get("chunked_loss", False),
            "chunk_size": st.session_state.get("chunk_size", 1024),
        },
        "output": {
            "save_dir": st.session_state.get("save_dir", ""),
        },
    }
    return config


def _apply_config_dict(config: dict):
    """Apply a loaded config dictionary to session state."""
    mappings = {
        # dataset
        ("dataset", "name"): "dataset_name",
        ("dataset", "train_images"): "train_img_path",
        ("dataset", "val_images"): "val_img_path",
        ("dataset", "format"): "ann_format",
        ("dataset", "num_classes"): "upload_num_classes",
        ("dataset", "class_names"): "class_names",
        # model
        ("model", "architecture"): "arch_family",
        ("model", "model_size"): "model_arch",
        ("model", "pretrained"): "pretrain_option",
        ("model", "custom_weights"): "custom_weights",
        # training
        ("training", "epochs"): "epochs",
        ("training", "batch_size"): "batch_size",
        ("training", "learning_rate"): "lr",
        ("training", "image_size"): "img_size",
        ("training", "warmup_epochs"): "warmup_epochs",
        ("training", "patience"): "patience",
        ("training", "grad_accum"): "grad_accum",
        ("training", "num_workers"): "num_workers",
        # optimizer
        ("optimizer", "name"): "optimizer",
        ("optimizer", "weight_decay"): "weight_decay",
        ("optimizer", "use_8bit"): "use_8bit_optimizer",
        # augmentation
        ("augmentation", "mosaic"): "aug_mosaic",
        ("augmentation", "mixup"): "aug_mixup",
        ("augmentation", "copy_paste"): "aug_copypaste",
        # finetune
        ("finetune", "strategy"): "finetune_strategy",
        ("finetune", "lora_variant"): "lora_variant",
        ("finetune", "lora_rank"): "lora_rank",
        ("finetune", "lora_alpha"): "lora_alpha",
        ("finetune", "lora_dropout"): "lora_dropout",
        ("finetune", "lora_targets"): "lora_targets",
        ("finetune", "qlora"): "qlora",
        ("finetune", "qlora_dtype"): "qlora_dtype",
        # advanced
        ("advanced", "amp"): "amp",
        ("advanced", "compile"): "compile_model",
        ("advanced", "multi_gpu"): "ddp",
        ("advanced", "activation_checkpointing"): "activation_checkpointing",
        ("advanced", "activation_offloading"): "activation_offloading",
        ("advanced", "optimizer_in_bwd"): "optimizer_in_bwd",
        ("advanced", "chunked_loss"): "chunked_loss",
        ("advanced", "chunk_size"): "chunk_size",
        # output
        ("output", "save_dir"): "save_dir",
    }

    for (section, key), state_key in mappings.items():
        if section in config and key in config[section]:
            val = config[section][key]
            if val is not None and val != "":
                st.session_state[state_key] = val

    # Run name from project section
    if "project" in config and "name" in config["project"]:
        st.session_state["run_name"] = config["project"]["name"]


def _get_folder_size_str(path: str) -> str:
    """Get human-readable folder size."""
    total = 0
    try:
        for dirpath, _dirs, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(dirpath, f))
    except OSError:
        return "0 B"
    if total > 1_073_741_824:
        return f"{total / 1_073_741_824:.1f} GB"
    elif total > 1_048_576:
        return f"{total / 1_048_576:.0f} MB"
    elif total > 1024:
        return f"{total / 1024:.0f} KB"
    return f"{total} B"


def _generate_run_name() -> str:
    """Generate a descriptive default run name."""
    from datetime import datetime
    arch = st.session_state.get("model_arch", "FlashDet-Pico")
    size_code = arch.split("-")[-1].lower()[:4] if "-" in arch else "det"
    dataset = st.session_state.get("dataset_name", "")
    ds_code = dataset.split("(")[0].strip().replace(" ", "")[:8].lower() if dataset else "custom"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return f"{size_code}_{ds_code}_{timestamp}"


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


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
    if _has_cuda():
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
    """Render workspace cleanup dialog."""
    st.divider()
    with st.container(border=True):
        st.markdown("### 🧹 Clean Workspace")

        workspace = st.session_state.get("save_dir", os.path.join(os.getcwd(), "flashstudio_runs"))
        parent_dir = os.path.dirname(workspace) if not os.path.isdir(workspace) else workspace

        if not os.path.isdir(parent_dir):
            st.info("Workspace directory does not exist yet. Nothing to clean.")
            if st.button("Close", key="close_clean"):
                st.session_state["show_clean_dialog"] = False
                st.rerun()
            return

        # List existing runs
        runs = []
        for d in sorted(os.listdir(parent_dir)):
            full = os.path.join(parent_dir, d)
            if os.path.isdir(full):
                size = _get_dir_size(full)
                has_best = os.path.isfile(os.path.join(full, "checkpoint_best.pth"))
                runs.append({"name": d, "path": full, "size": size, "has_best": has_best})

        if not runs:
            st.info("No training runs found.")
            if st.button("Close", key="close_clean"):
                st.session_state["show_clean_dialog"] = False
                st.rerun()
            return

        st.markdown(f"**{len(runs)} runs** in `{parent_dir}`")

        # Options
        clean_mode = st.radio(
            "Clean Mode",
            [
                "Delete selected runs",
                "Keep only best checkpoints (delete logs, visualizations)",
                "Delete all incomplete runs (no checkpoint_best.pth)",
                "Delete everything (full reset)",
            ],
            key="clean_mode",
        )

        if clean_mode == "Delete selected runs":
            selected = st.multiselect(
                "Select runs to delete",
                [r["name"] for r in runs],
                key="runs_to_delete",
            )
            if selected and st.button("🗑️ Delete Selected", type="primary", key="do_delete_selected"):
                for name in selected:
                    path = os.path.join(parent_dir, name)
                    if os.path.isdir(path):
                        import shutil
                        shutil.rmtree(path)
                st.success(f"Deleted {len(selected)} run(s).")
                st.session_state["show_clean_dialog"] = False
                st.rerun()

        elif clean_mode == "Keep only best checkpoints (delete logs, visualizations)":
            st.caption("This will remove logs, visualizations, and non-best checkpoints — keeping only `checkpoint_best.pth` and `results.json`.")
            if st.button("🧹 Clean Up", type="primary", key="do_keep_best"):
                cleaned = 0
                for r in runs:
                    cleaned += _cleanup_run_keep_best(r["path"])
                st.success(f"Cleaned {cleaned} files across {len(runs)} runs.")
                st.session_state["show_clean_dialog"] = False
                st.rerun()

        elif clean_mode == "Delete all incomplete runs (no checkpoint_best.pth)":
            incomplete = [r for r in runs if not r["has_best"]]
            st.caption(f"Found **{len(incomplete)}** incomplete runs (no best checkpoint).")
            if incomplete and st.button("🗑️ Delete Incomplete", type="primary", key="do_delete_incomplete"):
                import shutil
                for r in incomplete:
                    shutil.rmtree(r["path"])
                st.success(f"Deleted {len(incomplete)} incomplete run(s).")
                st.session_state["show_clean_dialog"] = False
                st.rerun()

        elif clean_mode == "Delete everything (full reset)":
            st.error("⚠️ This will permanently delete ALL training runs!")
            if st.button("🗑️ DELETE ALL", type="primary", key="do_delete_all"):
                import shutil
                shutil.rmtree(parent_dir)
                os.makedirs(parent_dir, exist_ok=True)
                st.success("Workspace cleaned. All runs deleted.")
                st.session_state["show_clean_dialog"] = False
                st.rerun()

        if st.button("Cancel", key="cancel_clean"):
            st.session_state["show_clean_dialog"] = False
            st.rerun()


def _render_resume_dialog():
    """Render resume training dialog."""
    st.divider()
    with st.container(border=True):
        st.markdown("### 🔄 Resume Training")

        workspace = st.session_state.get("save_dir", os.path.join(os.getcwd(), "flashstudio_runs"))
        parent_dir = os.path.dirname(workspace) if not os.path.isdir(workspace) else workspace

        if not os.path.isdir(parent_dir):
            st.warning("No workspace found.")
            if st.button("Close", key="close_resume"):
                st.session_state["show_resume_dialog"] = False
                st.rerun()
            return

        # Find runs with last checkpoint
        resumable = []
        for d in sorted(os.listdir(parent_dir), reverse=True):
            full = os.path.join(parent_dir, d)
            last_ckpt = os.path.join(full, "checkpoint_last.pth")
            if os.path.isdir(full) and os.path.isfile(last_ckpt):
                resumable.append({"name": d, "path": full, "ckpt": last_ckpt})

        if not resumable:
            st.info("No resumable runs found (requires `checkpoint_last.pth`).")
            if st.button("Close", key="close_resume_empty"):
                st.session_state["show_resume_dialog"] = False
                st.rerun()
            return

        selected = st.selectbox(
            "Select run to resume",
            [r["name"] for r in resumable],
            key="resume_run_select",
        )

        if st.button("🔄 Resume", type="primary", key="do_resume"):
            for r in resumable:
                if r["name"] == selected:
                    st.session_state["resume_training"] = True
                    st.session_state["resume_path"] = r["ckpt"]
                    st.session_state["save_dir"] = r["path"]
                    st.session_state["training_active"] = True
                    st.session_state["training_status"] = "Resuming"
                    st.session_state["show_resume_dialog"] = False
                    _start_training()
                    break

        if st.button("Cancel", key="cancel_resume"):
            st.session_state["show_resume_dialog"] = False
            st.rerun()


def _get_dir_size(path: str) -> str:
    """Get human-readable directory size."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    if total > 1024 * 1024 * 1024:
        return f"{total / (1024**3):.1f} GB"
    elif total > 1024 * 1024:
        return f"{total / (1024**2):.1f} MB"
    return f"{total / 1024:.0f} KB"


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
    st.session_state["training_pid"] = None
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
            st.success("✅ Dataset converted to COCO format.")
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
