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
    """Auto-detect workspace: flashdet default is workspace/ in cwd."""
    save_dir = st.session_state.get("save_dir", "")
    if save_dir and os.path.isdir(save_dir):
        return save_dir

    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, "workspace"),
        os.path.join(cwd, "flashstudio_runs"),
        os.path.join(cwd, "..", "FlashDet", "workspace"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c):
            return c

    # Default to workspace/ (flashdet's default output location)
    return os.path.join(cwd, "workspace")


def _check_training_process():
    """Check if the training subprocess is still alive; extract errors if it crashed."""
    pid = st.session_state.get("training_pid")
    if not pid or not st.session_state.get("training_active"):
        return

    # Check if process is still running
    import signal
    try:
        os.kill(pid, 0)
        return  # still alive
    except (ProcessLookupError, PermissionError):
        pass  # process is dead

    # Process exited — read log to find errors
    log_path = st.session_state.get("training_log_file", "")
    error_lines = []
    last_lines = []
    if log_path and os.path.isfile(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
            last_lines = all_lines[-30:] if len(all_lines) > 30 else all_lines

            in_traceback = False
            for line in all_lines:
                stripped = line.strip()
                if "Traceback" in stripped:
                    in_traceback = True
                    error_lines = [stripped]
                elif in_traceback:
                    error_lines.append(stripped)
                elif stripped.startswith("[") and "[ERROR]" in stripped:
                    error_lines.append(stripped)
                elif "Error:" in stripped or "Exception:" in stripped:
                    error_lines.append(stripped)
        except OSError:
            pass

    st.session_state["training_active"] = False
    st.session_state["training_pid"] = None

    if error_lines:
        st.session_state["training_status"] = "Failed"
        st.session_state["training_error"] = "\n".join(error_lines[-15:])
    else:
        complete_markers = ("model_final_inference.pth", "model_final_fp16.pth")
        save_dir = st.session_state.get("save_dir", "")
        completed = any(
            os.path.isfile(os.path.join(save_dir, m)) for m in complete_markers
        ) if save_dir else False
        if completed:
            st.session_state["training_status"] = "Complete"
        else:
            tail = "".join(last_lines[-5:]).strip() if last_lines else "No log output"
            st.session_state["training_status"] = "Stopped"
            st.session_state["training_error"] = f"Training process exited.\n\nLast log output:\n{tail}"


def render_training_page():
    """Render training dashboard — ultra-compact."""
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Training")

    show_flashes()

    if "training_active" not in st.session_state:
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Not started"

    # Check if training process crashed
    _check_training_process()

    # Show persistent training error if any
    training_err = st.session_state.pop("training_error", None)
    if training_err:
        lines = training_err.strip().split("\n")
        error_summary = ""
        for line in reversed(lines):
            stripped = line.strip()
            if "Error:" in stripped or "Exception:" in stripped:
                error_summary = stripped
                break
            if stripped.startswith("[") and "[ERROR]" in stripped:
                error_summary = stripped
                break
        if error_summary:
            st.error(f"**Training Error:** {error_summary}")
        else:
            st.error("**Training failed** — see log details below")
        with st.expander("Full error log", expanded=True):
            st.code(training_err, language="text")

    # Inline status
    is_active = st.session_state.get("training_active", False)
    pid = st.session_state.get("training_pid")
    status = st.session_state.get("training_status", "")
    if is_active:
        if st.session_state.get("training_paused"):
            st.warning(f"Paused (PID {pid})")
        else:
            st.success(f"Running (PID {pid})")
    elif status == "Failed":
        st.error(f"Training failed")
    elif status == "Complete":
        st.success("Training complete")

    tab_start, tab_monitor = st.tabs(["Launch", "Monitor"])

    with tab_start:
        _render_start_tab()
    with tab_monitor:
        _render_monitor_tab()



def _render_monitor_tab():
    """Tab: monitor an existing training run from workspace — with rename & delete."""
    # Always default to workspace/ (flashdet's default output location)
    default_ws = os.path.join(os.getcwd(), "workspace")

    wc1, wc2, wc3 = st.columns([5, 1, 1])
    with wc1:
        workspace = st.text_input("Workspace", value=default_ws, key="workspace_path",
                                  label_visibility="collapsed")
    with wc2:
        st.caption("Workspace")
    with wc3:
        if st.button("Refresh", key="mon_refresh_btn", use_container_width=True):
            st.rerun()

    if not os.path.isdir(workspace):
        st.warning("Workspace not found")
        _render_run_dashboard(None)
        return

    # Collect all sub-folders recursively (max 2 levels deep)
    all_folders = []
    for d in sorted(os.listdir(workspace)):
        full = os.path.join(workspace, d)
        if os.path.isdir(full):
            all_folders.append(d)
            try:
                for sd in sorted(os.listdir(full)):
                    sf = os.path.join(full, sd)
                    if os.path.isdir(sf):
                        all_folders.append(os.path.join(d, sd))
            except OSError:
                pass

    if not all_folders:
        st.info("No runs found. Start training first.")
        _render_run_dashboard(None)
        return

    # Sort by modification time (most recent first)
    all_folders.sort(
        key=lambda d: os.path.getmtime(os.path.join(workspace, d)),
        reverse=True,
    )

    labels = []
    for r in all_folders:
        meta = _get_run_meta(os.path.join(workspace, r))
        parts = [meta["status"].split(" ")[0], r]
        if meta["mAP"]:
            parts.append(f"mAP={meta['mAP']:.3f}")
        if meta["size"]:
            parts.append(meta["size"])
        labels.append(" ".join(parts))

    sc1, sc2 = st.columns([7, 1])
    with sc1:
        selected_idx = st.selectbox("Select Run", range(len(all_folders)),
                                    format_func=lambda i: labels[i],
                                    key="selected_run_idx", label_visibility="collapsed")
    selected_run = all_folders[selected_idx]
    run_dir = os.path.join(workspace, selected_run)
    with sc2:
        if st.button("Del", key="delete_run_btn", help="Delete this run"):
            st.session_state["show_delete_run"] = selected_run

    if st.session_state.get("show_delete_run"):
        dc1, dc2 = st.columns([1, 1])
        with dc1:
            if st.button(f"Confirm delete {st.session_state['show_delete_run']}", type="primary",
                         key="confirm_delete_run"):
                import shutil
                from flashstudio.utils import flash
                deleted_name = st.session_state["show_delete_run"]
                shutil.rmtree(os.path.join(workspace, deleted_name), ignore_errors=True)
                st.session_state.pop("show_delete_run", None)
                flash(f"Deleted run `{deleted_name}`", "success")
                st.rerun()
        with dc2:
            if st.button("Cancel", key="cancel_delete_run"):
                st.session_state.pop("show_delete_run", None)
                st.rerun()

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

    # Status: check for checkpoints (FlashDet uses model_final_* for completion)
    has_final = (os.path.isfile(os.path.join(run_dir, "model_final_inference.pth"))
                 or os.path.isfile(os.path.join(run_dir, "model_final_fp16.pth"))
                 or os.path.isfile(os.path.join(run_dir, "checkpoint_best.pth")))
    has_last = os.path.isfile(os.path.join(run_dir, "checkpoint_last.pth"))
    log_files = glob_module.glob(os.path.join(run_dir, "train_*.log"))
    has_log = bool(log_files)
    has_csv = os.path.isfile(os.path.join(run_dir, "training_log.csv"))

    if has_final:
        meta["status"] = "Complete"
    elif has_last and (has_log or has_csv):
        meta["status"] = "In Progress"
    elif has_log or has_csv:
        meta["status"] = "Started"
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

    # Try to get mAP from results.json (FlashDet writes this at training end)
    results_file = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_file):
        try:
            with open(results_file) as f:
                results = json.load(f)
            mAP = results.get("best_mAP50", 0)
            if mAP > 0:
                meta["mAP"] = mAP
            meta["epochs"] = results.get("epochs_trained", meta["epochs"])
            meta["model"] = meta["model"] or results.get("architecture", "")
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: parse training_log.csv for epoch count and mAP
    csv_path = os.path.join(run_dir, "training_log.csv")
    if os.path.isfile(csv_path) and meta["epochs"] == "?":
        try:
            import csv
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                meta["epochs"] = len(rows)
                mAP_vals = [float(r.get("mAP@0.5", 0)) for r in rows
                            if r.get("mAP@0.5", "").strip()]
                if mAP_vals and max(mAP_vals) > 0 and not meta["mAP"]:
                    meta["mAP"] = max(mAP_vals)
        except Exception:
            pass

    # Build enriched display name
    parts = [name]
    if meta["model"]:
        parts.append(meta["model"])
    if meta["mAP"]:
        parts.append(f"mAP={meta['mAP']:.3f}")
    meta["display_name"] = " | ".join(parts) if len(parts) > 1 else name

    return meta


def _parse_training_csv(run_dir: str):
    """Parse training_log.csv — the primary metrics source from FlashDet."""
    csv_path = os.path.join(run_dir, "training_log.csv")
    if not os.path.isfile(csv_path):
        return None

    import csv
    history = {
        "epochs": [], "lr": [], "train_loss": [], "val_loss": [], "mAP50": [],
        "train_box": [], "train_cls": [], "train_l1": [],
        "val_box": [], "val_cls": [], "val_l1": [],
        "model_info": "", "device": "", "classes": [],
        "total_epochs": 0, "batch_size": 0,
    }

    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    epoch = int(float(row.get("epoch", 0)))
                except (ValueError, TypeError):
                    continue
                history["epochs"].append(epoch)
                history["lr"].append(float(row.get("lr", 0)))
                history["train_loss"].append(float(row.get("train_loss", 0)))

                val_loss = row.get("val_loss", "")
                history["val_loss"].append(float(val_loss) if val_loss else None)

                mAP = row.get("mAP@0.5", "")
                history["mAP50"].append(float(mAP) if mAP else None)

                for key in ("train_box", "train_cls", "train_l1", "val_box", "val_cls", "val_l1"):
                    val = row.get(key, "")
                    history[key].append(float(val) if val else None)
    except Exception:
        return None

    # Filter out None from val_loss/mAP (some epochs may not have validation)
    history["val_loss"] = [v for v in history["val_loss"] if v is not None]
    history["mAP50"] = [v for v in history["mAP50"] if v is not None]

    # Fill in metadata from results.json if available
    results_path = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_path):
        try:
            with open(results_path) as f:
                results = json.load(f)
            history["total_epochs"] = results.get("total_epochs", len(history["epochs"]))
            history["model_info"] = results.get("architecture", "")
            tc = results.get("training_config", {})
            history["batch_size"] = tc.get("batch_size", 0)
        except (json.JSONDecodeError, OSError):
            pass

    if not history["total_epochs"]:
        history["total_epochs"] = len(history["epochs"])

    # Fill metadata from log header
    log_file = _find_log_file(run_dir)
    if log_file:
        try:
            with open(log_file, "r") as f:
                header_lines = f.readlines()[:30]
            for line in header_lines:
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
                    if m and not history["total_epochs"]:
                        history["total_epochs"] = int(m.group(1))
                if "Batch Size:" in line:
                    m = re.search(r"Batch Size: (\d+)", line)
                    if m and not history["batch_size"]:
                        history["batch_size"] = int(m.group(1))
        except OSError:
            pass

    return history if history["epochs"] else None


def _render_run_dashboard(run_dir):
    """Render full dashboard for a single training run — always shows all 5 sub-tabs."""
    log_file = None
    history = None

    if run_dir and os.path.isdir(run_dir):
        log_file = _find_log_file(run_dir)
        # Prefer CSV (clean numeric data), fall back to log parsing
        history = _parse_training_csv(run_dir)
        if not history:
            history = _parse_training_log(log_file) if log_file else None
        _render_metrics_from_history(history, run_dir)

        # Check for errors in log file — show error banner at top
        if log_file and os.path.isfile(log_file):
            try:
                with open(log_file) as f:
                    log_content = f.read()
                if "Traceback" in log_content or "Error:" in log_content:
                    last_lines = log_content.strip().split("\n")[-5:]
                    error_summary = "\n".join(last_lines)
                    st.error(f"Training error detected. Check the **Log** tab for details.\n```\n{error_summary}\n```")
            except OSError:
                pass

    tab_curves, tab_viz, tab_gt, tab_log, tab_files = st.tabs([
        "Curves", "Visualizations", "Ground Truth", "Log", "Files"
    ])

    with tab_curves:
        if run_dir and os.path.isdir(run_dir):
            _render_curves(history, run_dir)
        else:
            st.info("No training data yet. Start a training run to see curves.")

    with tab_viz:
        if run_dir and os.path.isdir(run_dir):
            _render_visualizations(run_dir)
        else:
            st.info("No visualizations yet.")

    with tab_gt:
        if run_dir and os.path.isdir(run_dir):
            _render_gt_verification(run_dir)
        else:
            st.info("No GT verification data yet.")

    with tab_log:
        if log_file:
            _render_full_log(log_file)
        else:
            st.info("No training log yet.")

    with tab_files:
        if run_dir and os.path.isdir(run_dir):
            _render_checkpoints(run_dir)
        else:
            st.info("No checkpoint files yet.")


def _find_log_file(run_dir: str):
    """Find the training log file in a run directory."""
    logs = glob_module.glob(os.path.join(run_dir, "train_*.log"))
    if logs:
        return max(logs, key=os.path.getmtime)
    return None


def _parse_training_log(log_path: str):
    """Parse FlashDet training log and extract metrics per epoch.

    Handles the actual FlashDet log format:
      Epoch N/T (lr=X.XXXXXX)
        Val Loss: V (loss_total: ..., o2m_cls: ..., o2m_box: ..., o2o_cls: ..., o2o_box: ...) | mAP@0.5: M
    """
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

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    current_epoch = 0

    for line in lines:
        # Parse header: Model: p, Input: (320, 320)
        model_m = re.search(r"Model: (\w+), Input: \((\d+), (\d+)\)", line)
        if model_m:
            history["model_info"] = f"{model_m.group(1)} ({model_m.group(2)}x{model_m.group(3)})"

        if "Device:" in line:
            dm = re.search(r"Device:\s*(\S+)", line)
            if dm:
                history["device"] = dm.group(1)

        if "Classes" in line and ":" in line:
            m = re.search(r"Classes \((\d+)\): \[(.+)\]", line)
            if m:
                history["classes"] = [c.strip().strip("'") for c in m.group(2).split(",")]

        # Epochs: 100, Batch: 16, LR: 0.001
        header_m = re.search(r"Epochs:\s*(\d+),\s*Batch:\s*(\d+),\s*LR:\s*([\d.e+-]+)", line)
        if header_m:
            history["total_epochs"] = int(header_m.group(1))
            history["batch_size"] = int(header_m.group(2))

        # Epoch N/T (lr=X.XXXXXX) — with or without ema_decay
        epoch_m = re.search(r"Epoch (\d+)/(\d+)\s*\(lr=([\d.e+-]+)", line)
        if epoch_m:
            current_epoch = int(epoch_m.group(1))
            if history["total_epochs"] == 0:
                history["total_epochs"] = int(epoch_m.group(2))
            history["lr"].append(float(epoch_m.group(3)))
            ema_m = re.search(r"ema_decay=([\d.e+-]+)", line)
            if ema_m:
                history["ema_decay"].append(float(ema_m.group(1)))

        # Val Loss line:  Val Loss: 561.98 (loss_total: ..., o2m_cls: X, o2m_box: Y, ..., o2o_cls: A, o2o_box: B, ...) | mAP@0.5: M
        val_m = re.search(r"Val Loss:\s*([\d.]+)\s*\((.+?)\)\s*\|\s*mAP@0\.5:\s*([\d.]+)", line)
        if val_m:
            history["val_loss"].append(float(val_m.group(1)))
            history["mAP50"].append(float(val_m.group(3)))

            detail = val_m.group(2)
            # Extract loss_total as train loss
            lt = re.search(r"loss_total:\s*([\d.]+)", detail)
            if lt:
                while len(history["train_loss"]) < current_epoch:
                    history["train_loss"].append(None)
                if current_epoch > 0:
                    history["train_loss"][current_epoch - 1] = float(lt.group(1))

            # Extract sub-losses
            for key, pattern in [
                ("o2m_cls", r"o2m_cls:\s*([\d.]+)"),
                ("o2m_box", r"o2m_box:\s*([\d.]+)"),
                ("o2o_cls", r"o2o_cls:\s*([\d.]+)"),
                ("o2o_box", r"o2o_box:\s*([\d.]+)"),
            ]:
                sm = re.search(pattern, detail)
                while len(history[key]) < current_epoch:
                    history[key].append(None)
                if sm and current_epoch > 0:
                    history[key][current_epoch - 1] = float(sm.group(1))

        # Also try: Validation - Loss: X | mAP@0.5: Y (older format)
        if not val_m:
            alt_val = re.search(r"Validation.*Loss:\s*([\d.]+).*mAP@0\.5:\s*([\d.]+)", line)
            if alt_val:
                history["val_loss"].append(float(alt_val.group(1)))
                history["mAP50"].append(float(alt_val.group(2)))

        # Batch-level loss: Epoch [N] Batch [B/T] Loss: X (older format)
        batch_m = re.search(
            r"Epoch \[(\d+)\] Batch \[\d+/\d+\] Loss: ([\d.]+).*?"
            r"o2m_cls: ([\d.]+).*?o2m_box: ([\d.]+).*?o2o_cls: ([\d.]+).*?o2o_box: ([\d.]+)",
            line
        )
        if batch_m:
            ep = int(batch_m.group(1))
            while len(history["train_loss"]) < ep:
                history["train_loss"].append(None)
            history["train_loss"][ep - 1] = float(batch_m.group(2))
            for i, key in enumerate(["o2m_cls", "o2m_box", "o2o_cls", "o2o_box"], 3):
                while len(history[key]) < ep:
                    history[key].append(None)
                history[key][ep - 1] = float(batch_m.group(i))

        # Epoch time
        time_m = re.search(r"Epoch time:\s*([\d.]+)s", line)
        if time_m:
            history["epoch_time"].append(float(time_m.group(1)))

    # Fill in epochs list
    n = max(len(history["train_loss"]), len(history["lr"]),
            len(history["val_loss"]), len(history["mAP50"]), 1) - 1
    if n < 0:
        n = 0
    n = max(len(history["train_loss"]), len(history["lr"]),
            len(history["val_loss"]), len(history["mAP50"]))
    history["epochs"] = list(range(1, n + 1))

    if not history["epochs"]:
        return None
    return history


def _render_metrics_from_history(history, run_dir):
    """Render top metric cards from parsed history (CSV or log-based)."""
    cols = st.columns(7)

    if history:
        n_epochs = len(history["epochs"])
        total = history.get("total_epochs") or n_epochs

        with cols[0]:
            st.metric("Epoch", f"{n_epochs}/{total}")
        with cols[1]:
            losses = [x for x in history["train_loss"] if x is not None]
            val = f"{losses[-1]:.1f}" if losses else "—"
            st.metric("Train Loss", val)
        with cols[2]:
            val = f"{history['val_loss'][-1]:.2f}" if history.get("val_loss") else "—"
            st.metric("Val Loss", val)
        with cols[3]:
            val = f"{history['mAP50'][-1]:.4f}" if history.get("mAP50") else "—"
            st.metric("mAP@0.5", val)
        with cols[4]:
            best = f"{max(history['mAP50']):.4f}" if history.get("mAP50") else "—"
            st.metric("Best mAP", best)
        with cols[5]:
            lr_val = f"{history['lr'][-1]:.2e}" if history.get("lr") else "—"
            st.metric("LR", lr_val)
        with cols[6]:
            model = history.get("model_info", "")
            device = history.get("device", "")
            label = model or device or "—"
            st.metric("Model", label[:12])

        if n_epochs < total:
            st.progress(n_epochs / max(total, 1))
    else:
        for col in cols:
            with col:
                st.metric("—", "No data")


def _render_curves(history, run_dir):
    """Render training curves — from FlashDet's plots/ images and CSV data."""
    # FlashDet saves pre-rendered plots in plots/ directory
    plots_dir = os.path.join(run_dir, "plots")
    training_curves_img = os.path.join(plots_dir, "training_curves.png")
    map_curve_img = os.path.join(plots_dir, "mAP_curve.png")

    if os.path.isfile(training_curves_img) or os.path.isfile(map_curve_img):
        st.caption("FlashDet Generated Plots")
        cc1, cc2 = st.columns(2)
        if os.path.isfile(training_curves_img):
            with cc1:
                st.image(training_curves_img, caption="Training Curves", use_container_width=True)
        if os.path.isfile(map_curve_img):
            with cc2:
                st.image(map_curve_img, caption="mAP Curve", use_container_width=True)

    if not history or not history.get("train_loss"):
        if not os.path.isfile(training_curves_img):
            st.info("No training data yet.")
        return

    st.caption("Interactive Charts")
    epochs = history.get("epochs", [])
    losses = [x for x in history["train_loss"] if x is not None]
    epochs_for_loss = [epochs[i] if i < len(epochs) else i + 1
                       for i, x in enumerate(history["train_loss"]) if x is not None]

    # Determine sub-loss keys (CSV uses train_box/train_cls, log uses o2m_cls/o2m_box)
    has_csv_subloss = any(history.get("train_box", []))
    has_log_subloss = any(history.get("o2m_cls", []))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Total Loss", "mAP@0.5",
                        "Sub-Losses (box / cls / l1)" if has_csv_subloss else "Sub-Losses (o2m/o2o)",
                        "Learning Rate"),
    )

    fig.add_trace(
        go.Scatter(x=epochs_for_loss, y=losses, mode="lines+markers",
                   name="Train Loss", line=dict(color="#7C3AED", width=2), marker=dict(size=4)),
        row=1, col=1,
    )
    if history.get("val_loss"):
        val_epochs = list(range(1, len(history["val_loss"]) + 1))
        fig.add_trace(
            go.Scatter(x=val_epochs, y=history["val_loss"], mode="lines+markers",
                       name="Val Loss", line=dict(color="#F59E0B", width=2), marker=dict(size=4)),
            row=1, col=1,
        )

    if history.get("mAP50"):
        map_epochs = list(range(1, len(history["mAP50"]) + 1))
        fig.add_trace(
            go.Scatter(x=map_epochs, y=history["mAP50"], mode="lines+markers",
                       name="mAP@0.5", line=dict(color="#10B981", width=2), marker=dict(size=5)),
            row=1, col=2,
        )

    # Sub-losses from CSV (train_box, train_cls, train_l1)
    if has_csv_subloss:
        for key, color, dash in [
            ("train_cls", "#EF4444", None), ("train_box", "#3B82F6", None),
            ("train_l1", "#F97316", "dash"),
            ("val_cls", "#EF4444", "dot"), ("val_box", "#3B82F6", "dot"),
        ]:
            vals = [x for x in history.get(key, []) if x is not None]
            if vals:
                ep = list(range(1, len(vals) + 1))
                fig.add_trace(
                    go.Scatter(x=ep, y=vals, mode="lines", name=key,
                               line=dict(color=color, width=1.5, dash=dash)),
                    row=2, col=1,
                )
    elif has_log_subloss:
        for key, color, dash in [
            ("o2m_cls", "#EF4444", None), ("o2m_box", "#F97316", "dash"),
            ("o2o_cls", "#3B82F6", None), ("o2o_box", "#10B981", "dash"),
        ]:
            vals = [x for x in history.get(key, []) if x is not None]
            if vals:
                ep = list(range(1, len(vals) + 1))
                fig.add_trace(
                    go.Scatter(x=ep, y=vals, mode="lines", name=key,
                               line=dict(color=color, width=1.5, dash=dash)),
                    row=2, col=1,
                )

    if history.get("lr"):
        lr_epochs = list(range(1, len(history["lr"]) + 1))
        fig.add_trace(
            go.Scatter(x=lr_epochs, y=history["lr"], mode="lines",
                       name="LR", line=dict(color="#6366F1", width=2)),
            row=2, col=2,
        )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=30, r=10, t=30, b=25),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_visualizations(run_dir):
    """Show visualization images from FlashDet output — checks multiple possible locations."""
    img_dirs = [
        os.path.join(run_dir, "visualizations"),
        os.path.join(run_dir, "plots"),
        os.path.join(run_dir, "vis"),
    ]
    all_images = []
    for d in img_dirs:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and f != "latest_visualization.jpg":
                    all_images.append(os.path.join(d, f))

    # Also check for any .png/.jpg directly in run_dir (FlashDet may save plots here)
    for f in sorted(os.listdir(run_dir)):
        fp = os.path.join(run_dir, f)
        if os.path.isfile(fp) and f.lower().endswith((".png", ".jpg", ".jpeg")):
            all_images.append(fp)

    if not all_images:
        st.info("No visualization images found in this run. FlashDet saves plots in the `plots/` directory when available.")
        # Show checkpoint summary as alternative
        pth_files = [f for f in os.listdir(run_dir) if f.endswith(".pth")]
        if pth_files:
            st.caption(f"Checkpoints found: {len(pth_files)}")
            for f in sorted(pth_files):
                sz = os.path.getsize(os.path.join(run_dir, f))
                st.caption(f"  {f} ({sz / 1024 / 1024:.1f} MB)")
        return

    # Show latest first
    latest = os.path.join(run_dir, "visualizations", "latest_visualization.jpg")
    if os.path.isfile(latest):
        st.image(latest, caption="Latest Visualization", use_container_width=True)

    st.caption(f"{len(all_images)} images")
    for i in range(0, len(all_images), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(all_images):
                with col:
                    st.image(all_images[idx], caption=os.path.basename(all_images[idx])[:25],
                             use_container_width=True)


def _render_gt_verification(run_dir):
    """GT verification — from dedicated directory, or extracted from log file."""
    gt_dir = os.path.join(run_dir, "gt_verification")

    # If dedicated GT directory exists, show it
    if os.path.isdir(gt_dir):
        report_path = os.path.join(gt_dir, "verification_report.json")
        summary_path = os.path.join(gt_dir, "verification_summary.txt")

        if os.path.isfile(report_path):
            with open(report_path) as f:
                report = json.load(f)
            st.success("Verification: PASSED") if report.get("passed") else st.error("Verification: FAILED")

            tc = report.get("splits", {}).get("train", {}).get("coco", {})
            vc = report.get("splits", {}).get("val", {}).get("coco", {})
            td = report.get("splits", {}).get("train", {}).get("dataloader", {})

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.metric("Train Imgs", tc.get("num_images", 0))
            with m2:
                st.metric("Train Ann", tc.get("num_annotations", 0))
            with m3:
                st.metric("Val Imgs", vc.get("num_images", 0))
            with m4:
                st.metric("Val Ann", vc.get("num_annotations", 0))
            with m5:
                st.metric("Classes", report.get("num_classes", "?"))
            with m6:
                st.metric("Avg Boxes/Img", f"{td.get('avg_boxes_per_sample', 0):.1f}")

        if os.path.isfile(summary_path):
            with st.expander("Verification Summary", expanded=False):
                with open(summary_path) as f:
                    st.code(f.read(), language="text")

        raw_dir = os.path.join(gt_dir, "images", "raw")
        dl_dir = os.path.join(gt_dir, "images", "dataloader")
        gt_tab_raw, gt_tab_dl = st.tabs(["Raw GT Images", "Dataloader GT Images"])
        with gt_tab_raw:
            _render_image_grid(raw_dir, "Raw ground truth (before augmentations)")
        with gt_tab_dl:
            _render_image_grid(dl_dir, "After dataloader transforms & augmentations")
        return

    # No dedicated directory — extract verification info from the log file
    log_file = _find_log_file(run_dir)
    if not log_file:
        st.info("No GT verification data yet. Start training to generate.")
        return

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        log_content = f.read()

    # Extract dataset verification block from log
    ver_start = log_content.find("Dataset Verification")
    if ver_start == -1:
        st.info("No dataset verification found in log.")
        return

    ver_end = log_content.find("Starting training", ver_start)
    if ver_end == -1:
        ver_end = ver_start + 800
    ver_block = log_content[ver_start:ver_end].strip()

    # Parse verification info
    train_imgs = re.search(r"train.*?Images:\s*(\d+)", ver_block, re.DOTALL)
    train_ann = re.search(r"train.*?Annotations:\s*(\d+)", ver_block, re.DOTALL)
    train_found = re.search(r"train.*?Files found:\s*(\d+)/(\d+)", ver_block, re.DOTALL)
    val_imgs = re.search(r"valid.*?Images:\s*(\d+)", ver_block, re.DOTALL)
    val_ann = re.search(r"valid.*?Annotations:\s*(\d+)", ver_block, re.DOTALL)

    # Extract classes from header
    cls_m = re.search(r"Classes \((\d+)\): \[(.+?)\]", log_content)
    num_cls = int(cls_m.group(1)) if cls_m else "?"
    class_names = [c.strip().strip("'") for c in cls_m.group(2).split(",")] if cls_m else []

    # Check if all passed
    train_ok = "✓ train" in ver_block
    val_ok = "✓ valid" in ver_block
    if train_ok and val_ok:
        st.success("Dataset Verification: PASSED")
    elif train_ok:
        st.warning("Dataset Verification: Train OK, Val missing")
    else:
        st.error("Dataset Verification: FAILED")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Train Imgs", train_imgs.group(1) if train_imgs else "—")
    with m2:
        st.metric("Train Ann", train_ann.group(1) if train_ann else "—")
    with m3:
        st.metric("Val Imgs", val_imgs.group(1) if val_imgs else "—")
    with m4:
        st.metric("Val Ann", val_ann.group(1) if val_ann else "—")
    with m5:
        st.metric("Classes", num_cls)
    with m6:
        if train_found:
            st.metric("Files OK", f"{train_found.group(1)}/{train_found.group(2)}")
        else:
            st.metric("Files OK", "—")

    if class_names:
        with st.expander(f"Class Names ({len(class_names)})", expanded=False):
            st.code("\n".join(class_names), language="text")

    with st.expander("Verification Log", expanded=False):
        st.code(ver_block, language="text")


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
    """Log viewer with refresh and error detection."""
    if not log_path or not os.path.isfile(log_path):
        st.info("No log found.")
        return

    with open(log_path) as f:
        content = f.read()
    lines = content.strip().split("\n") if content.strip() else []

    # Detect errors in log output
    has_error = False
    error_lines = []
    for line in lines:
        if any(kw in line for kw in ("Traceback", "Error:", "Exception:", "FAILED", "TypeError", "ValueError", "RuntimeError")):
            has_error = True
        if has_error:
            error_lines.append(line)

    lc1, lc2, lc3 = st.columns([4, 1, 1])
    with lc1:
        st.caption(f"`{os.path.basename(log_path)}` — {len(lines)} lines")
    with lc2:
        show_all = st.checkbox("Full", value=False, key="show_full_log")
    with lc3:
        if st.button("Refresh", key="refresh_log_btn", use_container_width=True):
            st.rerun()

    if has_error and error_lines:
        st.error("Training failed with error:")
        st.code("\n".join(error_lines[-20:]), language="python")

    if lines:
        st.code("\n".join(lines if show_all else lines[-30:]), language="bash")
    else:
        st.warning("Log file is empty. Training may still be starting...")


def _render_checkpoints(run_dir):
    """Files tab — list all artifacts from a FlashDet training run."""
    files_info = []
    EXTENSIONS = (".pth", ".json", ".csv", ".log", ".onnx", ".onnx.data", ".txt")
    for f in sorted(os.listdir(run_dir)):
        fpath = os.path.join(run_dir, f)
        if os.path.isfile(fpath) and (f.endswith(EXTENSIONS) or f in ("model.onnx", "model.onnx.data")):
            size = os.path.getsize(fpath)
            if size > 1_073_741_824:
                size_str = f"{size / (1024**3):.1f}GB"
            elif size > 1_048_576:
                size_str = f"{size / (1024*1024):.1f}MB"
            else:
                size_str = f"{size / 1024:.0f}KB"
            files_info.append({"File": f, "Size": size_str, "Type": _file_type(f)})

    # Also list subdirectories with their file counts
    for d in sorted(os.listdir(run_dir)):
        dpath = os.path.join(run_dir, d)
        if os.path.isdir(dpath):
            n_files = sum(1 for _ in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, _)))
            files_info.append({"File": f"{d}/", "Size": f"{n_files} files", "Type": "Directory"})

    if files_info:
        st.dataframe(files_info, use_container_width=True, hide_index=True,
                      height=min(35 * len(files_info) + 40, 400))
    else:
        st.info("No files.")

    results_path = os.path.join(run_dir, "results.json")
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
        st.caption("Training Results")
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        with r1:
            st.metric("Architecture", results.get("architecture", "—"))
        with r2:
            st.metric("Epochs", f"{results.get('epochs_trained', '?')}/{results.get('total_epochs', '?')}")
        with r3:
            st.metric("Best mAP@50", f"{results.get('best_mAP50', 0):.4f}")
        with r4:
            st.metric("Best Val Loss", f"{results.get('best_val_loss', 0):.2f}")
        with r5:
            st.metric("Params", f"{results.get('model_params_M', 0):.2f}M")
        with r6:
            st.metric("Input Size", results.get("input_size", "?"))

        tc = results.get("training_config", {})
        if tc:
            with st.expander("Training Config", expanded=False):
                config_cols = st.columns(4)
                items = list(tc.items())
                for i, (k, v) in enumerate(items):
                    with config_cols[i % 4]:
                        st.text(f"{k}: {v}")


def _file_type(filename):
    """Categorize a file by its name — aligned with FlashDet output."""
    if "final" in filename and "inference" in filename:
        return "Final inference weights"
    if "final" in filename and "fp16" in filename:
        return "Final FP16 weights"
    if "best" in filename and "inference" in filename:
        return "Best inference weights"
    if "best" in filename:
        return "Best checkpoint"
    if "last" in filename and "inference" in filename:
        return "Latest inference weights"
    if "last" in filename and "fp16" in filename:
        return "Latest FP16 weights"
    if "checkpoint_last" in filename:
        return "Latest checkpoint (full)"
    if "inference" in filename:
        return "Inference weights"
    if "fp16" in filename:
        return "FP16 weights"
    if filename == "model.onnx":
        return "ONNX model"
    if filename == "model.onnx.data":
        return "ONNX weights data"
    if filename == "results.json":
        return "Training results"
    if filename.endswith(".json"):
        return "Report/Config"
    if filename.endswith(".csv"):
        return "Training metrics CSV"
    if filename.endswith(".log"):
        return "Training log"
    if filename.endswith(".txt"):
        return "Summary"
    return "Other"


# ---- Start New Training Tab ----

def _render_start_tab():
    """Tab: launch a new training run — professional step-based layout."""

    # ─── STEP 1: Experiment Folder ───
    # runs_root is always the workspace root — never a run subfolder
    runs_root = os.path.join(os.getcwd(), "workspace")

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
                from flashstudio.utils import flash
                folder_path = os.path.join(runs_root, new_folder_name)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path, exist_ok=True)
                    st.session_state["run_name"] = new_folder_name
                    st.session_state["active_workflow_folder"] = new_folder_name
                    flash(f"Folder `{new_folder_name}` created", "success")
                    st.rerun()
                else:
                    flash(f"Folder `{new_folder_name}` already exists", "warning")
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
                        from flashstudio.utils import flash
                        st.session_state["active_workflow_folder"] = existing_folders[sel_idx]
                        st.session_state["run_name"] = existing_folders[sel_idx]
                        flash(f"Using folder `{existing_folders[sel_idx]}`", "success")
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
    full_run_path = os.path.join(runs_root, run_name)

    # Config overview — single row
    nc = st.session_state.get("num_classes", 0)
    with st.container(border=True):
        st.markdown("#### 2. Config")
        cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8 = st.columns(8)
        with cc1:
            st.metric("Model", st.session_state.get("model_arch", "Pico").replace("FlashDet-", ""))
        with cc2:
            st.metric("Dataset", (st.session_state.get("dataset_name") or "—")[:8])
        with cc3:
            st.metric("Classes", nc if nc else "—")
        with cc4:
            st.metric("Epochs", st.session_state.get("epochs", 100))
        with cc5:
            st.metric("Batch", st.session_state.get("batch_size", 16))
        with cc6:
            st.metric("LR", f"{st.session_state.get('lr', 0.001):.1e}")
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
    from flashstudio.utils import flash

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
                            flash(f"Rename failed: `{rename_val}` already exists", "error")
                        else:
                            os.rename(old_path, new_path)
                            if st.session_state.get("active_workflow_folder") == target:
                                st.session_state["active_workflow_folder"] = rename_val
                                st.session_state["run_name"] = rename_val
                            flash(f"Renamed `{target}` → `{rename_val}`", "success")
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
                    flash(f"Deleted `{target}`", "success")
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

    # 3. Classes
    nc = st.session_state.get("num_classes", 0)
    cls_names = st.session_state.get("class_names", "")
    if nc and nc > 0:
        checks.append(("Classes", True, f"{nc} classes"))
    elif cls_names.strip():
        n = len([c for c in cls_names.strip().split("\n") if c.strip()])
        checks.append(("Classes", True, f"{n} classes"))
    else:
        checks.append(("Classes", False, "No classes — go to Data → Upload"))

    # 4. Model config
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
        parent_dir = os.path.join(os.getcwd(), "workspace")

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
        parent_dir = os.path.join(os.getcwd(), "workspace")
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
    """Remove non-essential files from a run, keeping final/best models and results."""
    keep = {
        "checkpoint_best.pth", "results.json", "training_log.csv",
        "model_best_inference.pth", "model_best_fp16.pth",
        "model_final_inference.pth", "model_final_fp16.pth",
        "model.onnx", "model.onnx.data",
    }
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
    from flashstudio.utils import flash

    pid = st.session_state.get("training_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            flash(f"Training stopped (PID {pid})", "success")
        except ProcessLookupError:
            flash("Training process already finished", "info")
        except PermissionError:
            try:
                os.kill(pid, signal.SIGKILL)
                flash(f"Training force-killed (PID {pid})", "warning")
            except Exception as e:
                flash(f"Could not kill process {pid}: {e}", "error")
        except Exception as e:
            flash(f"Error stopping training: {e}", "error")
    else:
        flash("No training process to stop", "warning")

    st.session_state["training_active"] = False
    st.session_state["training_status"] = "Stopped"
    st.session_state["training_paused"] = False
    st.session_state["training_pid"] = None
    st.rerun()


def _pause_training():
    """Pause the training subprocess by sending SIGSTOP."""
    import signal
    from flashstudio.utils import flash

    pid = st.session_state.get("training_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGSTOP)
            st.session_state["training_paused"] = True
            st.session_state["training_status"] = "Paused"
            flash(f"Training paused (PID {pid}). GPU memory still held.", "success")
        except ProcessLookupError:
            flash("Training process already finished", "info")
            st.session_state["training_active"] = False
            st.session_state["training_paused"] = False
        except Exception as e:
            flash(f"Could not pause training: {e}", "error")
    else:
        flash("No training process found to pause", "warning")
    st.rerun()


def _resume_active_training():
    """Resume a paused training subprocess by sending SIGCONT."""
    import signal
    from flashstudio.utils import flash

    pid = st.session_state.get("training_pid")
    if pid:
        try:
            os.kill(pid, signal.SIGCONT)
            st.session_state["training_paused"] = False
            st.session_state["training_status"] = "Running"
            flash(f"Training resumed (PID {pid})", "success")
        except ProcessLookupError:
            flash("Training process already finished", "info")
            st.session_state["training_active"] = False
            st.session_state["training_paused"] = False
        except Exception as e:
            flash(f"Could not resume training: {e}", "error")
    else:
        flash("No training process found to resume", "warning")
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

    # Convert to absolute paths to avoid relative path issues
    if train_images:
        train_images = os.path.abspath(train_images)
    if val_images:
        val_images = os.path.abspath(val_images)

    # Auto-find val path if missing but train exists
    if train_images and os.path.isdir(train_images) and (not val_images or not os.path.isdir(val_images)):
        parent = os.path.dirname(train_images)
        for vname in ("valid", "val", "test"):
            vpath = os.path.join(parent, vname)
            if os.path.isdir(vpath):
                val_images = vpath
                break
        if not val_images or not os.path.isdir(val_images):
            val_images = train_images

    if not train_images:
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "No dataset"
        st.session_state["training_error"] = (
            "No dataset paths configured. Go to the Data page and either "
            "enter train/val image paths manually, or download a dataset first."
        )
        return

    # Auto-detect classes from annotation if not already set
    if not st.session_state.get("class_names", "").strip():
        ann_file = os.path.join(train_images, "_annotations.coco.json")
        if not os.path.isfile(ann_file):
            json_files = [f for f in os.listdir(train_images) if f.endswith(".json")] if os.path.isdir(train_images) else []
            if json_files:
                ann_file = os.path.join(train_images, json_files[0])
        if os.path.isfile(ann_file):
            try:
                with open(ann_file, encoding="utf-8") as _af:
                    ann_data = json.load(_af)
                cats = ann_data.get("categories", [])
                if cats:
                    sorted_cats = sorted(cats, key=lambda c: c.get("id", 0))
                    names = [c["name"] for c in sorted_cats]
                    st.session_state["class_names"] = "\n".join(names)
                    st.session_state["num_classes"] = len(names)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

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
            st.session_state["training_active"] = False
            st.session_state["training_status"] = "Conversion failed"
            st.session_state["training_error"] = f"Format conversion failed: {e}"
            return

    # Verify COCO annotation file exists
    train_ann = os.path.join(train_images, "_annotations.coco.json")
    if not os.path.isfile(train_ann):
        ann_candidates = [f for f in os.listdir(train_images) if f.endswith(".json")] if os.path.isdir(train_images) else []
        if ann_candidates:
            st.info(f"Found annotation file: `{ann_candidates[0]}` (expected `_annotations.coco.json`)")
        else:
            st.session_state["training_active"] = False
            st.session_state["training_status"] = "Missing annotations"
            st.session_state["training_error"] = (
                f"Missing annotations! FlashDet expects _annotations.coco.json in: {train_images}"
            )
            return

    # Normalize dataset layout for FlashDet:
    # FlashDet computes data_root = dirname(train_images) and expects
    # data_root/valid/_annotations.coco.json to exist.
    data_root = os.path.dirname(os.path.normpath(train_images))
    valid_dir_in_root = os.path.join(data_root, "valid")
    val_ann_check = os.path.join(valid_dir_in_root, "_annotations.coco.json")

    if not os.path.exists(val_ann_check) and val_images and os.path.isdir(val_images):
        val_ann_file = os.path.join(val_images, "_annotations.coco.json")
        if os.path.isfile(val_ann_file):
            try:
                if os.path.exists(valid_dir_in_root):
                    if os.path.islink(valid_dir_in_root):
                        os.unlink(valid_dir_in_root)
                    else:
                        pass
                if not os.path.exists(valid_dir_in_root):
                    os.symlink(os.path.abspath(val_images), valid_dir_in_root)
                    val_images = valid_dir_in_root
                    st.session_state["val_img_path"] = val_images
            except OSError:
                pass

    if not os.path.exists(val_ann_check):
        try:
            os.makedirs(valid_dir_in_root, exist_ok=True)
            import shutil
            shutil.copytree(train_images, valid_dir_in_root, dirs_exist_ok=True)
            val_images = valid_dir_in_root
            st.session_state["val_img_path"] = val_images
        except Exception:
            pass

    st.session_state["train_img_path"] = train_images
    st.session_state["val_img_path"] = val_images

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
    save_dir = st.session_state.get("save_dir", os.path.join(os.getcwd(), "workspace", "flashdet_output"))

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

    # Map session state to Trainer API (only parameters accepted by flashdet.Trainer)
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
    grad_accum = st.session_state.get("grad_accum", 1)
    patience = st.session_state.get("patience", 50)
    input_size = st.session_state.get("img_size", 320)
    activation_checkpointing = st.session_state.get("activation_checkpointing", False)
    activation_offloading = st.session_state.get("activation_offloading", False)
    optimizer_in_bwd = st.session_state.get("optimizer_in_bwd", False)
    use_8bit_optimizer = st.session_state.get("use_8bit_optimizer", False)
    compile_model = st.session_state.get("compile_model", False)
    multi_gpu = st.session_state.get("ddp", False)

    # LoRA settings
    finetune_strategy = st.session_state.get("finetune_strategy", "Full fine-tune")
    lora = finetune_strategy == "LoRA"
    lora_rank = st.session_state.get("lora_rank", 8)
    lora_alpha = st.session_state.get("lora_alpha", 16.0)
    lora_dropout = st.session_state.get("lora_dropout", 0.05)
    lora_variant = st.session_state.get("lora_variant", "standard")
    lora_targets = st.session_state.get("lora_targets", ["backbone", "fpn"])

    # QLoRA
    qlora = st.session_state.get("qlora", False)
    qlora_dtype = st.session_state.get("qlora_dtype", "int8")

    # Chunked loss
    chunked_loss = st.session_state.get("chunked_loss", False)
    chunk_size = st.session_state.get("chunk_size", 1024)

    # Pretrained / resume
    pretrain_option = st.session_state.get("pretrain_option", "COCO pretrained")
    pretrained_ckpt = None
    if pretrain_option == "Custom":
        pretrained_ckpt = st.session_state.get("custom_weights", "")

    resume_ckpt = None
    if st.session_state.get("resume_training", False):
        resume_ckpt = st.session_state.get("resume_path", "")

    # Class file
    class_file = st.session_state.get("class_file", None)
    class_names_raw = st.session_state.get("class_names", "")
    if class_names_raw.strip() and not class_file:
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

    os.makedirs(save_dir, exist_ok=True)

    import time as _time
    log_filename = f"train_{_time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(save_dir, log_filename)

    log_file_handle = open(log_path, "w")
    process = subprocess.Popen(
        [sys.executable, "-c", train_script],
        stdout=log_file_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    st.session_state["training_pid"] = process.pid
    st.session_state["training_log_file"] = log_path

    # Wait briefly to catch immediate crashes (import errors, bad args, etc.)
    import time as _tw
    _tw.sleep(2)
    exit_code = process.poll()
    if exit_code is not None:
        log_file_handle.close()
        st.session_state["training_active"] = False
        st.session_state["training_pid"] = None
        st.session_state["training_status"] = "Failed"
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as _lf:
                log_content = _lf.read()
        except OSError:
            log_content = "(could not read log)"
        from flashstudio.utils import flash
        flash("Training failed to start — see error below", "error")
        st.session_state["training_error"] = log_content[-2000:] if len(log_content) > 2000 else log_content
        st.rerun()
        return

    from flashstudio.utils import flash
    flash(f"Training started (PID {process.pid})", "success")
    st.rerun()
