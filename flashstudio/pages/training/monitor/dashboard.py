"""Run dashboard — main display panel for a single training run."""

import os
import streamlit as st

from flashstudio.pages.training.monitor.parsers import _find_log_file, _parse_training_csv, _parse_training_log
from flashstudio.pages.training.monitor.curves import _render_curves
from flashstudio.pages.training.monitor.visualizations import _render_visualizations
from flashstudio.pages.training.monitor.gt_verification import _render_gt_verification
from flashstudio.pages.training.monitor.log_viewer import _render_full_log
from flashstudio.pages.training.monitor.checkpoints import _render_checkpoints


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
                if "Traceback" in log_content:
                    all_lines = log_content.strip().split("\n")
                    tb_idx = -1
                    for idx_l, ln in enumerate(all_lines):
                        if "Traceback" in ln:
                            tb_idx = idx_l
                    if tb_idx >= 0:
                        error_summary = "\n".join(all_lines[tb_idx:][-10:])
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
            vl = [x for x in history.get("val_loss", []) if x is not None]
            val = f"{vl[-1]:.2f}" if vl else "—"
            st.metric("Val Loss", val)
        with cols[3]:
            mp = [x for x in history.get("mAP50", []) if x is not None]
            val = f"{mp[-1]:.4f}" if mp else "—"
            st.metric("mAP@0.5", val)
        with cols[4]:
            best = f"{max(mp):.4f}" if mp else "—"
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
            epoch_times = history.get("epoch_time", [])
            if epoch_times:
                avg_time = sum(epoch_times) / len(epoch_times)
                remaining = (total - n_epochs) * avg_time
                if remaining > 3600:
                    eta = f"~{remaining / 3600:.1f}h"
                elif remaining > 60:
                    eta = f"~{remaining / 60:.0f}m"
                else:
                    eta = f"~{remaining:.0f}s"
                st.caption(f"ETA: {eta} ({avg_time:.1f}s/epoch)")
    else:
        for col in cols:
            with col:
                st.metric("—", "No data")
