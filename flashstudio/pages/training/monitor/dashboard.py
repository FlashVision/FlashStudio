"""Run dashboard — main display panel for a single training run."""

import os
import streamlit as st

from flashstudio.pages.training.monitor.parsers import _find_log_file, _parse_training_csv, _parse_training_log
from flashstudio.pages.training.monitor.curves import _render_curves
from flashstudio.pages.training.monitor.visualizations import _render_visualizations
from flashstudio.pages.training.monitor.gt_verification import _render_gt_verification
from flashstudio.pages.training.monitor.log_viewer import _render_full_log
from flashstudio.pages.training.monitor.checkpoints import _render_checkpoints
from flashstudio.pages.training.progress import _render_training_progress


def _load_run_data(run_dir):
    """Return ``(log_file, history)`` for a run directory.

    CSV is the primary source (epoch averages), but it does not contain the
    objective `loss_total` or the o2m/o2o sub-losses. Those only exist in the
    training log, so merge the missing series from the log into the CSV data.
    """
    log_file = None
    history = None
    if run_dir and os.path.isdir(run_dir):
        log_file = _find_log_file(run_dir)
        history = _parse_training_csv(run_dir)
        log_history = _parse_training_log(log_file) if log_file else None
        if history and log_history:
            for key in (
                "train_loss_total", "o2m_cls", "o2m_box",
                "o2o_cls", "o2o_box",
            ):
                current = history.get(key) or []
                if not any(v is not None for v in current):
                    values = log_history.get(key) or []
                    if any(v is not None for v in values):
                        history[key] = values
        elif not history:
            history = log_history
    return log_file, history


def _render_run_dashboard(run_dir, show_summary: bool = True):
    """Render full dashboard for a single training run — always shows all 5 sub-tabs.

    ``show_summary`` controls the run-level status/metric cards above the tabs;
    the Launch tab already shows them, so Display can skip them to save space.
    """
    log_file, history = _load_run_data(run_dir)

    if run_dir and os.path.isdir(run_dir):
        if show_summary:
            # Live progress (works mid-epoch, unlike the per-epoch history).
            _render_training_progress(run_dir, show_eta=False)
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
            st.info("No ground-truth verification data yet.")

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
    """Render metric cards from parsed history (CSV or log-based).

    Two rows of three keep long labels (e.g. "Validation Loss") readable at
    normal widths; the Model/Device info already lives in the run meta cards.
    """
    row1 = st.columns(3)
    row2 = st.columns(3)

    if history:
        n_epochs = len(history["epochs"])
        total = history.get("total_epochs") or n_epochs
        train_series = history.get("train_loss_total") or history["train_loss"]
        losses = [x for x in train_series if x is not None]
        vl = [x for x in history.get("val_loss", []) if x is not None]
        mp = [x for x in history.get("mAP50", []) if x is not None]
        lr_values = [x for x in history.get("lr", []) if x is not None]

        with row1[0]:
            st.metric("Epoch", f"{n_epochs}/{total}")
        with row1[1]:
            st.metric("Training Loss", f"{losses[-1]:.1f}" if losses else "—")
        with row1[2]:
            st.metric("Validation Loss", f"{vl[-1]:.2f}" if vl else "—")
        with row2[0]:
            st.metric("mAP@0.5", f"{mp[-1]:.4f}" if mp else "—")
        with row2[1]:
            st.metric("Best mAP", f"{max(mp):.4f}" if mp else "—")
        with row2[2]:
            st.metric("LR", f"{lr_values[-1]:.2e}" if lr_values else "—")

    else:
        for col in [*row1, *row2]:
            with col:
                st.metric("—", "No data")
