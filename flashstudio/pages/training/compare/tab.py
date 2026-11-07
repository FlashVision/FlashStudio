"""Training — compare multiple runs."""

import os

import plotly.graph_objects as go
import streamlit as st

from flashstudio.constants import TRAINING_LOG_CSV, TRAINING_LOG_GLOB
from flashstudio.pages.training._common import _get_save_dir
from flashstudio.pages.training.monitor.parsers import (
    _parse_training_csv, _parse_training_log, _find_log_file,
)
from flashstudio.pages.training.monitor.run_meta import _get_run_meta


def _is_run(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return (
        TRAINING_LOG_CSV in entries
        or any(e.startswith("train_") and e.endswith(".log") for e in entries)
        or any(e.endswith(".pth") or e.endswith(".pt") for e in entries)
    )


def _run_dirs(workspace: str) -> list:
    if not os.path.isdir(workspace):
        return []
    found = []
    try:
        entries = sorted(os.listdir(workspace))
    except OSError:
        return found
    for name in entries:
        path = os.path.join(workspace, name)
        if _is_run(path):
            found.append((name, path))
    def _mtime(item):
        try:
            return os.path.getmtime(item[1])
        except OSError:
            return 0.0
    found.sort(key=_mtime, reverse=True)
    return found


def _history(run_dir: str):
    history = _parse_training_csv(run_dir)
    if not history:
        log = _find_log_file(run_dir)
        history = _parse_training_log(log) if log else None
    return history


def _render_compare_tab():
    workspace = _get_save_dir()
    runs = _run_dirs(workspace)
    if not runs:
        st.info("No runs found in the workspace.")
        return

    labels = {name: path for name, path in runs}
    selected = st.multiselect(
        "Runs to compare",
        list(labels.keys()),
        default=list(labels.keys())[: min(4, len(labels))],
        key="compare_runs",
    )
    if not selected:
        return

    rows = []
    histories = {}
    for name in selected:
        path = labels[name]
        meta = _get_run_meta(path)
        history = _history(path)
        histories[name] = history
        best_map = meta.get("mAP")
        if best_map is None and history and history.get("mAP50"):
            vals = [v for v in history["mAP50"] if v is not None]
            best_map = max(vals) if vals else None
        try:
            epochs_value = int(meta.get("epochs"))
        except (TypeError, ValueError):
            epochs_value = None
        rows.append({
            "Run": name,
            "Status": meta.get("status", "—"),
            "Model": meta.get("model", "—"),
            "Epochs": epochs_value,
            "Best mAP@0.5": round(float(best_map), 4) if best_map is not None else None,
            "Size": meta.get("size", "—"),
        })
    st.dataframe(
        rows, use_container_width=True, hide_index=True,
        column_config={
            "Epochs": st.column_config.NumberColumn("Epochs", format="%d"),
            "Best mAP@0.5": st.column_config.NumberColumn(
                "Best mAP@0.5", format="%.4f"
            ),
        },
    )

    map_fig = go.Figure()
    loss_fig = go.Figure()
    for name in selected:
        history = histories.get(name)
        if not history:
            continue
        epochs = history.get("epochs", [])
        mAP = history.get("mAP50", [])
        val_epochs = history.get("val_epochs", list(range(1, len(mAP) + 1)))
        if mAP:
            map_fig.add_trace(go.Scatter(
                x=val_epochs, y=mAP, mode="lines+markers", name=name,
            ))
        train_loss = history.get("train_loss", [])
        if train_loss:
            loss_fig.add_trace(go.Scatter(
                x=epochs[: len(train_loss)], y=train_loss, mode="lines", name=f"{name} · train",
            ))
        val_loss = history.get("val_loss", [])
        if val_loss:
            loss_fig.add_trace(go.Scatter(
                x=val_epochs[: len(val_loss)], y=val_loss, mode="lines", name=f"{name} · validation",
            ))

    map_fig.update_layout(
        title="mAP@0.5", template="plotly_white", height=360,
        margin=dict(l=30, r=10, t=40, b=25), legend=dict(orientation="h"),
    )
    loss_fig.update_layout(
        title="Loss", template="plotly_white", height=360,
        margin=dict(l=30, r=10, t=40, b=25), legend=dict(orientation="h"),
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(map_fig, use_container_width=True)
    with right:
        st.plotly_chart(loss_fig, use_container_width=True)
