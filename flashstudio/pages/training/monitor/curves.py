"""Training curves rendering."""

import os
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _render_curves(history, run_dir):
    """Render training curves — from FlashDet's plots/ images and CSV data."""
    from flashstudio.constants import VIS_DIR_NAMES
    plots_dir = os.path.join(run_dir, VIS_DIR_NAMES[1])
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
    has_csv_subloss = any(x for x in history.get("train_box", []) if x is not None)
    has_log_subloss = any(x for x in history.get("o2m_cls", []) if x is not None)

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
        ve = history.get("val_epochs", list(range(1, len(history["val_loss"]) + 1)))
        fig.add_trace(
            go.Scatter(x=ve, y=history["val_loss"], mode="lines+markers",
                       name="Val Loss", line=dict(color="#F59E0B", width=2), marker=dict(size=4)),
            row=1, col=1,
        )

    if history.get("mAP50"):
        ve = history.get("val_epochs", list(range(1, len(history["mAP50"]) + 1)))
        fig.add_trace(
            go.Scatter(x=ve, y=history["mAP50"], mode="lines+markers",
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
