"""Training curves rendering."""

import os

import plotly.graph_objects as go
import streamlit as st


_COLORS = {
    "train_loss": "#7C3AED",
    "val_loss": "#F59E0B",
    "map": "#10B981",
    "lr": "#6366F1",
    "cls": "#EF4444",
    "box": "#3B82F6",
    "l1": "#F97316",
}

_PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _style(fig, title, y_title, *, log_y=False):
    """Apply the shared, compact chart style."""
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=13)),
        template="plotly_white",
        height=280,
        margin=dict(l=10, r=10, t=44, b=10),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, x=0,
            font=dict(size=10), bgcolor="rgba(255,255,255,.7)",
        ),
        font=dict(size=11, color="#374151"),
        xaxis=dict(
            title="Epoch", showgrid=True, gridcolor="#EEF0F6",
            zeroline=False, tickformat="d", automargin=True,
        ),
        yaxis=dict(
            title=y_title, showgrid=True, gridcolor="#EEF0F6",
            zeroline=False, type="log" if log_y else "linear",
            automargin=True,
        ),
    )
    return fig


def _pairs(xs, values):
    return [(x, v) for x, v in zip(xs, values) if v is not None]


def _add_line(fig, xs, values, name, color, *, dash=None, markers=True):
    pairs = _pairs(xs, values)
    if not pairs:
        return
    mode = "lines+markers" if markers and len(pairs) <= 60 else "lines"
    fig.add_trace(go.Scatter(
        x=[p[0] for p in pairs],
        y=[p[1] for p in pairs],
        mode=mode,
        name=name,
        line=dict(color=color, width=2, dash=dash),
        marker=dict(size=4),
        hovertemplate="Epoch %{x} · %{y:.4g}<extra>" + name + "</extra>",
    ))


def _log_tickvals(values, max_ticks=5):
    """Pick a readable subset of 1-2-5 log ticks for the given range."""
    values = [v for v in values if v is not None and v > 0]
    if not values:
        return None
    lo, hi = min(values), max(values)
    base = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    ticks = [t for t in base if lo * 0.75 <= t <= hi * 1.25]
    if len(ticks) > max_ticks:
        ticks = [
            ticks[round(i * (len(ticks) - 1) / (max_ticks - 1))]
            for i in range(max_ticks)
        ]
    return ticks or None


def _empty_chart(fig, message):
    """Show a centered placeholder and hide the empty axes."""
    fig.add_annotation(
        text=message, showarrow=False, x=0.5, y=0.5,
        xref="paper", yref="paper",
        font=dict(size=11, color="#9CA3AF"),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)


def _render_curves(history, run_dir):
    """Render training curves — from FlashDet's plots/ images and CSV data."""
    from flashstudio.constants import VIS_DIR_NAMES

    plots_dir = os.path.join(run_dir, VIS_DIR_NAMES[1])
    training_curves_img = os.path.join(plots_dir, "training_curves.png")
    map_curve_img = os.path.join(plots_dir, "mAP_curve.png")
    has_static = os.path.isfile(training_curves_img) or os.path.isfile(map_curve_img)

    if not history or not any(v is not None for v in history.get("train_loss", [])):
        if has_static:
            with st.expander("FlashDet-generated plots", expanded=True):
                cc1, cc2 = st.columns(2)
                if os.path.isfile(training_curves_img):
                    with cc1:
                        st.image(training_curves_img, caption="Training curves",
                                 use_container_width=True)
                if os.path.isfile(map_curve_img):
                    with cc2:
                        st.image(map_curve_img, caption="mAP curve",
                                 use_container_width=True)
        else:
            st.info("No training data yet.")
        return

    epochs = history.get("epochs", [])
    val_epochs = history.get("val_epochs") or []
    # Prefer the objective `loss_total` from the log: the CSV `train_loss`
    # column is the outer batch loss (roughly 4× larger for FlashDet), so
    # plotting it next to `val_loss` compared two different scales.
    objective_loss = history.get("train_loss_total") or []
    train_loss = (
        objective_loss
        if any(v is not None for v in objective_loss)
        else history.get("train_loss", [])
    )
    val_loss = history.get("val_loss", [])
    map50 = history.get("mAP50", [])
    if epochs:
        train_loss = train_loss[: len(epochs)]

    # Compact headline above the charts.
    bits = [f"**{len(train_loss)}** epochs logged"]
    map_pairs = _pairs(
        val_epochs or list(range(1, len(map50) + 1)), map50,
    )
    if map_pairs:
        best_epoch, best_map = max(map_pairs, key=lambda pair: pair[1])
        bits.append(f"best mAP@0.5 **{best_map:.4f}** · epoch **{best_epoch}**")
    tail_loss = [v for v in train_loss if v is not None]
    if tail_loss:
        bits.append(f"last train loss **{tail_loss[-1]:.3f}**")
    st.caption(" · ".join(bits))

    # Total loss often starts with a huge warmup epoch (e.g. 527 → 40), which
    # compresses the rest of the curve against zero on a linear axis. Default
    # to a log scale when the range spans more than ~8×, and let the user
    # switch back.
    loss_values = [v for v in train_loss if v is not None]
    loss_values += [v for v in val_loss if v is not None]
    loss_positive = bool(loss_values) and all(v > 0 for v in loss_values)
    auto_log = loss_positive and (
        max(loss_values) / min(loss_values) >= 8
    )
    log_loss = st.toggle(
        "Log loss scale", value=auto_log, key="curves_log_loss",
        help="Keeps later epochs readable when the first warmup epoch is much "
             "larger than the rest.",
    ) and loss_positive

    # ── Loss ──────────────────────────────────────────────────
    loss_fig = go.Figure()
    _add_line(loss_fig, epochs, train_loss, "Train loss", _COLORS["train_loss"])
    if val_loss:
        _add_line(
            loss_fig,
            val_epochs or list(range(1, len(val_loss) + 1)),
            val_loss, "Val loss", _COLORS["val_loss"],
        )
    _style(loss_fig, "Total loss", "Loss (log)" if log_loss else "Loss",
           log_y=log_loss)
    if log_loss:
        ticks = _log_tickvals(loss_values)
        if ticks:
            loss_fig.update_yaxes(
                tickvals=ticks, ticktext=[str(t) for t in ticks], nticks=6,
            )

    # ── mAP ───────────────────────────────────────────────────
    map_fig = go.Figure()
    if map_pairs:
        _add_line(
            map_fig,
            [p[0] for p in map_pairs],
            [p[1] for p in map_pairs],
            "mAP@0.5", _COLORS["map"],
        )
        best_epoch, best_map = max(map_pairs, key=lambda pair: pair[1])
        map_fig.add_trace(go.Scatter(
            x=[best_epoch], y=[best_map], mode="markers+text",
            marker=dict(symbol="star", size=13, color=_COLORS["map"]),
            text=[f"{best_map:.4f}"], textposition="top center",
            textfont=dict(size=10, color=_COLORS["map"]),
            name="Best", hovertemplate="Best %{y:.4f} @ epoch %{x}<extra></extra>",
            showlegend=False,
        ))
        map_fig.update_yaxes(rangemode="tozero")
    _style(map_fig, "mAP@0.5", "mAP")
    if not map_pairs:
        _empty_chart(map_fig, "mAP is recorded after validation epochs")

    # ── Sub-losses ────────────────────────────────────────────
    has_csv_subloss = any(v is not None for v in history.get("train_box", []))
    has_log_subloss = any(v is not None for v in history.get("o2m_cls", []))
    sub_fig = go.Figure()
    if has_csv_subloss:
        _add_line(sub_fig, epochs, history.get("train_cls", []), "Train cls",
                  _COLORS["cls"])
        _add_line(sub_fig, epochs, history.get("train_box", []), "Train box",
                  _COLORS["box"])
        _add_line(sub_fig, epochs, history.get("train_l1", []), "Train L1",
                  _COLORS["l1"])
        _add_line(sub_fig, val_epochs, history.get("val_cls", []), "Val cls",
                  _COLORS["cls"], dash="dot")
        _add_line(sub_fig, val_epochs, history.get("val_box", []), "Val box",
                  _COLORS["box"], dash="dot")
    elif has_log_subloss:
        _add_line(sub_fig, epochs, history.get("o2m_cls", []), "o2m cls",
                  _COLORS["cls"])
        _add_line(sub_fig, epochs, history.get("o2m_box", []), "o2m box",
                  _COLORS["box"])
        _add_line(sub_fig, epochs, history.get("o2o_cls", []), "o2o cls",
                  _COLORS["cls"], dash="dash")
        _add_line(sub_fig, epochs, history.get("o2o_box", []), "o2o box",
                  _COLORS["box"], dash="dash")
    sub_values = [
        v
        for key in ("train_cls", "train_box", "train_l1", "val_cls", "val_box",
                    "o2m_cls", "o2m_box", "o2o_cls", "o2o_box")
        for v in history.get(key, []) if v is not None
    ]
    sub_log = log_loss and bool(sub_values) and all(v > 0 for v in sub_values)
    _style(sub_fig, "Sub-losses", "Loss (log)" if sub_log else "Loss",
           log_y=sub_log)
    if sub_log:
        ticks = _log_tickvals(sub_values)
        if ticks:
            sub_fig.update_yaxes(
                tickvals=ticks, ticktext=[str(t) for t in ticks], nticks=6,
            )
    if not has_csv_subloss and not has_log_subloss:
        _empty_chart(sub_fig, "No sub-loss values in this run")

    # ── Learning rate ─────────────────────────────────────────
    lr_fig = go.Figure()
    lr = history.get("lr", [])
    if any(v is not None for v in lr):
        _add_line(
            lr_fig, list(range(1, len(lr) + 1)), lr, "Learning rate",
            _COLORS["lr"], markers=False,
        )
        lr_fig.update_yaxes(tickformat=".3e", nticks=5)
    _style(lr_fig, "Learning rate", "LR", log_y=True)
    if not any(v is not None for v in lr):
        _empty_chart(lr_fig, "No learning-rate values in this run")

    # Short runs get one tick per epoch; long runs keep Plotly's auto ticks.
    all_epochs = [x for x in epochs if isinstance(x, (int, float))]
    all_epochs += [x for x in val_epochs if isinstance(x, (int, float))]
    if all_epochs and max(all_epochs) - min(all_epochs) <= 20:
        for chart in (loss_fig, map_fig, sub_fig, lr_fig):
            chart.update_xaxes(dtick=1, tick0=1)

    row1 = st.columns(2, gap="medium")
    with row1[0]:
        st.plotly_chart(loss_fig, use_container_width=True,
                        config=_PLOT_CONFIG, key="curve_loss")
    with row1[1]:
        st.plotly_chart(map_fig, use_container_width=True,
                        config=_PLOT_CONFIG, key="curve_map")

    row2 = st.columns(2, gap="medium")
    with row2[0]:
        st.plotly_chart(sub_fig, use_container_width=True,
                        config=_PLOT_CONFIG, key="curve_subloss")
    with row2[1]:
        st.plotly_chart(lr_fig, use_container_width=True,
                        config=_PLOT_CONFIG, key="curve_lr")

    if has_static:
        with st.expander("FlashDet-generated plots", expanded=False):
            cc1, cc2 = st.columns(2)
            if os.path.isfile(training_curves_img):
                with cc1:
                    st.image(training_curves_img, caption="Training curves",
                             use_container_width=True)
            if os.path.isfile(map_curve_img):
                with cc2:
                    st.image(map_curve_img, caption="mAP curve",
                             use_container_width=True)
