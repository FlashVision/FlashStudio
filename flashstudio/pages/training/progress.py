"""Live training progress estimation.

The Monitor parsers only report *finished* epochs, so a freshly started run
shows no progress until the first epoch completes.  This module also reads the
in-flight log tail (``Epoch [N] Batch [B/T]``), which lets the UI show a
percentage immediately after training starts.
"""

from __future__ import annotations

import glob as glob_module
import os
import re
import time

import streamlit as st

from flashstudio.constants import TRAINING_LOG_CSV, TRAINING_LOG_GLOB

_BATCH_RE = re.compile(r"Epoch \[(\d+)\] Batch \[(\d+)/(\d+)\]")
_EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+)")
_TOTAL_RE = re.compile(r"Epochs:\s*(\d+)")
# FlashDet also logs compact progress lines like "  [420/529] Loss: ...".
_COMPACT_BATCH_RE = re.compile(r"\[(\d+)/(\d+)\]")

_TAIL_BYTES = 200_000
_HEAD_BYTES = 16_000


def _state(key, default=None):
    """Read session state without exploding outside a Streamlit script run."""
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default


def _find_log(run_dir):
    if not run_dir or not os.path.isdir(run_dir):
        return None
    logs = glob_module.glob(os.path.join(run_dir, TRAINING_LOG_GLOB))
    if not logs:
        return None
    return max(logs, key=lambda path: os.path.getsize(path))


def _read_log(path, limit, *, tail=False):
    """Read at most *limit* bytes from the head or tail of a log file."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if tail and size > limit:
                fh.seek(size - limit)
            data = fh.read(limit)
        return data.decode("utf-8", "replace")
    except OSError:
        return ""


def _csv_epoch_count(run_dir):
    """Number of completed epochs recorded in training_log.csv."""
    if not run_dir:
        return 0
    path = os.path.join(run_dir, TRAINING_LOG_CSV)
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            rows = sum(1 for _ in fh)
        return max(rows - 1, 0)
    except OSError:
        return 0


def _total_epochs_from_log(log_path):
    if not log_path:
        return 0
    head = _read_log(log_path, _HEAD_BYTES)
    match = _TOTAL_RE.search(head)
    if match:
        return int(match.group(1))
    match = _EPOCH_RE.search(head)
    if match:
        return int(match.group(2))
    return 0


def _parse_live_position(text):
    """Return (epoch, batch, total_batches) from the newest log lines."""
    epoch = batch = total_batches = 0
    for match in _BATCH_RE.finditer(text):
        epoch = int(match.group(1))
        batch = int(match.group(2))
        total_batches = int(match.group(3))
    if not epoch:
        for match in _EPOCH_RE.finditer(text):
            epoch = int(match.group(1))
    if epoch and not total_batches:
        for match in _COMPACT_BATCH_RE.finditer(text):
            batch = int(match.group(1))
            total_batches = int(match.group(2))
    return epoch, batch, total_batches


def format_eta(seconds):
    """Format an ETA in seconds as a short human string (or '' when unknown)."""
    if seconds is None or seconds <= 0:
        return ""
    if seconds >= 3600:
        return f"~{seconds / 3600:.1f}h left"
    if seconds >= 60:
        return f"~{seconds / 60:.0f}m left"
    return f"~{seconds:.0f}s left"


def format_percent(fraction):
    """Format a 0..1 fraction as a percentage with useful precision."""
    pct = max(0.0, min(float(fraction), 1.0)) * 100
    if pct >= 99.95:
        return "100%"
    if pct >= 10:
        return f"{pct:.0f}%"
    return f"{pct:.1f}%"


def _training_progress(run_dir=None, total_epochs=None):
    """Estimate progress for a run (defaults to the active run).

    Returns a dict, or ``None`` when there is nothing to report.  The dict
    always contains ``fraction`` (0..1), ``percent`` (string), ``detail``
    (short human label), ``eta_seconds`` and the raw epoch/batch fields.
    """
    active_run = _state("active_run_path") or ""
    run_dir = run_dir or active_run

    # Only trust the session log path for the active run; the Monitor tab may
    # be showing a different (historical) run.
    log_path = ""
    if active_run and os.path.abspath(str(run_dir)) == os.path.abspath(str(active_run)):
        log_path = _state("training_log_file") or ""
    if log_path and not os.path.isfile(log_path):
        log_path = ""
    if not log_path:
        log_path = _find_log(run_dir)

    completed = _csv_epoch_count(run_dir)
    total = int(total_epochs or 0) or _total_epochs_from_log(log_path)
    is_active_run = bool(active_run) and (
        os.path.abspath(str(run_dir)) == os.path.abspath(str(active_run))
    )
    if not total and is_active_run:
        try:
            total = int(_state("epochs") or 0)
        except (TypeError, ValueError):
            total = 0
    if total <= 0:
        total = completed

    tail = _read_log(log_path, _TAIL_BYTES, tail=True) if log_path else ""
    epoch, batch, total_batches = _parse_live_position(tail)
    complete = "Training Complete!" in tail

    if complete and total:
        completed = total

    if not total and not completed and not epoch:
        if _state("training_active") and is_active_run:
            # Training has started but has not written its header yet.
            return {
                "fraction": 0.0,
                "percent": "0%",
                "detail": "Preparing training…",
                "eta_seconds": None,
                "epoch": 0,
                "total_epochs": 0,
                "batch": 0,
                "batches_per_epoch": 0,
                "complete": False,
            }
        return None

    fraction = 0.0
    if complete:
        fraction = 1.0
    elif total > 0:
        if epoch > 0 and total_batches > 0:
            within_epoch = min(batch / total_batches, 1.0)
            fraction = ((epoch - 1) + within_epoch) / total
        elif epoch > 0:
            fraction = (epoch - 1) / total
        elif completed:
            fraction = completed / total
    if not complete:
        # Never claim 100% while the process is still running.
        fraction = min(max(fraction, 0.0), 0.995)

    eta_seconds = None
    if not complete and _state("training_active"):
        start = _state("training_start_time")
        if start and fraction > 0.005:
            try:
                elapsed = max(time.time() - float(start), 0.0)
            except (TypeError, ValueError):
                elapsed = 0.0
            if elapsed > 0:
                eta_seconds = elapsed * (1.0 - fraction) / fraction

    if complete:
        detail = f"Epoch {total}/{total}" if total else "Training complete"
    elif epoch and total and total_batches:
        detail = f"Epoch {epoch}/{total} · batch {batch}/{total_batches}"
    elif epoch and total:
        detail = f"Epoch {epoch}/{total}"
    elif completed and total:
        detail = f"Epoch {completed}/{total}"
    else:
        detail = "Preparing training…"

    return {
        "fraction": fraction,
        "percent": format_percent(fraction),
        "detail": detail,
        "eta_seconds": eta_seconds,
        "epoch": epoch or completed,
        "total_epochs": total,
        "batch": batch,
        "batches_per_epoch": total_batches,
        "complete": complete,
    }


def _render_training_progress(run_dir=None, *, show_eta=True):
    """Render a progress bar with percentage + epoch detail. Returns the info."""
    info = _training_progress(run_dir)
    if not info:
        return None
    text = f"{info['percent']} · {info['detail']}"
    eta = format_eta(info.get("eta_seconds")) if show_eta else ""
    if eta:
        text += f" · {eta}"
    st.progress(info["fraction"], text=text)
    return info
