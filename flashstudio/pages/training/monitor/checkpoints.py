"""Checkpoint and file listing for training runs."""

import os
import re
import streamlit as st

from flashstudio.pages.training.monitor.parsers import _find_log_file


def _render_checkpoints(run_dir):
    """Files tab — list all artifacts from a FlashDet training run."""
    from flashstudio.constants import ONNX_MODEL_FILE, ONNX_DATA_FILE
    files_info = []
    EXTENSIONS = (".pth", ".json", ".csv", ".log", ".onnx", ".onnx.data", ".txt")
    for f in sorted(os.listdir(run_dir)):
        fpath = os.path.join(run_dir, f)
        if os.path.isfile(fpath) and (f.endswith(EXTENSIONS) or f in (ONNX_MODEL_FILE, ONNX_DATA_FILE)):
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

    # Show training summary from log if complete
    log_file = _find_log_file(run_dir)
    if log_file:
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as _lf:
                _tail = _lf.readlines()[-10:]
            for _ln in _tail:
                bm = re.search(r"Best mAP@0\.5:\s*([\d.]+)\s*\|\s*Best Loss:\s*([\d.]+)", _ln)
                if bm:
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.metric("Best mAP@0.5", bm.group(1))
                    with rc2:
                        st.metric("Best Val Loss", bm.group(2))
                    break
        except OSError:
            pass


def _file_type(filename):
    """Categorize a file by its name — aligned with FlashDet output."""
    if "final" in filename and "inference" in filename:
        return "Final inference weights"
    if "final" in filename and "fp16" in filename:
        return "Final FP16 weights"
    if "best" in filename and "inference" in filename:
        return "Best inference weights"
    if "best" in filename and "fp16" in filename:
        return "Best FP16 weights"
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
    from flashstudio.constants import ONNX_MODEL_FILE, ONNX_DATA_FILE, RESULTS_JSON_FILE
    if filename == ONNX_MODEL_FILE:
        return "ONNX model"
    if filename == ONNX_DATA_FILE:
        return "ONNX weights data"
    if filename == RESULTS_JSON_FILE:
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
