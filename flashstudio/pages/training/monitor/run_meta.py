"""Run metadata extraction for the monitor tab."""

import os
import re
import glob as glob_module

from flashstudio.constants import (
    CKPT_BEST_INFERENCE, CKPT_BEST, CKPT_LAST_INFERENCE,
    CKPT_LAST, CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16,
    TRAINING_LOG_CSV, format_bytes,
)


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
        meta["size"] = format_bytes(total)
    except OSError:
        pass

    # Status: check for checkpoints
    has_final = (os.path.isfile(os.path.join(run_dir, CKPT_FINAL_INFERENCE))
                 or os.path.isfile(os.path.join(run_dir, CKPT_FINAL_FP16)))
    has_best = (os.path.isfile(os.path.join(run_dir, CKPT_BEST_INFERENCE))
                or os.path.isfile(os.path.join(run_dir, CKPT_BEST)))
    has_last = (os.path.isfile(os.path.join(run_dir, CKPT_LAST))
                or os.path.isfile(os.path.join(run_dir, CKPT_LAST_INFERENCE)))
    from flashstudio.constants import TRAINING_LOG_GLOB
    log_files = glob_module.glob(os.path.join(run_dir, TRAINING_LOG_GLOB))
    has_log = bool(log_files)
    has_csv = os.path.isfile(os.path.join(run_dir, TRAINING_LOG_CSV))

    if has_final:
        meta["status"] = "Complete"
    elif has_best and has_last:
        # Check log for "Training Complete!" to distinguish complete vs in-progress
        completed_in_log = False
        if log_files:
            try:
                with open(max(log_files, key=lambda p: os.path.getsize(p)), "r", encoding="utf-8", errors="replace") as _f:
                    _tail = _f.readlines()[-5:]
                completed_in_log = any("Training Complete!" in l for l in _tail)
            except OSError:
                pass
        meta["status"] = "Complete" if completed_in_log else "In Progress"
    elif has_last and (has_log or has_csv):
        meta["status"] = "In Progress"
    elif has_log or has_csv:
        meta["status"] = "Started"
    else:
        meta["status"] = "Empty"

    # Extract model/epochs from log header (use largest log file)
    if log_files:
        try:
            best_log = max(log_files, key=lambda p: os.path.getsize(p))
            with open(best_log, "r", encoding="utf-8", errors="replace") as f:
                header_lines = f.readlines()[:30]
            for line in header_lines:
                lc = line.strip()
                # FlashDet format: "Model Size: FlashDet-N"
                if "Model Size:" in lc and not meta["model"]:
                    meta["model"] = lc.split("Model Size:")[-1].strip()
                # Older/pico format: "Model: p, Input: (320, 320)"
                if not meta["model"]:
                    mm = re.search(r"Model:\s*(\w+),\s*Input:\s*\((\d+),\s*(\d+)\)", lc)
                    if mm:
                        meta["model"] = f"{mm.group(1)} ({mm.group(2)}x{mm.group(3)})"
                # Input size on separate line: "Input Size: (320, 320)"
                if "Input Size:" in lc and not meta.get("input_size"):
                    ism = re.search(r"Input Size:\s*\((\d+),\s*(\d+)\)", lc)
                    if ism:
                        meta["input_size"] = f"{ism.group(1)}x{ism.group(2)}"
                # Epochs on own line: "Epochs: 1000"
                if "Epochs:" in lc and meta["epochs"] == "?":
                    em = re.search(r"Epochs:\s*(\d+)", lc)
                    if em:
                        meta["epochs"] = em.group(1)
                # Also: "Epochs: 100, Batch: 16, LR: 0.001" (compact format)
                if "Batch Size:" in lc and not meta.get("batch_size"):
                    bm = re.search(r"Batch Size:\s*(\d+)", lc)
                    if bm:
                        meta["batch_size"] = bm.group(1)
                # Device
                if "Device:" in lc and not meta.get("device"):
                    meta["device"] = lc.split("Device:")[-1].strip()
        except OSError:
            pass

    # Fallback: parse training_log.csv for epoch count and mAP
    csv_path = os.path.join(run_dir, TRAINING_LOG_CSV)
    if os.path.isfile(csv_path) and meta["epochs"] == "?":
        try:
            import csv
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if rows:
                meta["epochs"] = len(rows)
                mAP_vals = []
                for r in rows:
                    v = r.get("mAP@0.5") or r.get("val_mAP") or ""
                    if isinstance(v, str) and v.strip():
                        try:
                            mAP_vals.append(float(v))
                        except ValueError:
                            pass
                if mAP_vals and max(mAP_vals) > 0 and not meta["mAP"]:
                    meta["mAP"] = max(mAP_vals)
        except Exception:
            pass

    # Fallback: parse log tail for "Best mAP@0.5:" and epoch count
    if (meta["epochs"] == "?" or not meta["mAP"]) and log_files:
        try:
            best_log = max(log_files, key=lambda p: os.path.getsize(p))
            with open(best_log, "r", encoding="utf-8", errors="replace") as f:
                tail = f.readlines()[-20:]
            for line in tail:
                bm = re.search(r"Best mAP@0\.5:\s*([\d.]+)", line)
                if bm and not meta["mAP"]:
                    meta["mAP"] = float(bm.group(1))
            if meta["epochs"] == "?":
                for line in tail:
                    em = re.search(r"Epoch (\d+)/(\d+)", line)
                    if em:
                        meta["epochs"] = em.group(1)
        except OSError:
            pass

    # Build enriched display name
    parts = [name]
    if meta["model"]:
        parts.append(meta["model"])
    if meta["mAP"]:
        parts.append(f"mAP={meta['mAP']:.3f}")
    meta["display_name"] = " | ".join(parts) if len(parts) > 1 else name

    return meta
