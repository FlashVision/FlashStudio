"""Training log and CSV parsers."""

import os
import re
import glob as glob_module

from flashstudio.constants import TRAINING_LOG_CSV, TRAINING_LOG_GLOB


def _find_log_file(run_dir: str):
    """Find the best training log file — prefer the largest (stdout-redirected has more data)."""
    logs = glob_module.glob(os.path.join(run_dir, TRAINING_LOG_GLOB))
    if not logs:
        return None
    return max(logs, key=lambda p: os.path.getsize(p))


def _parse_training_csv(run_dir: str):
    """Parse training_log.csv — the primary metrics source from FlashDet."""
    csv_path = os.path.join(run_dir, TRAINING_LOG_CSV)
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

                mAP = row.get("mAP@0.5") or row.get("val_mAP") or ""
                history["mAP50"].append(float(mAP) if mAP else None)

                for key in ("train_box", "train_cls", "train_l1", "val_box", "val_cls", "val_l1"):
                    val = row.get(key, "")
                    history[key].append(float(val) if val else None)
    except Exception:
        return None

    # Build paired val_epochs list (epochs where validation actually ran)
    val_epochs = []
    val_loss_clean = []
    mAP50_clean = []
    for i, ep in enumerate(history["epochs"]):
        vl = history["val_loss"][i] if i < len(history["val_loss"]) else None
        mp = history["mAP50"][i] if i < len(history["mAP50"]) else None
        if vl is not None or mp is not None:
            val_epochs.append(ep)
            val_loss_clean.append(vl)
            mAP50_clean.append(mp)
    history["val_epochs"] = val_epochs
    history["val_loss"] = val_loss_clean
    history["mAP50"] = mAP50_clean

    # Fill metadata from log header FIRST — log has authoritative total_epochs
    log_file = _find_log_file(run_dir)
    if log_file:
        try:
            with open(log_file, "r") as f:
                header_lines = f.readlines()[:30]
            for line in header_lines:
                lc = line.strip()
                if "Model Size:" in lc:
                    history["model_info"] = lc.split("Model Size:")[-1].strip()
                if not history["model_info"]:
                    model_m = re.search(r"Model:\s*(\w+),\s*Input:\s*\((\d+),\s*(\d+)\)", lc)
                    if model_m:
                        history["model_info"] = f"{model_m.group(1)} ({model_m.group(2)}x{model_m.group(3)})"
                if "Device:" in lc:
                    history["device"] = lc.split("Device:")[-1].strip()
                if "Classes" in lc and ":" in lc:
                    m = re.search(r"Classes \((\d+)\):\s*\[(.+)\]", lc)
                    if m:
                        history["classes"] = [c.strip().strip("'").strip('"') for c in m.group(2).split(",")]
                if "Epochs:" in lc:
                    m = re.search(r"Epochs:\s*(\d+)", lc)
                    if m:
                        history["total_epochs"] = int(m.group(1))
                    bm = re.search(r"Batch:\s*(\d+)", lc)
                    if bm and not history["batch_size"]:
                        history["batch_size"] = int(bm.group(1))
                if "Batch Size:" in lc:
                    m = re.search(r"Batch Size:\s*(\d+)", lc)
                    if m and not history["batch_size"]:
                        history["batch_size"] = int(m.group(1))
                if "Learning Rate:" in lc and not history.get("initial_lr"):
                    m = re.search(r"Learning Rate:\s*([\d.e+-]+)", lc)
                    if m:
                        history["initial_lr"] = float(m.group(1))
        except OSError:
            pass

    # Fallback: if log didn't provide total_epochs, use the number of CSV rows
    if not history["total_epochs"]:
        history["total_epochs"] = len(history["epochs"])

    return history if history["epochs"] else None


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

        # Val Loss line — two formats:
        #   1. Val Loss: 561.98 (loss_total: ..., o2m_cls: X, ...) | mAP@0.5: M  (with sub-losses)
        #   2. Val Loss: 561.98 | mAP@0.5: M  (without sub-losses)
        val_m = re.search(r"Val Loss:\s*([\d.]+)\s*\((.+?)\)\s*\|\s*mAP@0\.5:\s*([\d.]+)", line)
        val_simple = None
        if not val_m:
            val_simple = re.search(r"Val Loss:\s*([\d.]+).*?mAP@0\.5:\s*([\d.]+)", line)
        if val_m:
            history["val_loss"].append(float(val_m.group(1)))
            history["mAP50"].append(float(val_m.group(3)))

            detail = val_m.group(2)
            lt = re.search(r"loss_total:\s*([\d.]+)", detail)
            if lt:
                while len(history["train_loss"]) < current_epoch:
                    history["train_loss"].append(None)
                if current_epoch > 0:
                    history["train_loss"][current_epoch - 1] = float(lt.group(1))

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
        elif val_simple:
            history["val_loss"].append(float(val_simple.group(1)))
            history["mAP50"].append(float(val_simple.group(2)))

        # Also try: Validation - Loss: X | mAP@0.5: Y (older format)
        if not val_m and not val_simple:
            alt_val = re.search(r"Validation.*Loss:\s*([\d.]+).*mAP@0\.5:\s*([\d.]+)", line)
            if alt_val:
                history["val_loss"].append(float(alt_val.group(1)))
                history["mAP50"].append(float(alt_val.group(2)))

        # Batch-level loss: Epoch [N] Batch [B/T] Loss: X (loss_total: Y, ...)
        batch_m = re.search(r"Epoch \[(\d+)\] Batch \[\d+/\d+\] Loss:\s*([\d.]+)", line)
        if batch_m:
            ep = int(batch_m.group(1))
            raw_loss = float(batch_m.group(2))
            # Use loss_total if available (more meaningful), else use raw Loss
            lt = re.search(r"loss_total:\s*([\d.]+)", line)
            loss_val = float(lt.group(1)) if lt else raw_loss
            while len(history["train_loss"]) < ep:
                history["train_loss"].append(None)
            history["train_loss"][ep - 1] = loss_val
            # Extract sub-losses
            for key, pattern in [
                ("o2m_cls", r"o2m_cls:\s*([\d.]+)"),
                ("o2m_box", r"o2m_box:\s*([\d.]+)"),
                ("o2o_cls", r"o2o_cls:\s*([\d.]+)"),
                ("o2o_box", r"o2o_box:\s*([\d.]+)"),
            ]:
                sm = re.search(pattern, line)
                while len(history[key]) < ep:
                    history[key].append(None)
                if sm and ep > 0:
                    history[key][ep - 1] = float(sm.group(1))

        # Epoch time — FlashDet logs "Time: Xs" at end of batch lines
        time_m = re.search(r"Epoch time:\s*([\d.]+)s", line)
        if not time_m:
            time_m = re.search(r"Time:\s*([\d.]+)s\s*$", line)
        if time_m:
            history["epoch_time"].append(float(time_m.group(1)))

    # Fill in epochs list from the longest metric array
    n = max(len(history["train_loss"]), len(history["lr"]),
            len(history["val_loss"]), len(history["mAP50"]))
    history["epochs"] = list(range(1, n + 1))

    if not history["epochs"]:
        return None
    return history
