"""Standalone FlashDet training launcher.

The Streamlit UI writes a JSON manifest into the run directory and starts this
module in a subprocess::

    python -m flashstudio.utils.trainer_launcher /path/to/run/train.json

Using FlashDet's public ``Trainer`` API means training works with a normal
``pip install flashdet`` (which does not ship FlashDet's repository-level
``train.py``).
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict


def _load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid training manifest: {path}")
    return data


def _class_names(params: dict) -> list:
    names = params.get("class_names") or []
    if isinstance(names, str):
        names = [n.strip() for n in names.splitlines() if n.strip()]
    return [str(n).strip() for n in names if str(n).strip()]


def _build_config(params: dict):
    """Build a FlashDet ``Config`` from the UI manifest."""
    from flashdet.cfg import get_config

    model_size = str(params.get("model_size") or "n")
    input_size = int(params.get("input_size") or 320)
    num_classes = int(params.get("num_classes") or 80)
    cfg = get_config(model_size=model_size, input_size=input_size, num_classes=num_classes)

    mc = cfg.model
    mc.architecture = params.get("architecture") or "flashdet"
    mc.size = model_size
    mc.input_size = (input_size, input_size)
    mc.num_classes = num_classes

    # ``Trainer._apply_config_defaults`` maps legacy backbone sizes back to
    # model sizes and would overwrite the selected size.  A non-legacy value
    # keeps the explicit ``model_size`` passed to Trainer.
    mc.backbone_size = "__flashstudio__"
    mc.backbone_type = params.get("backbone_type") or "lite"
    mc.neck_type = params.get("neck_type")

    if mc.architecture in ("yolov8", "yolov9", "yolov10", "yolov11", "yolox"):
        mc.width_mult = float(params.get("yolo_width_mult") or 1.0)
        mc.depth_mult = float(params.get("yolo_depth_mult") or 1.0)
        mc.use_pgi = bool(params.get("yolo_use_pgi", True))
        mc.use_psa = bool(params.get("yolo_use_psa", True))
        mc.use_c2psa = bool(params.get("yolo_use_c2psa", True))

    train = cfg.train
    train.epochs = int(params.get("epochs") or 100)
    train.batch_size = int(params.get("batch_size") or 16)
    train.learning_rate = float(params.get("lr") or 1e-3)
    train.weight_decay = float(params.get("weight_decay") or 0.05)
    train.warmup_epochs = int(params.get("warmup_epochs") or 0)
    train.patience = int(params.get("patience") or 0)
    train.grad_accum = max(1, int(params.get("grad_accum") or 1))
    train.val_interval = max(1, int(params.get("val_interval") or 1))
    train.lrf = float(params.get("lr_final_ratio") or 0.1)
    train.grad_clip_norm = float(
        params.get("grad_clip_norm")
        or (10.0 if params.get("grad_clip", True) else float("inf"))
    )
    train.save_dir = params.get("save_dir") or ""
    train.resume = params.get("resume") or None
    train.amp = bool(params.get("amp"))
    train.multi_gpu = bool(params.get("ddp"))
    train.mosaic = bool(params.get("mosaic"))
    train.mixup = bool(params.get("mixup"))
    train.copy_paste = bool(params.get("copy_paste"))
    train.multi_scale = bool(params.get("multiscale"))
    train.channels_last = bool(params.get("channels_last"))
    train.nbs = int(params.get("nbs") or 64)
    train.ema_decay = float(params.get("ema_decay") or 0.9998)
    train.ema_warmup = int(params.get("ema_warmup") or 2000)
    train.warmup_momentum = float(params.get("warmup_momentum") or 0.8)
    train.base_momentum = float(params.get("base_momentum") or 0.937)
    train.val_score_thr = float(params.get("val_score_thr") or 0.01)
    train.max_det = int(params.get("max_det") or 300)
    train.chunked_cross_entropy = bool(params.get("chunked_loss"))
    train.ce_chunk_size = int(params.get("chunk_size") or 1024)
    train.use_lora = bool(params.get("lora"))
    train.lora_variant = params.get("lora_variant") or "standard"
    train.lora_rank = int(params.get("lora_rank") or 8)
    train.lora_alpha = float(params.get("lora_alpha") or 16.0)
    train.lora_dropout = float(params.get("lora_dropout") or 0.05)
    train.lora_target_modules = list(params.get("lora_targets") or ["backbone", "neck"])
    train.use_qlora = bool(params.get("qlora"))
    train.qlora_quant_dtype = params.get("qlora_dtype") or "int8"
    train.enable_activation_checkpointing = bool(params.get("activation_checkpointing"))
    train.enable_activation_offloading = bool(params.get("activation_offloading"))
    train.optimizer_in_bwd = bool(params.get("optimizer_in_bwd"))
    train.use_8bit_optimizer = bool(params.get("use_8bit_optimizer"))
    train.compile_model = bool(params.get("compile_model"))

    cfg.data.train_images = params.get("train_images") or ""
    cfg.data.val_images = params.get("val_images") or ""
    cfg.data.num_workers = int(params.get("num_workers") or 0)
    cfg.class_names = _class_names(params)
    return cfg


def _patch_checkpoint_metadata(trainer_module, params: dict) -> None:
    """Inject YOLO width/depth into saved checkpoint configs.

    ``Trainer`` records architecture/size but not the YOLO multipliers, which
    makes later inference/export rebuild the wrong model.  Wrapping the two
    save helpers keeps the metadata with the weights.
    """
    extras = {
        "width_mult": float(params.get("yolo_width_mult") or 1.0),
        "depth_mult": float(params.get("yolo_depth_mult") or 1.0),
        "use_pgi": bool(params.get("yolo_use_pgi", True)),
        "use_psa": bool(params.get("yolo_use_psa", True)),
        "use_c2psa": bool(params.get("yolo_use_c2psa", True)),
    }

    for name in ("save_checkpoint", "save_inference_weights"):
        original = getattr(trainer_module, name, None)
        if original is None:
            continue

        def _wrapped(*args, _original=original, **kwargs):
            config = kwargs.get("config")
            if config is not None:
                merged = dict(config)
                for key, value in extras.items():
                    merged.setdefault(key, value)
                kwargs["config"] = merged
            return _original(*args, **kwargs)

        setattr(trainer_module, name, _wrapped)


def _install_optimizer(trainer_module, params: dict, logger):
    """Make the UI optimizer selector work with Trainer's AdamW-only path."""
    import torch

    optimizer_name = str(params.get("optimizer") or "AdamW").strip().lower()
    original = trainer_module.create_optimizer

    def _create(model, lr, weight_decay, use_8bit=False, optimizer_in_bwd=False,
                betas=(0.9, 0.999), **kwargs):
        # Delegate memory-saving variants to FlashDet's implementation.
        if optimizer_in_bwd or use_8bit:
            return original(
                model, lr=lr, weight_decay=weight_decay,
                use_8bit=use_8bit, optimizer_in_bwd=optimizer_in_bwd, betas=betas,
                **kwargs,
            )
        if optimizer_name == "sgd":
            logger.info("Optimizer: SGD (momentum=0.9, nesterov)")
            return torch.optim.SGD(
                model.parameters(), lr=lr, momentum=0.9,
                weight_decay=weight_decay, nesterov=True,
            )
        if optimizer_name == "musgd":
            try:
                from flashdet.engine.core.musgd import build_musgd
                logger.info("Optimizer: MuSGD")
                return build_musgd(model, lr=lr, momentum=0.9, weight_decay=weight_decay)
            except Exception as exc:  # older FlashDet builds
                logger.warning("MuSGD unavailable (%s); falling back to SGD", exc)
                return torch.optim.SGD(
                    model.parameters(), lr=lr, momentum=0.9,
                    weight_decay=weight_decay, nesterov=True,
                )
        logger.info("Optimizer: AdamW")
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas,
        )

    trainer_module.create_optimizer = _create


def _freeze_modules(model, mode: str, logger) -> None:
    """Freeze backbone / neck according to the UI fine-tune strategy."""
    mode = (mode or "").strip().lower()
    if mode not in ("freeze backbone", "freeze bb+neck", "freeze bb_neck", "freeze backbone + neck"):
        return
    frozen = []
    for name in ("backbone", "neck", "fpn"):
        submodule = getattr(model, name, None)
        if submodule is None:
            continue
        if name == "backbone" or mode in ("freeze bb+neck", "freeze bb_neck", "freeze backbone + neck"):
            for p in submodule.parameters():
                p.requires_grad = False
            frozen.append(name)
    if frozen:
        logger.info("Frozen modules: %s", ", ".join(frozen))


# ──────────────────────────────────────────────────────────────────────
# Ultralytics backend
# ──────────────────────────────────────────────────────────────────────

def _ultralytics_model_name(params: dict) -> str:
    """Map the Model page family/size to an Ultralytics weight stem."""
    family = str(params.get("arch_family") or "").strip()
    arch = str(params.get("architecture") or "").strip().lower()
    if "YOLOX" in family.upper():
        raise RuntimeError(
            "The Ultralytics backend cannot build YOLOX models. "
            "Select YOLOv8/YOLOv9/YOLOv10/YOLOv11 or switch to the FlashDet backend."
        )
    if "YOLO" not in family.upper() and not arch.startswith("yolo"):
        raise RuntimeError(
            "The Ultralytics backend needs a YOLO family in Model → Architecture. "
            "Choose YOLOv8, YOLOv9, YOLOv10, or YOLOv11, or switch to FlashDet."
        )

    size = str(params.get("model_size") or "n").lower()
    width = float(params.get("yolo_width_mult") or 0.5)
    depth = float(params.get("yolo_depth_mult") or 0.33)
    # FlashDet size codes are p/n/s/m/l/x; YOLO families only have n/s/m/l/x.
    if size in ("p", ""):
        size = "n"
    if size not in ("n", "s", "m", "l", "x"):
        size = "n"
    scale = min(width, depth)
    if "YOLOv8" in family:
        base = "yolov8"
    elif "YOLOv9" in family:
        return "yolov9e" if size == "x" else "yolov9c"
    elif "YOLOv10" in family:
        base = "yolov10"
    elif "YOLOv11" in family:
        base = "yolo11"
    else:
        # Architecture was already a YOLO name.
        base = arch if arch.startswith("yolo") else "yolov8"
    if base in ("yolov10",):
        size = "b" if scale < 0.6 and size == "m" else size
    return f"{base}{size}"


def _coco_to_yolo(train_dir: str, val_dir: str, output_dir: str,
                  class_names: list | None = None) -> str:
    """Convert the two COCO split folders into a YOLO dataset + data.yaml."""
    import json
    import shutil

    def _read(split_dir: str):
        path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.isfile(path):
            candidates = sorted(
                f for f in os.listdir(split_dir) if f.endswith(".json")
            ) if os.path.isdir(split_dir) else []
            if not candidates:
                raise RuntimeError(f"No COCO annotation file found in {split_dir}")
            path = os.path.join(split_dir, candidates[0])
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    train_data = _read(train_dir)
    val_data = _read(val_dir)

    merged = {}
    for data in (train_data, val_data):
        for cat in data.get("categories", []):
            merged.setdefault(int(cat.get("id")), cat.get("name") or f"class_{cat.get('id')}")
    if class_names:
        names = [str(n).strip() for n in class_names if str(n).strip()]
    else:
        names = [merged[k] for k in sorted(merged)]
    if not names:
        raise RuntimeError("No classes found in the COCO annotations.")
    name_to_idx = {name: i for i, name in enumerate(names)}
    cat_to_idx = {
        cid: name_to_idx.get(name, i if i < len(names) else len(names) - 1)
        for i, (cid, name) in enumerate(sorted(merged.items()))
    }

    os.makedirs(output_dir, exist_ok=True)
    for split_name, src, data in (("train", train_dir, train_data), ("valid", val_dir, val_data)):
        img_out = os.path.join(output_dir, split_name, "images")
        lbl_out = os.path.join(output_dir, split_name, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        images = {int(i["id"]): i for i in data.get("images", [])}
        anns = {}
        for ann in data.get("annotations", []):
            anns.setdefault(int(ann["image_id"]), []).append(ann)

        for image_id, image in images.items():
            fname = image.get("file_name")
            if not fname:
                continue
            width = float(image.get("width") or 0)
            height = float(image.get("height") or 0)
            if width <= 0 or height <= 0:
                continue
            src_path = os.path.join(src, fname)
            if not os.path.isfile(src_path):
                continue
            dst_path = os.path.join(img_out, fname)
            if not os.path.exists(dst_path):
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except OSError:
                    shutil.copy2(src_path, dst_path)

            lines = []
            for ann in anns.get(image_id, []):
                bbox = ann.get("bbox") or []
                if len(bbox) != 4:
                    continue
                bx, by, bw, bh = [float(v) for v in bbox]
                if bw <= 0 or bh <= 0:
                    continue
                cls = cat_to_idx.get(int(ann.get("category_id")), 0)
                cx = max(0.0, min(1.0, (bx + bw / 2) / width))
                cy = max(0.0, min(1.0, (by + bh / 2) / height))
                nw = max(0.0, min(1.0, bw / width))
                nh = max(0.0, min(1.0, bh / height))
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            with open(os.path.join(lbl_out, os.path.splitext(fname)[0] + ".txt"),
                      "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"train: {os.path.join(output_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(output_dir, 'valid', 'images')}\n")
        f.write(f"nc: {len(names)}\n")
        f.write("names:\n")
        for i, name in enumerate(names):
            f.write(f"  {i}: {json.dumps(name)}\n")
    return yaml_path


def _attach_ultralytics_logger(model, save_dir: str) -> None:
    """Write the monitor's training_log.csv from Ultralytics callbacks."""
    import csv

    csv_path = os.path.join(save_dir, "training_log.csv")
    fields = ["epoch", "train_loss", "lr", "val_loss", "mAP@0.5"]
    initialized = {"value": False}
    written_epochs = set()

    def _num(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _write_row(row):
        if not initialized["value"]:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
            initialized["value"] = True
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

    def on_fit_epoch_end(trainer):
        try:
            metrics = dict(getattr(trainer, "metrics", {}) or {})
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            if epoch in written_epochs:
                return
            written_epochs.add(epoch)
            row = {"epoch": epoch}

            tloss = getattr(trainer, "tloss", None)
            if tloss is not None:
                try:
                    row["train_loss"] = f"{sum(float(v) for v in tloss):.6f}"
                except TypeError:
                    value = _num(tloss)
                    if value is not None:
                        row["train_loss"] = f"{value:.6f}"

            lr = _num(metrics.get("lr/pg0"))
            if lr is None:
                try:
                    lr = float(trainer.optimizer.param_groups[0]["lr"])
                except Exception:
                    lr = None
            if lr is not None:
                row["lr"] = f"{lr:.8f}"

            val_parts = [
                _num(metrics.get(f"val/{key}_loss"))
                for key in ("box", "cls", "dfl")
            ]
            val_parts = [v for v in val_parts if v is not None]
            if val_parts:
                row["val_loss"] = f"{sum(val_parts):.6f}"

            map50 = _num(metrics.get("metrics/mAP50(B)"))
            if map50 is None:
                map50 = _num(metrics.get("metrics/mAP50(M)"))
            if map50 is not None:
                row["mAP@0.5"] = f"{map50:.6f}"

            _write_row(row)
        except Exception as exc:  # never break training for logging
            print(f"[FlashStudio] Ultralytics log warning: {exc}")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def _run_ultralytics(params: dict, save_dir: str) -> int:
    """Train a YOLO model with the Ultralytics engine."""
    import csv
    import shutil

    from flashstudio.utils.model_loader import _patch_torchvision_nms

    # Some torch/torchvision CPU builds omit the NMS kernel; Ultralytics needs
    # it during validation, so install the portable implementation first.
    _patch_torchvision_nms()

    from ultralytics import YOLO

    train_images = os.path.abspath(params.get("train_images") or "")
    val_images = os.path.abspath(params.get("val_images") or "")
    if not os.path.isdir(train_images) or not os.path.isdir(val_images):
        raise RuntimeError("Ultralytics training needs both train and validation folders.")

    model_name = _ultralytics_model_name(params)
    yolo_root = os.path.join(save_dir, "yolo_dataset")
    data_yaml = _coco_to_yolo(
        train_images, val_images, yolo_root,
        class_names=_class_names(params) or None,
    )
    print(f"[FlashStudio] Ultralytics dataset: {data_yaml}")
    print(f"[FlashStudio] Ultralytics model:   {model_name}")

    weights = f"{model_name}.pt"
    custom = str(params.get("finetune") or "")
    if custom.lower().endswith(".pt") and os.path.isfile(custom):
        weights = custom
        print(f"[FlashStudio] Starting from custom weights: {custom}")
    else:
        try:
            from flashstudio.utils.weight_hub import download_ultralytics

            weights = download_ultralytics(weights)
            print(f"[FlashStudio] Cached pretrained weights: {weights}")
        except Exception as exc:
            print(f"[FlashStudio] Using default weight name ({exc})")
    try:
        model = YOLO(weights)
        print(f"[FlashStudio] Loaded pretrained {weights}")
    except Exception as exc:
        print(f"[FlashStudio] Pretrained {weights} unavailable ({exc}); building from YAML")
        model = YOLO(f"{model_name}.yaml")

    _attach_ultralytics_logger(model, save_dir)

    epoch_count = int(params.get("epochs") or 100)
    batch = int(params.get("batch_size") or 16)
    imgsz = int(params.get("input_size") or 320)
    workers = int(params.get("num_workers") or 0)
    device = str(params.get("device") or "cpu")
    batch = 1 if batch <= 0 else batch
    workers = 4 if workers <= 0 else workers
    patience = int(params.get("patience") or 0)

    optimizer = str(params.get("optimizer") or "AdamW")
    optimizer = {"musgd": "SGD", "auto": "auto"}.get(optimizer.lower(), optimizer)

    train_args = {
        "data": data_yaml,
        "epochs": epoch_count,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "project": save_dir,
        "name": "ultralytics",
        "exist_ok": True,
        "pretrained": True,
        "plots": True,
        "verbose": True,
        "seed": int(params.get("seed") or 0),
        "deterministic": True,
        "lr0": float(params.get("lr") or 1e-3),
        "lrf": float(params.get("lr_final_ratio") or 0.1),
        "weight_decay": float(params.get("weight_decay") or 0.05),
        "warmup_epochs": float(params.get("warmup_epochs") or 0),
        "optimizer": optimizer,
        "amp": bool(params.get("amp")) and device != "cpu",
        "mosaic": 1.0 if params.get("mosaic") else 0.0,
        "mixup": 0.15 if params.get("mixup") else 0.0,
        "copy_paste": 0.1 if params.get("copy_paste") else 0.0,
    }
    if patience > 0:
        train_args["patience"] = patience
    resume = params.get("resume")
    if resume and os.path.isfile(str(resume)):
        train_args["resume"] = str(resume)

    print(f"[FlashStudio] Training {model_name} for {epoch_count} epochs on {device}")
    results = model.train(**train_args)

    ul_dir = os.path.join(save_dir, "ultralytics")
    best = os.path.join(ul_dir, "weights", "best.pt")
    last = os.path.join(ul_dir, "weights", "last.pt")
    if not os.path.isfile(best) and os.path.isfile(last):
        best = last
    if not os.path.isfile(best):
        raise RuntimeError(
            "Ultralytics finished without producing weights. Check the training log above."
        )
    shutil.copy2(best, os.path.join(save_dir, "model_best_inference.pt"))
    if os.path.isfile(last):
        shutil.copy2(last, os.path.join(save_dir, "model_last_inference.pt"))

    # Map Ultralytics results.csv into the monitor's training_log.csv.
    results_csv = os.path.join(ul_dir, "results.csv")
    if os.path.isfile(results_csv) and not os.path.isfile(
        os.path.join(save_dir, "training_log.csv")
    ):
        try:
            with open(results_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            with open(os.path.join(save_dir, "training_log.csv"), "w", newline="",
                      encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["epoch", "train_loss", "lr", "val_loss", "mAP@0.5"],
                )
                writer.writeheader()
                for row in rows:
                    def _get(*keys):
                        for k in keys:
                            if k in row and str(row[k]).strip():
                                try:
                                    return float(row[k])
                                except ValueError:
                                    pass
                        return None
                    losses = [
                        _get("train/box_loss"), _get("train/cls_loss"),
                        _get("train/dfl_loss"),
                    ]
                    losses = [v for v in losses if v is not None]
                    val_losses = [
                        _get("val/box_loss"), _get("val/cls_loss"),
                        _get("val/dfl_loss"),
                    ]
                    val_losses = [v for v in val_losses if v is not None]
                    writer.writerow({
                        "epoch": int(_get("epoch") or 0) + 1,
                        "train_loss": f"{sum(losses):.6f}" if losses else "",
                        "lr": f"{_get('lr/pg0') or 0:.8f}",
                        "val_loss": f"{sum(val_losses):.6f}" if val_losses else "",
                        "mAP@0.5": f"{_get('metrics/mAP50(B)') or 0:.6f}",
                    })
        except Exception as exc:
            print(f"[FlashStudio] Could not map results.csv: {exc}")

    best_map = None
    try:
        metrics = getattr(results, "results_dict", None) or {}
        value = metrics.get("metrics/mAP50(B)")
        if value is not None:
            best_map = float(value)
    except Exception:
        pass
    _write_results(save_dir, {
        "best_map50": best_map,
        "best_loss": None,
    }, {**params, "architecture": model_name})

    try:
        import yaml
        with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump({**params, "ultralytics_model": model_name}, f, sort_keys=False)
    except Exception:
        pass

    print("[FlashStudio] Ultralytics training complete")
    return 0


def _write_results(save_dir: str, metrics: dict, params: dict) -> None:
    results = {
        "best_mAP": metrics.get("best_map50"),
        "best_loss": metrics.get("best_loss"),
        "model_size": params.get("model_size"),
        "architecture": params.get("architecture"),
        "epochs": params.get("epochs"),
        "finished": True,
    }
    try:
        with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except OSError:
        pass


def run(manifest_path: str) -> int:
    params = _load_manifest(manifest_path)
    save_dir = os.path.abspath(params.get("save_dir") or os.getcwd())
    os.makedirs(save_dir, exist_ok=True)

    seed = int(params.get("seed") or 0)
    try:
        import random

        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if params.get("tf32"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if params.get("cudnn_benchmark"):
                torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    if seed:
        print(f"[FlashStudio] Random seed: {seed}")

    if str(params.get("train_backend") or "FlashDet") == "Ultralytics":
        return _run_ultralytics(params, save_dir)

    from flashdet.engine.training import trainer as trainer_module

    cfg = _build_config({**params, "save_dir": save_dir})
    Trainer = trainer_module.Trainer

    class StudioTrainer(Trainer):
        """Trainer subclass adding FlashStudio-specific behavior."""

        # FlashDet 1.3.0's _apply_config_defaults reads these before __init__
        # assigns them; class defaults make config-driven construction safe.
        chunked_loss = False
        chunk_size = 1024

        def _post_build_model(self, model):
            parent = getattr(super(), "_post_build_model", None)
            if callable(parent):
                parent(model)
            if params.get("channels_last"):
                try:
                    import torch

                    model.to(memory_format=torch.channels_last)
                    self._logger.info("Memory format: channels-last (NHWC)")
                except Exception as exc:
                    self._logger.warning("Channels-last unavailable: %s", exc)
            _freeze_modules(model, params.get("finetune_strategy", ""), self._logger)

    kwargs = {
        "model_size": params.get("model_size") or "n",
        "input_size": int(params.get("input_size") or 320),
        "architecture": params.get("architecture") or "flashdet",
        "save_dir": save_dir,
        "device": params.get("device") or "cpu",
        "train_images": params.get("train_images"),
        "val_images": params.get("val_images"),
        "class_file": params.get("class_file"),
        "finetune": params.get("finetune"),
        "resume": params.get("resume"),
        "lora": bool(params.get("lora")),
        "qlora": bool(params.get("qlora")),
        "mosaic": bool(params.get("mosaic")),
        "mixup": bool(params.get("mixup")),
        "copy_paste": bool(params.get("copy_paste")),
        "backbone_type": params.get("backbone_type"),
        "neck_type": params.get("neck_type"),
        "compile": bool(params.get("compile_model")),
        "config": cfg,
    }
    # Some FlashDet versions do not expose every keyword. Filter defensively.
    import inspect
    accepted = inspect.signature(Trainer.__init__).parameters
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    trainer = StudioTrainer(**kwargs)
    # These fields live only on the instance in current FlashDet releases.
    trainer.val_interval = int(params.get("val_interval") or 1)
    trainer.lrf = float(params.get("lr_final_ratio") or 0.1)
    trainer._logger.info("FlashStudio backend: flashdet.Trainer")
    trainer._logger.info("Model Size: FlashDet-%s", str(params.get("model_size") or "n").upper())
    for warning in params.get("warnings") or []:
        trainer._logger.warning(str(warning))
    if params.get("multiscale"):
        trainer._logger.warning(
            "Multi-Scale training is not supported by the installed FlashDet "
            "Trainer; ignoring this option."
        )

    _install_optimizer(trainer_module, params, trainer._logger)
    _patch_checkpoint_metadata(trainer_module, params)

    # Attach the artifact writers the FlashStudio monitor expects.
    from flashstudio.utils.training_hooks import StudioCSVLogger, StudioVisualizationCallback
    trainer.add_callback(StudioCSVLogger(save_dir))
    try:
        trainer.add_callback(StudioVisualizationCallback(save_dir))
    except Exception as exc:
        trainer._logger.warning("Visualization callback unavailable: %s", exc)

    # Expose the val loader/model to the visualization callback.
    original_validate = getattr(trainer, "_validate", None)
    if callable(original_validate):
        def _validate_with_context(model, dataloader, ema, class_names, epoch=0):
            trainer._studio_val_loader = dataloader
            trainer._studio_eval_model = ema.ema if ema is not None else model
            return original_validate(model, dataloader, ema, class_names, epoch=epoch)

        trainer._validate = _validate_with_context

    metrics = trainer.train()
    _write_results(save_dir, metrics, params)

    if params.get("save_best"):
        # "Save Best Only" still keeps the final inference weights, but drops
        # the resumable last checkpoints to save disk space.
        for name in ("checkpoint_last.pth", "model_last_inference.pth", "model_last_fp16.pth"):
            try:
                os.remove(os.path.join(save_dir, name))
            except OSError:
                pass

    try:
        import yaml
        with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    except Exception:
        pass
    return 0


def main(argv: list | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m flashstudio.utils.trainer_launcher <manifest.json>", file=sys.stderr)
        return 2
    try:
        return run(argv[0])
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
