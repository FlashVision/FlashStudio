"""Training runner — dataset validation, format conversion, and FlashDet subprocess launch."""

import os
import json
import streamlit as st

from flashstudio.constants import (
    FLASHDET_MODELS, DEFAULT_DATA_DIR, TRAIN_WEIGHT_DECAY, TRAIN_VAL_INTERVAL, DEFAULT_OPTIMIZER,
    TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR, TRAIN_WARMUP_EPOCHS, TRAIN_PATIENCE,
    TRAIN_NUM_WORKERS, TRAIN_GRAD_ACCUM, TRAIN_LR_FINAL_RATIO,
    LORA_RANK_DEFAULT, LORA_ALPHA_DEFAULT, LORA_DROPOUT_DEFAULT,
    QLORA_DTYPE_DEFAULT, CHUNKED_LOSS_CHUNK_SIZE,
)
from flashstudio.pages.training._common import _get_save_dir
from flashstudio.utils.dataset_paths import (
    ensure_coco_annotation, find_validation_dir, has_images,
    load_coco_classes, resolve_annotation_file,
)


def _generate_run_name() -> str:
    """Generate a descriptive default run name."""
    from datetime import datetime
    from flashstudio.utils import get_state
    arch = get_state("model_arch")
    size_code = arch.split("-")[-1].lower()[:4] if "-" in arch else "det"
    dataset = st.session_state.get("dataset_name", "")
    ds_code = dataset.split("(")[0].strip().replace(" ", "")[:8].lower() if dataset else "custom"
    timestamp = datetime.now().strftime("%m%d_%H%M")
    return f"{size_code}_{ds_code}_{timestamp}"


def _start_training():
    """Start training using flashdet.Trainer Python API with pre-flight validation."""
    try:
        from flashdet.data import detect_dataset_format, convert_dataset
    except ImportError as exc:
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "FlashDet missing"
        st.session_state["training_error"] = (
            f"FlashDet is not installed or is missing data utilities: {exc}. "
            "Install flashdet>=1.3.0."
        )
        return

    train_images = st.session_state.get("train_img_path", "")
    val_images = st.session_state.get("val_img_path", "")

    # Validate dataset paths exist
    if not train_images or not os.path.isdir(train_images):
        dataset_name = st.session_state.get("dataset_name", "")
        dataset_id = "".join(
            ch for ch in dataset_name.lower().replace("(demo)", "sample")
            if ch.isalnum()
        )
        try:
            from flashstudio.components.project_manager import get_active_project_data_dir
            data_base = get_active_project_data_dir() or DEFAULT_DATA_DIR
        except Exception:
            data_base = DEFAULT_DATA_DIR
        for candidate_id in ["coco2017-val", "voc2007", "voc2012", "coco2017", "sample"]:
            candidate_key = "".join(
                ch for ch in candidate_id.lower() if ch.isalnum()
            )
            if candidate_key in dataset_id or dataset_id in candidate_key:
                candidate_dir = os.path.join(data_base, candidate_id)
                if os.path.isdir(os.path.join(candidate_dir, "train")):
                    train_images = os.path.join(candidate_dir, "train")
                    val_images = os.path.join(candidate_dir, "valid")
                    break

    # Convert to absolute paths to avoid relative path issues
    if train_images:
        train_images = os.path.abspath(train_images)
    if val_images:
        val_images = os.path.abspath(val_images)

    if not train_images or not os.path.isdir(train_images):
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "No dataset"
        st.session_state["training_error"] = (
            "No dataset paths configured. Go to the Data page and either "
            "enter train/val image paths manually, or download a dataset first."
        )
        return

    # A validation folder used as the training split would make FlashDet look
    # for ``<...>/valid/train/_annotations.coco.json``.  Fail with a clear
    # message instead of the confusing derived path.
    from flashstudio.utils.dataset_paths import split_dir as _split_dir

    train_split = _split_dir(train_images)
    if os.path.basename(train_split).lower().startswith(("val", "test")):
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Invalid dataset layout"
        st.session_state["training_error"] = (
            f"Train images point at a validation/test split: `{train_images}`. "
            "Load the dataset again (or set Train → Images to the training "
            "folder) so the train and validation splits come from the same dataset."
        )
        return

    # Auto-find val path if missing but train exists.  This must handle both
    # dataset/train + dataset/valid and dataset/train/images +
    # dataset/valid/images layouts.
    if not val_images or not os.path.isdir(val_images):
        val_images = find_validation_dir(train_images)

    if not val_images or not os.path.isdir(val_images):
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "No validation set"
        st.session_state["training_error"] = (
            "A validation set is required. Point Data → Load data → Upload → "
            "Validation → Images folder at a separate split with "
            "_annotations.coco.json. Training on the train split would produce "
            "misleading metrics."
        )
        return

    # Auto-detect classes from annotation if not already set.  The annotation
    # may live in a parent folder or an annotations/ sibling.
    _cls_raw = st.session_state.get("class_names", "")
    if isinstance(_cls_raw, list):
        _cls_raw = "\n".join(_cls_raw)
    if not _cls_raw.strip():
        ann_file = resolve_annotation_file(
            train_images,
            split="train",
            hint=st.session_state.get("train_ann_path", ""),
        )
        names = load_coco_classes(ann_file) if ann_file else []
        if names:
            st.session_state["class_names"] = "\n".join(names)
            st.session_state["num_classes"] = len(names)

    # Auto-detect and convert format if not COCO.  Prefer the dataset root so
    # an existing validation split is converted too, instead of re-splitting
    # only the train folder.
    parent_dir = (
        os.path.dirname(train_images)
        if os.path.basename(train_images) in ("train", "images", "imgs")
        else train_images
    )
    conversion_root = parent_dir
    if os.path.basename(train_images).lower() in ("images", "imgs"):
        split_directory = os.path.dirname(train_images)
        candidate_root = os.path.dirname(split_directory)
        if find_validation_dir(train_images, root=candidate_root):
            conversion_root = candidate_root
    detected_fmt = detect_dataset_format(conversion_root)
    if detected_fmt == "unknown":
        from flashstudio.pages.data.helpers import _find_labels

        label_roots = (
            conversion_root,
            os.path.dirname(conversion_root),
            os.path.dirname(os.path.dirname(conversion_root)),
        )
        for candidate in label_roots:
            if _find_labels(candidate):
                detected_fmt = "txt"
                # Convert from the common dataset root so an existing
                # validation split is preserved.
                conversion_root = candidate
                break
    parent_dir = conversion_root

    if detected_fmt in ("txt", "voc"):
        st.warning(f"Dataset format detected: **{detected_fmt}**. Converting to COCO JSON...")
        try:
            output_dir = parent_dir + "_coco"
            class_names = None
            raw_names = st.session_state.get("class_names", "")
            if isinstance(raw_names, list):
                raw_names = "\n".join(raw_names)
            if raw_names.strip():
                class_names = [c.strip() for c in raw_names.strip().split("\n") if c.strip()]

            import shutil

            shutil.rmtree(output_dir, ignore_errors=True)
            try:
                convert_dataset(
                    source_dir=parent_dir, output_dir=output_dir,
                    target_format="coco", class_names=class_names,
                )
            except Exception:
                if detected_fmt != "txt":
                    raise
                from flashstudio.pages.data.helpers import _convert_yolo_layout

                shutil.rmtree(output_dir, ignore_errors=True)
                stats = _convert_yolo_layout(
                    parent_dir, output_dir, class_names=class_names
                )
                if not stats:
                    raise

            train_images = os.path.join(output_dir, "train")
            val_dir = os.path.join(output_dir, "valid")
            if not os.path.isdir(val_dir):
                val_dir = os.path.join(output_dir, "val")
            val_images = val_dir
            st.session_state["train_img_path"] = train_images
            st.session_state["val_img_path"] = val_images
            st.success("Dataset converted to COCO format.")
        except Exception as e:
            st.session_state["training_active"] = False
            st.session_state["training_status"] = "Conversion failed"
            st.session_state["training_error"] = f"Format conversion failed: {e}"
            return

    # FlashDet requires the annotation file to be named exactly
    # ``_annotations.coco.json`` inside each image directory.  Find it even
    # when the upload kept it in a parent split folder or annotations/ sibling
    # and link/copy it into place.
    train_ann = ensure_coco_annotation(
        train_images,
        split="train",
        hint=st.session_state.get("train_ann_path", ""),
    )
    if train_ann:
        st.session_state["train_ann_path"] = train_ann
    if not train_ann or not os.path.isfile(
        os.path.join(train_images, "_annotations.coco.json")
    ):
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Missing annotations"
        st.session_state["training_error"] = (
            f"Missing annotations! FlashDet expects _annotations.coco.json in: {train_images}"
        )
        return

    # Verify validation annotations too — silently training on the train split
    # produces meaningless metrics.
    val_ann = ensure_coco_annotation(
        val_images,
        split="val",
        hint=st.session_state.get("val_ann_path", ""),
    )
    if val_ann:
        st.session_state["val_ann_path"] = val_ann
    if not val_ann or not os.path.isfile(
        os.path.join(val_images, "_annotations.coco.json")
    ):
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Missing val annotations"
        st.session_state["training_error"] = (
            f"Missing validation annotations! Expected "
            f"_annotations.coco.json in: {val_images}"
        )
        return

    # Conversion (YOLO/VOC -> COCO) may have created the first COCO
    # annotations; detect classes again now that they exist.
    remaining_classes = st.session_state.get("class_names", "")
    if isinstance(remaining_classes, list):
        remaining_classes = "\n".join(remaining_classes)
    if not remaining_classes.strip():
        names = load_coco_classes(
            os.path.join(train_images, "_annotations.coco.json")
        )
        if names:
            st.session_state["class_names"] = "\n".join(names)
            st.session_state["num_classes"] = len(names)

    # Normalize dataset layout for FlashDet: it computes
    # data_root = dirname(train_images) and expects
    # data_root/train/_annotations.coco.json and data_root/valid/... to exist.
    data_root = os.path.dirname(os.path.normpath(train_images))
    train_dir_in_root = os.path.join(data_root, "train")
    if os.path.abspath(train_images) != os.path.abspath(train_dir_in_root):
        if os.path.islink(train_dir_in_root):
            if os.path.realpath(train_dir_in_root) != os.path.realpath(train_images):
                try:
                    os.unlink(train_dir_in_root)
                except OSError:
                    pass
        if os.path.isdir(train_dir_in_root):
            if not os.path.isfile(os.path.join(train_dir_in_root, "_annotations.coco.json")):
                ensure_coco_annotation(
                    train_dir_in_root, split="train",
                    hint=st.session_state.get("train_ann_path", ""),
                )
            if (
                os.path.isfile(os.path.join(train_dir_in_root, "_annotations.coco.json"))
                and has_images(train_dir_in_root)
            ):
                train_images = train_dir_in_root
        elif not os.path.exists(train_dir_in_root):
            try:
                os.symlink(
                    os.path.abspath(train_images), train_dir_in_root,
                    target_is_directory=True,
                )
            except OSError:
                pass

    valid_dir_in_root = os.path.join(data_root, "valid")
    if os.path.abspath(val_images) != os.path.abspath(valid_dir_in_root):
        if os.path.islink(valid_dir_in_root):
            current_target = os.path.realpath(valid_dir_in_root)
            if current_target != os.path.realpath(val_images):
                try:
                    os.unlink(valid_dir_in_root)
                except OSError:
                    pass
        if not os.path.exists(valid_dir_in_root):
            try:
                os.symlink(
                    os.path.abspath(val_images), valid_dir_in_root,
                    target_is_directory=True,
                )
            except OSError:
                pass

        if os.path.isdir(valid_dir_in_root):
            # A real directory can contain a nested images/ folder (mixed
            # train/valid layouts).  FlashDet's verifier only needs an
            # annotation file here; the dataloader keeps using val_images.
            if not os.path.isfile(os.path.join(valid_dir_in_root, "_annotations.coco.json")):
                ensure_coco_annotation(
                    valid_dir_in_root, split="val",
                    hint=st.session_state.get("val_ann_path", ""),
                )
            if (
                os.path.isfile(os.path.join(valid_dir_in_root, "_annotations.coco.json"))
                and has_images(valid_dir_in_root)
            ):
                val_images = valid_dir_in_root

    if not os.path.isfile(os.path.join(train_dir_in_root, "_annotations.coco.json")):
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Invalid dataset layout"
        st.session_state["training_error"] = (
            f"FlashDet requires `{train_dir_in_root}/_annotations.coco.json`."
        )
        return
    if not os.path.isfile(os.path.join(valid_dir_in_root, "_annotations.coco.json")):
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Invalid dataset layout"
        st.session_state["training_error"] = (
            f"FlashDet requires `{valid_dir_in_root}/_annotations.coco.json`."
        )
        return

    st.session_state["train_img_path"] = train_images
    st.session_state["val_img_path"] = val_images

    # Run training
    _run_flashdet_training()


def _state_or(key, default):
    """Read config state with mirror support, falling back to *default*."""
    from flashstudio.utils import get_state
    value = get_state(key)
    return default if value is None else value


def _sanitize_run_name(run_name: str) -> str:
    name = str(run_name or "").strip() or "untitled_run"
    for ch in ("/", "\\", "\x00"):
        name = name.replace(ch, "_")
    name = name.replace("..", "_").strip(". ") or "untitled_run"
    return name[:120]


def _build_training_params(save_dir: str) -> dict:
    """Collect the Model page configuration into a JSON-serializable dict."""
    from flashstudio.utils import get_class_names_list

    size_map = {name: info["size"] for name, info in FLASHDET_MODELS.items()}
    arch_family = _state_or("arch_family", "FlashDet (recommended)")
    model_arch = _state_or("model_arch", "FlashDet-Pico")

    architecture = "flashdet"
    for family, arch in (
        ("YOLOv8", "yolov8"), ("YOLOv9", "yolov9"), ("YOLOv10", "yolov10"),
        ("YOLOv11", "yolov11"), ("YOLOX", "yolox"),
    ):
        if family in str(arch_family):
            architecture = arch
            break

    model_size = size_map.get(model_arch, "n")
    if architecture != "flashdet":
        model_size = str(_state_or("yolo_size", "n") or "n")
    input_size = int(_state_or("img_size", 320))
    class_names = get_class_names_list()
    num_classes = len(class_names) if class_names else int(_state_or("num_classes", 80) or 80)

    backbone_type = "lite"
    if model_arch == "FlashDet-Pico" and architecture == "flashdet":
        pico_bb = str(_state_or("pico_backbone", "") or "")
        if pico_bb in ("", "LiteBackbone", "LiteBackbone (default)"):
            backbone_type = "lite"
        elif "PicoBackbone" in pico_bb or "RepNeXt" in pico_bb or "pico_v2" in pico_bb:
            backbone_type = "pico_v2"
        else:
            backbone_type = pico_bb
    neck_type = _state_or("neck_type", "Auto")
    neck_type = None if str(neck_type) in ("", "Auto", "None") else str(neck_type)

    finetune_strategy = _state_or("finetune_strategy", "Full fine-tune")
    pretrain_option = _state_or("pretrain_option", "Random")
    custom_weights = str(_state_or("custom_weights", "") or "")
    finetune = None
    if pretrain_option == "Custom" and custom_weights:
        candidate = os.path.abspath(custom_weights)
        if os.path.isfile(candidate):
            finetune = candidate

    lora = finetune_strategy == "LoRA"
    lora_targets = _state_or("lora_targets", ["backbone", "neck"])
    if isinstance(lora_targets, str):
        lora_targets = [t.strip() for t in lora_targets.split(",") if t.strip()]
    valid_targets = {"backbone", "neck", "head", "fpn"}
    lora_targets = [t for t in lora_targets if t in valid_targets] or ["backbone", "neck"]

    resume_ckpt = None
    if st.session_state.get("resume_training"):
        candidate = str(st.session_state.get("resume_path", "") or "")
        if candidate:
            resume_ckpt = os.path.abspath(candidate)
        st.session_state["resume_training"] = False

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    warnings = []
    if bool(_state_or("multiscale", False)):
        warnings.append(
            "Multi-Scale is not supported by the installed FlashDet Trainer and was ignored."
        )
    if bool(_state_or("chunked_loss", False)):
        warnings.append(
            "Chunked Loss is not used by the installed FlashDet Trainer and was ignored."
        )
    if pretrain_option in ("COCO pretrained", "ImageNet backbone"):
        warnings.append(
            f"'{pretrain_option}' is not bundled with FlashDet; starting from random weights. "
            "Use 'Custom' to point at a checkpoint."
        )

    params = {
        "train_backend": st.session_state.get("train_backend", "FlashDet"),
        "train_images": st.session_state.get("train_img_path", ""),
        "val_images": st.session_state.get("val_img_path", ""),
        "class_names": class_names,
        "num_classes": num_classes,
        "model_arch": model_arch,
        "model_size": model_size,
        "architecture": architecture,
        "input_size": input_size,
        "backbone_type": backbone_type,
        "neck_type": neck_type,
        "device": device,
        "seed": int(_state_or("seed", 0) or 0),
        "ema_decay": float(_state_or("ema_decay", 0.9998) or 0.9998),
        "ema_warmup": int(_state_or("ema_warmup", 2000) or 2000),
        "nbs": int(_state_or("nbs", 64) or 64),
        "warmup_momentum": float(_state_or("warmup_momentum", 0.8) or 0.8),
        "base_momentum": float(_state_or("base_momentum", 0.937) or 0.937),
        "val_score_thr": float(_state_or("val_score_thr", 0.01) or 0.01),
        "max_det": int(_state_or("max_det", 300) or 300),
        "grad_clip_norm": float(_state_or("grad_clip_norm", 10.0) or 10.0),
        "channels_last": bool(_state_or("channels_last", False)),
        "cudnn_benchmark": bool(_state_or("cudnn_benchmark", False)),
        "tf32": bool(_state_or("tf32", False)),
        "epochs": _state_or("epochs", TRAIN_EPOCHS),
        "batch_size": _state_or("batch_size", TRAIN_BATCH_SIZE),
        "lr": _state_or("lr", TRAIN_LR),
        "weight_decay": _state_or("weight_decay", TRAIN_WEIGHT_DECAY),
        "warmup_epochs": _state_or("warmup_epochs", TRAIN_WARMUP_EPOCHS),
        "patience": _state_or("patience", TRAIN_PATIENCE),
        "grad_accum": _state_or("grad_accum", TRAIN_GRAD_ACCUM),
        "num_workers": _state_or("num_workers", TRAIN_NUM_WORKERS),
        "val_interval": _state_or("val_interval", TRAIN_VAL_INTERVAL),
        "lr_final_ratio": _state_or("lr_final_ratio", TRAIN_LR_FINAL_RATIO),
        "grad_clip": bool(_state_or("grad_clip", True)),
        "amp": bool(_state_or("amp", True)),
        "optimizer": _state_or("optimizer", DEFAULT_OPTIMIZER),
        "mosaic": bool(_state_or("aug_mosaic", True)),
        "mixup": bool(_state_or("aug_mixup", False)),
        "copy_paste": bool(_state_or("aug_copypaste", False)),
        "multiscale": bool(_state_or("multiscale", False)),
        "finetune_strategy": finetune_strategy,
        "pretrain_option": pretrain_option,
        "finetune": finetune,
        "lora": lora,
        "lora_variant": _state_or("lora_variant", "standard"),
        "lora_rank": _state_or("lora_rank", LORA_RANK_DEFAULT),
        "lora_alpha": _state_or("lora_alpha", LORA_ALPHA_DEFAULT),
        "lora_dropout": _state_or("lora_dropout", LORA_DROPOUT_DEFAULT),
        "lora_targets": lora_targets,
        "qlora": bool(_state_or("qlora", False)),
        "qlora_dtype": _state_or("qlora_dtype", QLORA_DTYPE_DEFAULT),
        "activation_checkpointing": bool(_state_or("activation_checkpointing", False)),
        "activation_offloading": bool(_state_or("activation_offloading", False)),
        "optimizer_in_bwd": bool(_state_or("optimizer_in_bwd", False)),
        "use_8bit_optimizer": bool(_state_or("use_8bit_optimizer", False)),
        "compile_model": bool(_state_or("compile_model", False)),
        "ddp": bool(_state_or("ddp", False)),
        "chunked_loss": bool(_state_or("chunked_loss", False)),
        "chunk_size": _state_or("chunk_size", CHUNKED_LOSS_CHUNK_SIZE),
        "yolo_width_mult": _state_or("yolo_width_mult", 1.0),
        "yolo_depth_mult": _state_or("yolo_depth_mult", 1.0),
        "yolo_use_pgi": bool(_state_or("yolo_use_pgi", True)),
        "yolo_use_psa": bool(_state_or("yolo_use_psa", True)),
        "yolo_use_c2psa": bool(_state_or("yolo_use_c2psa", True)),
        "resume": resume_ckpt,
        "save_dir": save_dir,
        "save_best": bool(_state_or("save_best", False)),
        "warnings": warnings,
    }
    return params


def _write_classes_file(save_dir: str, class_names: list) -> str | None:
    if not class_names:
        return None
    path = os.path.join(save_dir, "classes.txt")
    with open(path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    return path


def _run_flashdet_training():
    """Launch FlashDet training through the FlashStudio Trainer backend."""
    import subprocess
    import sys
    import time

    run_name = _sanitize_run_name(st.session_state.get("run_name", "untitled_run"))
    runs_root = os.path.abspath(_get_save_dir())
    default_run_path = os.path.join(runs_root, run_name)
    save_dir = os.path.abspath(st.session_state.get("active_run_path") or default_run_path)
    try:
        if os.path.commonpath([save_dir, runs_root]) != runs_root:
            save_dir = default_run_path
    except ValueError:
        save_dir = default_run_path

    os.makedirs(save_dir, exist_ok=True)
    params = _build_training_params(save_dir)

    class_file = _write_classes_file(save_dir, params["class_names"])
    if class_file:
        params["class_file"] = class_file

    manifest_path = os.path.join(save_dir, "train_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # Package root (FlashStudio/), so `-m flashstudio...` works from a source
    # checkout as well as an installed wheel.
    package_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        package_root + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    env["PYTHONUNBUFFERED"] = "1"

    log_path = os.path.join(save_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log_file_handle = open(log_path, "w", encoding="utf-8")
    cmd = [
        sys.executable, "-u", "-m",
        "flashstudio.utils.trainer_launcher", manifest_path,
    ]
    process = subprocess.Popen(
        cmd,
        stdout=log_file_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        cwd=package_root if os.path.isdir(package_root) else None,
    )
    st.session_state["_training_process"] = process
    st.session_state["_log_file_handle"] = log_file_handle
    st.session_state["training_pid"] = process.pid
    st.session_state["training_log_file"] = log_path
    st.session_state["training_backend"] = "flashdet.Trainer"
    st.session_state["training_manifest"] = manifest_path

    from flashstudio.pages.training.processes import write_pid_file
    write_pid_file(save_dir, process.pid)

    # Check for an immediate crash, but do not block the UI: a long sleep here
    # leaves the previous (dimmed) page visible until the rerun completes.
    # Later failures are picked up by _check_training_process on the next run.
    time.sleep(0.25)
    exit_code = process.poll()
    if exit_code is not None:
        from flashstudio.pages.training.processes import clear_pid_file
        clear_pid_file(save_dir)
        fh = st.session_state.pop("_log_file_handle", None)
        if fh and not fh.closed:
            try:
                fh.close()
            except Exception:
                pass
        st.session_state["_training_process"] = None
        st.session_state["training_active"] = False
        st.session_state["training_pid"] = None
        st.session_state["training_status"] = "Failed"
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as lf:
                log_content = lf.read()
        except OSError:
            log_content = "(could not read log)"
        from flashstudio.utils import flash
        flash("Training failed to start — see error below", "error")
        st.session_state["training_error"] = log_content[-4000:] if len(log_content) > 4000 else log_content
        st.rerun()
        return

    st.session_state["training_active"] = True
    st.session_state["training_status"] = "Running"
    from flashstudio.utils import flash
    flash(f"Training started (PID {process.pid})", "success")
    st.rerun()
