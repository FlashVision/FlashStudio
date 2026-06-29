"""Training runner — dataset validation, format conversion, and FlashDet subprocess launch."""

import os
import json
import streamlit as st

from flashstudio.constants import (
    FLASHDET_MODELS, DEFAULT_MODEL_ARCH, DEFAULT_ARCH_FAMILY, DEFAULT_SAVE_DIR,
    DEFAULT_DATA_DIR, TRAIN_WEIGHT_DECAY, TRAIN_VAL_INTERVAL, DEFAULT_OPTIMIZER,
    LORA_RANK_DEFAULT, LORA_ALPHA_DEFAULT, LORA_DROPOUT_DEFAULT, LORA_TARGETS_DEFAULT,
    QLORA_DTYPE_DEFAULT, CHUNKED_LOSS_CHUNK_SIZE, DEFAULT_FINETUNE_STRATEGY,
    DEFAULT_PRETRAIN_OPTION, VIS_QUEUE_SIZE,
)
from flashstudio.pages.training._common import _get_save_dir


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
    from flashdet.data import detect_dataset_format, convert_dataset, verify_dataset

    train_images = st.session_state.get("train_img_path", "")
    val_images = st.session_state.get("val_img_path", "")

    # Validate dataset paths exist
    if not train_images or not os.path.isdir(train_images):
        dataset_name = st.session_state.get("dataset_name", "")
        dataset_id = dataset_name.lower().replace(" ", "").replace("(demo)", "sample")
        for candidate_id in ["sample", "coco2017", "coco2017-val", "voc2007", "voc2012"]:
            if candidate_id in dataset_id or dataset_id in candidate_id:
                candidate_dir = os.path.join(DEFAULT_DATA_DIR, candidate_id)
                if os.path.isdir(os.path.join(candidate_dir, "train")):
                    train_images = os.path.join(candidate_dir, "train")
                    val_images = os.path.join(candidate_dir, "valid")
                    break

    # Convert to absolute paths to avoid relative path issues
    if train_images:
        train_images = os.path.abspath(train_images)
    if val_images:
        val_images = os.path.abspath(val_images)

    # Auto-find val path if missing but train exists
    if train_images and os.path.isdir(train_images) and (not val_images or not os.path.isdir(val_images)):
        parent = os.path.dirname(train_images)
        for vname in ("valid", "val", "test"):
            vpath = os.path.join(parent, vname)
            if os.path.isdir(vpath):
                val_images = vpath
                break
        if not val_images or not os.path.isdir(val_images):
            val_images = train_images

    if not train_images:
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "No dataset"
        st.session_state["training_error"] = (
            "No dataset paths configured. Go to the Data page and either "
            "enter train/val image paths manually, or download a dataset first."
        )
        return

    # Auto-detect classes from annotation if not already set
    _cls_raw = st.session_state.get("class_names", "")
    if isinstance(_cls_raw, list):
        _cls_raw = "\n".join(_cls_raw)
    if not _cls_raw.strip():
        ann_file = os.path.join(train_images, "_annotations.coco.json")
        if not os.path.isfile(ann_file):
            json_files = [f for f in os.listdir(train_images) if f.endswith(".json")] if os.path.isdir(train_images) else []
            if json_files:
                ann_file = os.path.join(train_images, json_files[0])
        if os.path.isfile(ann_file):
            try:
                with open(ann_file, encoding="utf-8") as _af:
                    ann_data = json.load(_af)
                cats = ann_data.get("categories", [])
                if cats:
                    sorted_cats = sorted(cats, key=lambda c: c.get("id", 0))
                    names = [c["name"] for c in sorted_cats]
                    st.session_state["class_names"] = "\n".join(names)
                    st.session_state["num_classes"] = len(names)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    # Auto-detect and convert format if not COCO
    parent_dir = os.path.dirname(train_images) if os.path.basename(train_images) in ("train", "images") else train_images
    detected_fmt = detect_dataset_format(parent_dir)

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

            convert_dataset(source_dir=parent_dir, output_dir=output_dir,
                            target_format="coco", class_names=class_names)

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

    # Verify COCO annotation file exists
    train_ann = os.path.join(train_images, "_annotations.coco.json")
    if not os.path.isfile(train_ann):
        ann_candidates = [f for f in os.listdir(train_images) if f.endswith(".json")] if os.path.isdir(train_images) else []
        if ann_candidates:
            st.info(f"Found annotation file: `{ann_candidates[0]}` (expected `_annotations.coco.json`)")
        else:
            st.session_state["training_active"] = False
            st.session_state["training_status"] = "Missing annotations"
            st.session_state["training_error"] = (
                f"Missing annotations! FlashDet expects _annotations.coco.json in: {train_images}"
            )
            return

    # Normalize dataset layout for FlashDet:
    # FlashDet computes data_root = dirname(train_images) and expects
    # data_root/valid/_annotations.coco.json to exist.
    data_root = os.path.dirname(os.path.normpath(train_images))
    valid_dir_in_root = os.path.join(data_root, "valid")
    val_ann_check = os.path.join(valid_dir_in_root, "_annotations.coco.json")

    if not os.path.exists(val_ann_check) and val_images and os.path.isdir(val_images):
        val_ann_file = os.path.join(val_images, "_annotations.coco.json")
        if os.path.isfile(val_ann_file):
            try:
                if os.path.exists(valid_dir_in_root):
                    if os.path.islink(valid_dir_in_root):
                        os.unlink(valid_dir_in_root)
                    else:
                        pass
                if not os.path.exists(valid_dir_in_root):
                    os.symlink(os.path.abspath(val_images), valid_dir_in_root)
                    val_images = valid_dir_in_root
                    st.session_state["val_img_path"] = val_images
            except OSError:
                pass

    if not os.path.exists(val_ann_check):
        try:
            os.makedirs(valid_dir_in_root, exist_ok=True)
            import shutil
            shutil.copytree(train_images, valid_dir_in_root, dirs_exist_ok=True)
            val_images = valid_dir_in_root
            st.session_state["val_img_path"] = val_images
        except Exception:
            pass

    st.session_state["train_img_path"] = train_images
    st.session_state["val_img_path"] = val_images

    # Run training
    _run_flashdet_training()


def _run_flashdet_training():
    """Run actual FlashDet training via CLI (non-blocking subprocess)."""
    import subprocess
    import sys

    size_map = {name: info["size"] for name, info in FLASHDET_MODELS.items()}

    from flashstudio.utils import get_state as _gs
    arch_family = _gs("arch_family")
    model_arch = _gs("model_arch")
    save_dir = st.session_state.get("active_run_path", os.path.join(_get_save_dir(), st.session_state.get("run_name", "untitled_run")))

    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass

    train_images = st.session_state.get("train_img_path", "")
    val_images = st.session_state.get("val_img_path", "")

    architecture = "flashdet"
    if "YOLOv8" in arch_family:
        architecture = "yolov8"
    elif "YOLOv9" in arch_family:
        architecture = "yolov9"
    elif "YOLOv10" in arch_family:
        architecture = "yolov10"
    elif "YOLOv11" in arch_family:
        architecture = "yolov11"
    elif "YOLOX" in arch_family:
        architecture = "yolox"

    from flashstudio.utils import get_state
    model_size = size_map.get(model_arch, "n")
    epochs = get_state("epochs")
    batch_size = get_state("batch_size")
    lr = get_state("lr")
    workers = get_state("num_workers")
    amp = st.session_state.get("amp", True) and device == "cuda"
    mosaic = st.session_state.get("aug_mosaic", True)
    mixup = st.session_state.get("aug_mixup", False)
    copy_paste = st.session_state.get("aug_copypaste", False)
    warmup_epochs = get_state("warmup_epochs")
    grad_accum = get_state("grad_accum")
    patience = get_state("patience")
    input_size = get_state("img_size")
    activation_checkpointing = st.session_state.get("activation_checkpointing", False)
    activation_offloading = st.session_state.get("activation_offloading", False)
    optimizer_in_bwd = st.session_state.get("optimizer_in_bwd", False)
    use_8bit_optimizer = st.session_state.get("use_8bit_optimizer", False)
    compile_model = st.session_state.get("compile_model", False)
    multi_gpu = st.session_state.get("ddp", False)

    # LoRA settings
    finetune_strategy = st.session_state.get("finetune_strategy", DEFAULT_FINETUNE_STRATEGY)
    lora = finetune_strategy == "LoRA"
    lora_rank = st.session_state.get("lora_rank", LORA_RANK_DEFAULT)
    lora_alpha = st.session_state.get("lora_alpha", LORA_ALPHA_DEFAULT)
    lora_dropout = st.session_state.get("lora_dropout", LORA_DROPOUT_DEFAULT)
    lora_variant = st.session_state.get("lora_variant", "standard")
    lora_targets = st.session_state.get("lora_targets", LORA_TARGETS_DEFAULT)

    # QLoRA
    qlora = st.session_state.get("qlora", False)
    qlora_dtype = st.session_state.get("qlora_dtype", QLORA_DTYPE_DEFAULT)

    # Chunked loss
    chunked_loss = st.session_state.get("chunked_loss", False)
    chunk_size = st.session_state.get("chunk_size", CHUNKED_LOSS_CHUNK_SIZE)

    # Pretrained / resume
    pretrain_option = st.session_state.get("pretrain_option", DEFAULT_PRETRAIN_OPTION)
    finetune_path = None
    if pretrain_option == "Custom":
        finetune_path = st.session_state.get("custom_weights", "")

    resume_ckpt = None
    if st.session_state.get("resume_training", False):
        resume_ckpt = st.session_state.get("resume_path", "")
        st.session_state["resume_training"] = False

    # Class file
    class_file = st.session_state.get("class_file", None)
    from flashstudio.utils import get_class_names_str
    class_names_raw = get_class_names_str()
    if class_names_raw.strip() and not class_file:
        import tempfile
        cls_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="classes_")
        for name in class_names_raw.strip().split("\n"):
            if name.strip():
                cls_tmp.write(name.strip() + "\n")
        cls_tmp.close()
        class_file = cls_tmp.name

    backbone_type = "lite"
    if model_arch == "FlashDet-Pico":
        pico_bb = st.session_state.get("pico_backbone", "")
        if "PicoBackbone" in pico_bb or "RepNeXt" in pico_bb:
            backbone_type = "pico_v2"

    # Build command-line arguments for FlashDet's full train.py
    # Locate train.py: try sibling FlashDet repo, then package location
    import flashdet
    flashdet_pkg_dir = os.path.dirname(flashdet.__file__)
    train_script_candidates = [
        os.path.join(flashdet_pkg_dir, "..", "train.py"),          # editable install
        os.path.join(flashdet_pkg_dir, "train.py"),                # inside package
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "..", "..", "..", "..", "FlashDet", "train.py"),  # sibling project
    ]
    train_script = None
    for candidate in train_script_candidates:
        if os.path.isfile(candidate):
            train_script = os.path.abspath(candidate)
            break

    if not train_script:
        st.session_state["training_active"] = False
        st.session_state["training_status"] = "Failed"
        st.session_state["training_error"] = (
            "Could not find FlashDet train.py. Ensure FlashDet is installed "
            "or the FlashDet repo is a sibling directory."
        )
        return

    cmd = [
        sys.executable, "-u", train_script,
        "--model-size", model_size,
        "--architecture", architecture,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--workers", str(workers),
        "--device", device,
        "--save-dir", save_dir,
        "--train-images", train_images,
        "--val-images", val_images,
        "--warmup-epochs", str(warmup_epochs),
        "--grad-accum", str(grad_accum),
        "--patience", str(patience),
        "--input-size", str(input_size),
        "--weight-decay", str(st.session_state.get("weight_decay", TRAIN_WEIGHT_DECAY)),
        "--optimizer", st.session_state.get("optimizer", DEFAULT_OPTIMIZER).lower(),
    ]

    val_interval = st.session_state.get("val_interval", TRAIN_VAL_INTERVAL)
    if val_interval:
        cmd += ["--val-interval", str(val_interval)]

    if amp:
        cmd.append("--amp")
    if mosaic:
        cmd.append("--mosaic")
    if mixup:
        cmd.append("--mixup")
    if copy_paste:
        cmd.append("--copy-paste")
    if st.session_state.get("multiscale", False):
        cmd.append("--multi-scale")
    if multi_gpu:
        cmd.append("--multi-gpu")

    if lora:
        cmd += ["--lora",
                "--lora-variant", lora_variant,
                "--lora-rank", str(lora_rank),
                "--lora-alpha", str(lora_alpha),
                "--lora-dropout", str(lora_dropout),
                "--lora-targets"] + lora_targets
    if qlora:
        cmd += ["--qlora", "--qlora-dtype", qlora_dtype]

    if activation_checkpointing:
        cmd.append("--activation-checkpointing")
    if activation_offloading:
        cmd.append("--activation-offloading")
    if optimizer_in_bwd:
        cmd.append("--optimizer-in-bwd")
    if use_8bit_optimizer:
        cmd.append("--use-8bit-optimizer")
    if compile_model:
        cmd.append("--compile")

    if finetune_path:
        cmd += ["--finetune", finetune_path]
    if resume_ckpt:
        cmd += ["--resume", resume_ckpt]
    if class_file:
        cmd += ["--class-file", class_file]
    if backbone_type != "lite":
        cmd += ["--backbone", backbone_type]

    os.makedirs(save_dir, exist_ok=True)

    import time as _time
    log_filename = f"train_{_time.strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(save_dir, log_filename)

    log_file_handle = open(log_path, "w")
    env = os.environ.copy()
    env["FLASHDET_VIS_QUEUE_SIZE"] = str(VIS_QUEUE_SIZE)
    process = subprocess.Popen(
        cmd,
        stdout=log_file_handle, stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    # Keep the file handle in session state so it stays open while the subprocess runs
    st.session_state["_log_file_handle"] = log_file_handle

    st.session_state["training_pid"] = process.pid
    st.session_state["training_log_file"] = log_path

    # Wait briefly to catch immediate crashes (import errors, bad args, etc.)
    import time as _tw
    _tw.sleep(2)
    exit_code = process.poll()
    if exit_code is not None:
        fh = st.session_state.pop("_log_file_handle", None)
        if fh and not fh.closed:
            try:
                fh.close()
            except Exception:
                pass
        st.session_state["training_active"] = False
        st.session_state["training_pid"] = None
        st.session_state["training_status"] = "Failed"
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as _lf:
                log_content = _lf.read()
        except OSError:
            log_content = "(could not read log)"
        from flashstudio.utils import flash
        flash("Training failed to start — see error below", "error")
        st.session_state["training_error"] = log_content[-2000:] if len(log_content) > 2000 else log_content
        st.rerun()
        return

    from flashstudio.utils import flash
    flash(f"Training started (PID {process.pid})", "success")
    st.rerun()
