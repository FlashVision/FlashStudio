"""FlashStudio training hooks — callbacks and pre-training verification.

Used by the generated training subprocess to produce outputs that the
Training dashboard (Visualizations, Ground Truth, Curves tabs) expects.
"""

import os
import json
import csv
import numpy as np
from typing import Any, Dict, List

from flashdet.engine.core.callbacks import Callback


def run_gt_verification(
    train_images: str,
    val_images: str,
    save_dir: str,
    class_names: List[str],
    input_size: tuple = (320, 320),
    num_images: int = 3,
):
    """Generate GT verification images and a report JSON.

    Creates:
        save_dir/gt_verification/verification_report.json
        save_dir/gt_verification/verification_summary.txt
        save_dir/gt_verification/images/raw/*.jpg
    """
    from flashstudio.constants import GT_VERIFICATION_DIR
    gt_dir = os.path.join(save_dir, GT_VERIFICATION_DIR)
    raw_dir = os.path.join(gt_dir, "images", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    train_ann = os.path.join(train_images, "_annotations.coco.json")
    val_ann = os.path.join(val_images, "_annotations.coco.json") if val_images else None

    report = {"passed": True, "num_classes": len(class_names), "splits": {}}
    summary_lines = []

    for split_name, img_dir, ann_file in [
        ("train", train_images, train_ann),
        ("val", val_images, val_ann),
    ]:
        if not ann_file or not os.path.isfile(ann_file):
            continue
        try:
            with open(ann_file, encoding="utf-8") as f:
                ann_data = json.load(f)

            num_imgs = len(ann_data.get("images", []))
            num_anns = len(ann_data.get("annotations", []))
            num_cats = len(ann_data.get("categories", []))

            report["splits"][split_name] = {
                "coco": {
                    "num_images": num_imgs,
                    "num_annotations": num_anns,
                    "num_categories": num_cats,
                },
                "dataloader": {
                    "avg_boxes_per_sample": num_anns / max(num_imgs, 1),
                },
            }
            summary_lines.append(
                f"{split_name}: {num_imgs} images, {num_anns} annotations, {num_cats} categories"
            )

            if split_name == "train" and num_imgs > 0:
                _save_gt_images(
                    ann_data, img_dir, raw_dir, class_names,
                    num_images=min(num_images, num_imgs),
                )
        except Exception as e:
            summary_lines.append(f"{split_name}: ERROR - {e}")
            report["passed"] = False

    from flashstudio.constants import GT_REPORT_FILE, GT_SUMMARY_FILE
    with open(os.path.join(gt_dir, GT_REPORT_FILE), "w") as f:
        json.dump(report, f, indent=2)

    with open(os.path.join(gt_dir, GT_SUMMARY_FILE), "w") as f:
        f.write("\n".join(summary_lines))


def _save_gt_images(ann_data, img_dir, out_dir, class_names, num_images=8):
    """Draw GT boxes on sample images and save them."""
    try:
        import cv2
    except ImportError:
        return

    images_info = ann_data.get("images", [])[:num_images]
    annotations = ann_data.get("annotations", [])

    img_id_to_anns = {}
    for ann in annotations:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    cat_id_to_name = {}
    for cat in ann_data.get("categories", []):
        cat_id_to_name[cat["id"]] = cat["name"]

    colors = _make_palette(len(class_names))

    for img_info in images_info:
        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.isfile(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        anns = img_id_to_anns.get(img_info["id"], [])
        for ann in anns:
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = [int(v) for v in bbox]
            cat_id = ann.get("category_id", 0)
            cat_name = cat_id_to_name.get(cat_id, str(cat_id))
            color = colors.get(cat_id % len(colors), (0, 255, 0))

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{cat_name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - th - 4), (x + tw, y), color, -1)
            cv2.putText(img, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out_path = os.path.join(out_dir, f"gt_{img_info['id']:06d}.jpg")
        cv2.imwrite(out_path, img)


def _make_palette(n):
    """Generate a BGR color palette."""
    import colorsys
    palette = {}
    for i in range(max(n, 1)):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
        palette[i] = (int(b * 255), int(g * 255), int(r * 255))
    return palette


class StudioCSVLogger(Callback):
    """Callback that saves training_log.csv compatible with the dashboard parser."""

    FIELDNAMES = ["epoch", "train_loss", "lr", "val_loss", "mAP@0.5"]

    def __init__(self, save_dir: str):
        from flashstudio.constants import TRAINING_LOG_CSV
        self.csv_path = os.path.join(save_dir, TRAINING_LOG_CSV)
        self._initialized = False

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        row = {"epoch": epoch}
        if "train_loss" in metrics:
            row["train_loss"] = f"{metrics['train_loss']:.6f}"
        if "lr" in metrics:
            row["lr"] = f"{metrics['lr']:.8f}"
        if "val_loss" in metrics:
            row["val_loss"] = f"{metrics['val_loss']:.6f}"
        if "val_mAP" in metrics:
            row["mAP@0.5"] = f"{metrics['val_mAP']:.6f}"

        if not self._initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()
                writer.writerow(row)
            self._initialized = True
        else:
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(row)


class StudioVisualizationCallback(Callback):
    """Callback that saves GT-vs-Pred visualization images during validation."""

    def __init__(self, save_dir: str, max_kept: int = 10):
        from flashstudio.constants import VIS_DIR_NAMES
        self.vis_dir = os.path.join(save_dir, VIS_DIR_NAMES[0])
        self.max_kept = max_kept
        os.makedirs(self.vis_dir, exist_ok=True)
        self._epoch = 0

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        self._epoch = epoch

    def on_val_end(self, trainer: Any, metrics: Dict) -> None:
        """After validation, save a visualization from the model."""
        try:
            self._save_vis(trainer)
        except Exception:
            pass

    def _save_vis(self, trainer):
        import torch
        import cv2
        from flashdet.utils.visualization import make_gt_pred_panel

        model = trainer._get_eval_model() if hasattr(trainer, '_get_eval_model') else None
        if model is None:
            return

        val_loader = getattr(trainer, '_val_loader', None)
        if val_loader is None:
            return

        model.eval()
        device = trainer.device

        for images, gt_meta in val_loader:
            images = images.to(device)
            with torch.no_grad():
                results = model.predict(images, None, score_thr=0.3)

            img = images[0].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([123.675, 116.28, 103.53])
            std = np.array([58.395, 57.12, 57.375])
            img = np.clip(img * std + mean, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            gt_boxes = gt_meta["gt_bboxes"][0] if gt_meta.get("gt_bboxes") else np.empty((0, 4))
            gt_labels = gt_meta["gt_labels"][0] if gt_meta.get("gt_labels") else np.empty(0)

            pred_boxes = np.empty((0, 4))
            pred_labels = np.empty(0, dtype=int)
            pred_scores = np.empty(0)
            if results and len(results) > 0:
                dets, lbs = results[0]
                if dets is not None and dets.numel() > 0:
                    dets_np = dets.cpu().numpy()
                    pred_boxes = dets_np[:, :4]
                    pred_scores = dets_np[:, 4]
                    pred_labels = lbs.cpu().numpy().astype(int)

            panel = make_gt_pred_panel(
                img_bgr, gt_boxes,
                gt_labels.astype(int) if len(gt_labels) else gt_labels,
                pred_boxes, pred_labels, pred_scores,
                title_extra=f"| Epoch {self._epoch}",
            )

            out_path = os.path.join(self.vis_dir, f"epoch{self._epoch:04d}.jpg")
            cv2.imwrite(out_path, panel)

            self._cleanup()
            break

        model.train()

    def _cleanup(self):
        """Keep only the latest N visualization images."""
        files = sorted(
            [f for f in os.listdir(self.vis_dir) if f.endswith(".jpg")],
            key=lambda f: os.path.getmtime(os.path.join(self.vis_dir, f)),
        )
        while len(files) > self.max_kept:
            try:
                os.remove(os.path.join(self.vis_dir, files.pop(0)))
            except OSError:
                break
