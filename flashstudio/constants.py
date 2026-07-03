"""FlashStudio — Centralized constants. Single source of truth for all magic values."""

import os

# ════════════════════════════════════════
# FILE EXTENSIONS
# ════════════════════════════════════════

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
IMG_EXTENSIONS_ALL = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
WEIGHT_EXTENSIONS = (".pt", ".pth", ".onnx", ".engine")
ARCHIVE_EXTENSIONS = (".zip", ".tar", ".gz", ".tar.gz", ".tgz")

# ════════════════════════════════════════
# CHECKPOINT FILE NAMES
# ════════════════════════════════════════

CKPT_BEST_INFERENCE = "model_best_inference.pth"
CKPT_BEST = "checkpoint_best.pth"
CKPT_LAST_INFERENCE = "model_last_inference.pth"
CKPT_LAST = "checkpoint_last.pth"
CKPT_FINAL_INFERENCE = "model_final_inference.pth"
CKPT_FINAL_FP16 = "model_final_fp16.pth"
CKPT_BEST_FP16 = "model_best_fp16.pth"

COMPLETE_MARKERS = (CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16, CKPT_BEST_INFERENCE, CKPT_BEST)
BEST_WEIGHT_PRIORITY = [CKPT_BEST_INFERENCE, CKPT_BEST, CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16]

TRAINING_LOG_CSV = "training_log.csv"

# ════════════════════════════════════════
# PATHS
# ════════════════════════════════════════

# Project root: FlashStudio/ directory (parent of flashstudio/ package)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = _PROJECT_ROOT

DEFAULT_SAVE_DIR = os.path.join(_PROJECT_ROOT, "flashstudio_workspaces")
DEFAULT_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
NETWORK_CHECK_URL = "http://images.cocodataset.org"
NETWORK_CHECK_TIMEOUT = 5

# ════════════════════════════════════════
# COLORS (for bounding boxes, visualization)
# ════════════════════════════════════════

BBOX_COLORS_RGB = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
]

BBOX_COLORS_HEX = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
]

# Theme colors
COLOR_PRIMARY = "#7C3AED"
COLOR_SUCCESS = "#10B981"
COLOR_WARNING = "#F59E0B"
COLOR_ERROR = "#EF4444"
COLOR_TEXT_PRIMARY = "#1A1A2E"
COLOR_TEXT_SECONDARY = "#6B7280"
COLOR_TEXT_MUTED = "#9CA3AF"
COLOR_TEXT_BODY = "#4B5563"
COLOR_BORDER = "#F0F0F5"
COLOR_BG_HIGHLIGHT = "#F5F3FF"

# ════════════════════════════════════════
# INFERENCE DEFAULTS
# ════════════════════════════════════════

INFER_CONF_THRESHOLD = 0.25
INFER_NMS_THRESHOLD = 0.45
INFER_IMG_SIZE = 640
INFER_NUM_CLASSES = 80
INFER_MAX_FRAMES = 300
INFER_STREAM_DURATION = 60
INFER_FRAME_SKIP = 1
INFER_DISPLAY_WIDTH = 650
INFER_DEFAULT_RESOLUTION = (640, 480)

# ════════════════════════════════════════
# EXPORT DEFAULTS
# ════════════════════════════════════════

EXPORT_IMG_SIZES = [320, 416, 640]
EXPORT_OPSET_MIN = 11
EXPORT_OPSET_MAX = 18
EXPORT_OPSET_DEFAULT = 13
EXPORT_FORMATS = ["ONNX", "TorchScript"]

EXPORT_WEIGHT_MAP = {
    "Best (inference)": [CKPT_BEST_INFERENCE, CKPT_BEST],
    "Best (FP16)": [CKPT_BEST_FP16],
    "Last": [CKPT_LAST_INFERENCE, CKPT_LAST],
}

# ════════════════════════════════════════
# TRAINING DEFAULTS (mirrors DEFAULTS in utils)
# ════════════════════════════════════════

TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
TRAIN_LR = 1e-3
TRAIN_IMG_SIZE = 320
TRAIN_WEIGHT_DECAY = 0.05
TRAIN_WARMUP_EPOCHS = 3
TRAIN_PATIENCE = 50
TRAIN_NUM_WORKERS = 4
TRAIN_GRAD_ACCUM = 1
TRAIN_VAL_INTERVAL = 5
TRAIN_LR_FINAL_RATIO = 0.1

BATCH_SIZE_OPTIONS = [2, 4, 8, 16, 32, 64, 128]
IMG_SIZE_OPTIONS = [320, 416, 640]

AUTOREFRESH_INTERVAL_MS = 5000
MAX_DISPLAY_RUNS = 8
VIS_QUEUE_SIZE = 3

VIS_DIR_NAMES = ("visualizations", "plots", "vis")
VIS_SKIP_FILE = "latest_visualization.jpg"
GT_VERIFICATION_DIR = "gt_verification"
GT_REPORT_FILE = "verification_report.json"
GT_SUMMARY_FILE = "verification_summary.txt"
TRAINING_LOG_GLOB = "train_*.log"
ONNX_MODEL_FILE = "model.onnx"
ONNX_DATA_FILE = "model.onnx.data"
CONFIG_YAML_FILE = "config.yaml"
RESULTS_JSON_FILE = "results.json"

# ════════════════════════════════════════
# MODEL ARCHITECTURE
# ════════════════════════════════════════

DEFAULT_MODEL_ARCH = "FlashDet-Pico"
DEFAULT_ARCH_FAMILY = "FlashDet (recommended)"
MODEL_SIZE_DEFAULT = "n"

ARCH_FAMILIES = ["FlashDet (recommended)", "YOLOv8", "YOLOv9", "YOLOv10", "YOLOv11", "YOLOX"]
OPTIMIZERS = ["AdamW", "SGD", "MuSGD"]
DEFAULT_OPTIMIZER = "AdamW"
LORA_VARIANTS = ["standard", "dora", "lora_plus", "adalora", "ortho", "lora_fa"]
LORA_RANK_DEFAULT = 8
LORA_ALPHA_DEFAULT = 16.0
LORA_DROPOUT_DEFAULT = 0.05
LORA_TARGETS_DEFAULT = ["backbone", "fpn"]
QLORA_DTYPE_DEFAULT = "int8"
CHUNKED_LOSS_CHUNK_SIZE = 1024
DEFAULT_FINETUNE_STRATEGY = "Full fine-tune"
DEFAULT_PRETRAIN_OPTION = "COCO pretrained"

FLASHDET_MODELS = {
    "FlashDet-Pico": {"size": "p", "params": "~298K", "speed": "Ultra-fast", "backbone": "LiteBackbone(0.5x)", "neck": "PicoNeck(64ch)", "head": "E2EDualHead", "for": "Edge/Mobile"},
    "FlashDet-Nano": {"size": "n", "params": "~790K", "speed": "Very fast", "backbone": "FlashBB(stem=32)", "neck": "PicoNeck(96ch)", "head": "E2EDualHead", "for": "IoT"},
    "FlashDet-Small": {"size": "s", "params": "~1.8M", "speed": "Fast", "backbone": "FlashBB(stem=48)", "neck": "PicoNeck(128ch)", "head": "E2EDualHead", "for": "General"},
    "FlashDet-Medium": {"size": "m", "params": "~3.6M", "speed": "Balanced", "backbone": "FlashBB(stem=64)", "neck": "PicoNeck(192ch)", "head": "E2EDualHead", "for": "Accuracy"},
    "FlashDet-Large": {"size": "l", "params": "~5.8M", "speed": "Accurate", "backbone": "FlashBB(stem=80)", "neck": "PicoNeck(256ch)", "head": "E2EDualHead", "for": "Accuracy"},
    "FlashDet-X": {"size": "x", "params": "~9.0M", "speed": "Max acc", "backbone": "FlashBB(stem=96)", "neck": "PicoNeck(320ch)", "head": "E2EDualHead", "for": "Server"},
}

# ════════════════════════════════════════
# ZONE DRAWING DEFAULTS
# ════════════════════════════════════════

ZONE_MAX_X = 1920
ZONE_MAX_Y = 1080
ZONE_DEFAULT_LINE = [(100, 240), (540, 240)]
ZONE_DEFAULT_POLYGON = "100,100\n500,100\n500,400\n100,400"

# ════════════════════════════════════════
# FONT
# ════════════════════════════════════════

def _resolve_font_path():
    """Find a usable TrueType font across platforms, with env override."""
    env = os.environ.get("FLASHSTUDIO_FONT_PATH", "")
    if env and os.path.isfile(env):
        return env
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",       # Debian/Ubuntu
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",      # Fedora/RHEL
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",                    # Arch
        "/System/Library/Fonts/Helvetica.ttc",                         # macOS
        "C:\\Windows\\Fonts\\arial.ttf",                               # Windows
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return ""

FONT_PATH = _resolve_font_path()
FONT_PATH_LINUX = FONT_PATH  # backward compat alias
FONT_SIZE_DEFAULT = 14

# ════════════════════════════════════════
# UI LIMITS
# ════════════════════════════════════════

MAX_PREVIEW_COLS = 4
MAX_PREVIEW_IMAGES = 6
MAX_WEIGHTS_DISPLAY = 4
SIDEBAR_LABEL_TRUNCATE = 20
GPU_NAME_TRUNCATE = 18
DATASET_NAME_TRUNCATE = 18

# ════════════════════════════════════════
# COCO CLASSES (standard 80-class COCO)
# ════════════════════════════════════════

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# ════════════════════════════════════════
# SIZE FORMATTING THRESHOLDS
# ════════════════════════════════════════

SIZE_GB = 1_073_741_824
SIZE_MB = 1_048_576
SIZE_KB = 1024


def format_bytes(total_bytes: int) -> str:
    """Format byte count to human-readable string."""
    if total_bytes > SIZE_GB:
        return f"{total_bytes / SIZE_GB:.1f} GB"
    elif total_bytes > SIZE_MB:
        return f"{total_bytes / SIZE_MB:.0f} MB"
    else:
        return f"{total_bytes / SIZE_KB:.0f} KB"
