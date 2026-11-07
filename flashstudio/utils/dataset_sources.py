"""Dataset source URLs shared by the UI and the background download worker.

Kept free of Streamlit imports so it can run in a subprocess.
"""

from urllib.parse import urlparse

# The old host.robots.ox.ac.uk hostname refuses connections from some
# networks (including Google Colab).  thor.robots.ox.ac.uk is the current
# official host; pjreddie.com is a long-standing mirror.
_COCO = {
    "val_images": [
        "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
    ],
    "train_images": [
        "https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/train2017.zip",
    ],
    "annotations": [
        "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    ],
}
_COCO["train_annotations"] = _COCO["annotations"]

DATASET_URL_MIRRORS = {
    "coco2017": dict(_COCO),
    "coco2017-val": {
        "val_images": _COCO["val_images"],
        "annotations": _COCO["annotations"],
    },
    "sample": {
        "val_images": _COCO["val_images"],
        "annotations": _COCO["annotations"],
    },
    "voc2007": {
        "trainval": [
            "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "https://data.pjreddie.com/files/VOCtrainval_06-Nov-2007.tar",
        ],
        "test": [
            "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            "https://data.pjreddie.com/files/VOCtest_06-Nov-2007.tar",
        ],
    },
    "voc2012": {
        "trainval": [
            "https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "https://data.pjreddie.com/files/VOCtrainval_11-May-2012.tar",
        ],
    },
}


def as_url_list(urls) -> list[str]:
    if isinstance(urls, str):
        return [urls]
    if not urls:
        return []
    return [url for url in urls if url]


def archive_extension(url: str) -> str:
    path = urlparse(url).path.rsplit(".", 1)[-1].lower()
    if path in ("gz", "tgz"):
        return "tar.gz"
    return path


def get_dataset_urls(dataset_id: str) -> dict:
    """Return the URL map for a flashdet dataset, with mirror overrides."""
    try:
        from flashdet.data.download import DATASET_REGISTRY
    except ImportError:
        return {}
    info = DATASET_REGISTRY.get(dataset_id)
    if not isinstance(info, dict):
        return {}
    urls = dict(info.get("urls") or {})
    urls.update(DATASET_URL_MIRRORS.get(dataset_id, {}))
    return urls
