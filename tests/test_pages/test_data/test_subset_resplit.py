"""Re-applying a subset to its own output must not delete the source."""

import json
import os

from PIL import Image
from streamlit.testing.v1 import AppTest


def _write_split(root, split, count):
    images_dir = os.path.join(root, split, "images")
    os.makedirs(images_dir, exist_ok=True)
    images, annotations = [], []
    for i in range(count):
        name = f"{split}{i}.jpg"
        Image.new("RGB", (100, 80), "white").save(
            os.path.join(images_dir, name)
        )
        images.append({
            "id": i + 1, "file_name": name, "width": 100, "height": 80,
        })
        annotations.append({
            "id": i + 1, "image_id": i + 1, "category_id": 1,
            "bbox": [1, 1, 10, 10], "area": 100, "iscrowd": 0,
        })
    with open(os.path.join(images_dir, "_annotations.coco.json"), "w") as fh:
        json.dump({
            "images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "cat"}],
        }, fh)
    return images_dir


def test_subset_can_reapply_to_its_own_output(tmp_path):
    source = tmp_path / "source"
    train = _write_split(str(source), "train", 8)
    valid = _write_split(str(source), "valid", 2)
    out = str(tmp_path / "subset")

    body = f"""
import json, os, sys
sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))!r})
import streamlit as st
from flashstudio.pages.data.split.tab import _subset_coco

out = {out!r}
_subset_coco({train!r}, {valid!r}, {train!r} + "/_annotations.coco.json",
             {valid!r} + "/_annotations.coco.json", out, 50, 1, "QA")
# Re-apply using the subset as the new source and the same output path.
_subset_coco(
    st.session_state["train_img_path"], st.session_state["val_img_path"],
    st.session_state["train_ann_path"], st.session_state["val_ann_path"],
    out, 50, 1, "QA",
)

def count(path):
    with open(os.path.join(path, "_annotations.coco.json")) as fh:
        return len(json.load(fh)["images"])

st.write("COUNTS", count(os.path.join(out, "train", "images")),
         count(os.path.join(out, "val", "images")))
"""
    probe = tmp_path / "probe_subset.py"
    probe.write_text(body)
    at = AppTest.from_file(str(probe), default_timeout=120)
    at.run()

    assert not at.exception, [str(e.value) for e in at.exception]
    marks = [str(m.value) for m in at.markdown]
    assert any(m.startswith("COUNTS") and "2" in m and "1" in m for m in marks)
