"""Verify tab — dataset verification and summary statistics."""

import os
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS_ALL, MAX_PREVIEW_COLS
from flashstudio.pages.data._common import _HAS_FLASHDET, _FLASHDET_INSTALL_MSG

try:
    from flashdet.data import verify_dataset, summarize_coco_root
except ImportError:
    pass


def _render_verify():
    train_path = st.session_state.get("train_img_path", "")
    val_path = st.session_state.get("val_img_path", "")
    if not train_path and not val_path:
        st.info("No dataset loaded. Upload or download a dataset first.")
        return

    dataset_root = st.session_state.get("dataset_output_path", "")
    if not dataset_root and train_path:
        parent = os.path.dirname(os.path.normpath(train_path))
        if os.path.isdir(parent):
            dataset_root = parent

    # FlashDet verify
    if dataset_root and os.path.isdir(dataset_root) and _HAS_FLASHDET:
        vc1, vc2 = st.columns([1, 5])
        with vc1:
            do_verify = st.button("Verify", type="primary", key="run_verify_btn", use_container_width=True)
        with vc2:
            st.caption(f"Root: `{dataset_root}`")
        if do_verify or st.session_state.get("_last_verify_ok") is not None:
            try:
                ok = verify_dataset(dataset_root)
                st.session_state["_last_verify_ok"] = ok
                st.success("Verification **PASSED**") if ok else st.warning("Verification: issues detected")
            except Exception as e:
                st.caption(str(e)[:60])

        try:
            s = summarize_coco_root(dataset_root)
            if s:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Train", s.get("train_images", 0))
                with m2:
                    st.metric("Val", s.get("val_images", 0))
                with m3:
                    st.metric("Annotations", s.get("total_annotations", 0))
                with m4:
                    st.metric("Categories", s.get("num_categories", 0))
                if s.get("class_distribution"):
                    with st.expander("Class Distribution"):
                        top = sorted(s["class_distribution"].items(), key=lambda x: x[1], reverse=True)[:15]
                        st.bar_chart({n: c for n, c in top})
        except Exception:
            pass
        return

    if not _HAS_FLASHDET and dataset_root and os.path.isdir(dataset_root):
        st.info(_FLASHDET_INSTALL_MSG)

    # Manual count — always show this as fallback
    ti = [f for f in os.listdir(train_path) if f.lower().endswith(IMG_EXTENSIONS_ALL)] if train_path and os.path.isdir(train_path) else []
    vi = [f for f in os.listdir(val_path) if f.lower().endswith(IMG_EXTENSIONS_ALL)] if val_path and os.path.isdir(val_path) else []

    # Count YOLO labels if present
    train_label_dir = st.session_state.get("train_label_path", "")
    if not train_label_dir and train_path:
        candidate = train_path.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
        if os.path.isdir(candidate):
            train_label_dir = candidate
    train_labels = len([f for f in os.listdir(train_label_dir) if f.endswith(".txt")]) if train_label_dir and os.path.isdir(train_label_dir) else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Train Images", len(ti))
    with m2:
        st.metric("Val Images", len(vi))
    with m3:
        st.metric("Labels", train_labels if train_labels else "—")
    with m4:
        status = "OK" if ti else "Empty"
        if ti and train_labels and train_labels == len(ti):
            status = "Complete"
        elif ti and train_labels and train_labels != len(ti):
            status = f"Partial ({train_labels}/{len(ti)})"
        st.metric("Status", status)

    if train_path and os.path.isdir(train_path):
        st.caption(f"Train: `{train_path}`")
    if val_path and os.path.isdir(val_path):
        st.caption(f"Val: `{val_path}`")

    # Sample preview
    if ti:
        with st.expander("Samples"):
            cols = st.columns(min(MAX_PREVIEW_COLS, len(ti)))
            for i, img_name in enumerate(ti[:MAX_PREVIEW_COLS]):
                with cols[i]:
                    try:
                        st.image(os.path.join(train_path, img_name), caption=img_name[:15], use_container_width=True)
                    except Exception:
                        pass
