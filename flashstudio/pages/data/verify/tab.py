"""Verify tab — dataset verification and summary statistics."""

import os
import streamlit as st

from flashstudio.constants import IMG_EXTENSIONS_ALL, MAX_PREVIEW_COLS
from flashstudio.pages.data._common import (
    _HAS_FLASHDET, _FLASHDET_INSTALL_MSG, _load_data_prompt,
)

try:
    from flashdet.data import verify_dataset, summarize_coco_root
except ImportError:
    pass


def _list_files(path: str, extensions) -> list:
    if not path or not os.path.isdir(path):
        return []
    return [
        os.path.relpath(os.path.join(root, name), path)
        for root, _dirs, files in os.walk(path)
        for name in files
        if name.lower().endswith(extensions)
    ]


def _render_verify():
    train_path = st.session_state.get("train_img_path", "")
    val_path = st.session_state.get("val_img_path", "")
    valid_paths = [p for p in (train_path, val_path) if p and os.path.isdir(p)]
    if not valid_paths:
        _load_data_prompt(
            "No dataset loaded. Upload or download a dataset first.",
            "verify_load_data",
        )
        return

    dataset_root = st.session_state.get("dataset_output_path", "")
    if not dataset_root and train_path:
        from flashstudio.utils.dataset_paths import split_dir

        parent = os.path.dirname(split_dir(train_path))
        if os.path.isdir(parent):
            dataset_root = parent

    if dataset_root and os.path.isdir(dataset_root):
        from flashstudio.pages.data.helpers import _ensure_dataset_annotations

        _ensure_dataset_annotations(
            dataset_root,
            st.session_state.get("train_img_path", ""),
            st.session_state.get("val_img_path", ""),
        )

    can_verify = bool(dataset_root) and (
        os.path.isfile(os.path.join(dataset_root, "train", "_annotations.coco.json"))
        and os.path.isfile(os.path.join(dataset_root, "valid", "_annotations.coco.json"))
    )

    auto_verify = st.session_state.pop("_auto_verify", False)

    # FlashDet verify
    if can_verify and _HAS_FLASHDET:
        vc1, vc2 = st.columns([1, 5])
        with vc1:
            do_verify = st.button("Verify", type="primary", key="run_verify_btn", use_container_width=True)
        do_verify = do_verify or auto_verify
        with vc2:
            st.caption(f"Root: `{dataset_root}`")
        if do_verify:
            try:
                ok = verify_dataset(dataset_root)
                st.session_state["_last_verify_ok"] = ok
            except Exception as e:
                st.caption(str(e)[:60])
        if "_last_verify_ok" in st.session_state:
            ok = st.session_state["_last_verify_ok"]
            if ok:
                st.success("Verification **PASSED**")
            else:
                st.warning("Verification: issues detected")
                if st.button(
                    "Open annotation repair", key="verify_to_fix",
                    type="primary",
                ):
                    st.session_state["_data_tab_next"] = "Fix"
                    st.rerun()

        try:
            s = summarize_coco_root(dataset_root)
            splits = s.get("splits", {}) if isinstance(s, dict) else {}
            train_count = splits.get("train", {}).get("images", 0)
            val_count = (
                splits.get("valid", {}).get("images", 0)
                or splits.get("val", {}).get("images", 0)
            )
            has_summary = bool(s) and (train_count > 0 or val_count > 0)
            if has_summary:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Train", train_count)
                with m2:
                    st.metric("Val", val_count)
                with m3:
                    st.metric("Annotations", s.get("n_annotations", 0))
                with m4:
                    st.metric("Categories", len(s.get("class_names", [])))
                if s.get("distribution"):
                    with st.expander("Class Distribution"):
                        top = sorted(
                            s["distribution"].items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:15]
                        st.bar_chart({name: count for name, count in top})
                return
        except Exception:
            pass

    if dataset_root and os.path.isdir(dataset_root) and not can_verify:
        st.caption(
            "Custom split layout detected — training will normalize it automatically."
        )

    # Fallback for non-standard split names (train2017/val2017, nested
    # images folders, annotations/ siblings) that FlashDet's root summary
    # does not understand.
    if train_path or val_path:
        from flashstudio.utils.dataset_paths import summarize_coco_splits

        local = summarize_coco_splits(train_path, val_path)
        if local["train_images"] or local["val_images"]:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Train", local["train_images"])
            with m2:
                st.metric("Val", local["val_images"])
            with m3:
                st.metric("Annotations", local["annotations"])
            with m4:
                st.metric("Categories", len(local["class_names"]))
            if local["distribution"]:
                with st.expander("Class Distribution"):
                    top = sorted(
                        local["distribution"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:15]
                    st.bar_chart({name: count for name, count in top})
            return

    if not _HAS_FLASHDET and dataset_root and os.path.isdir(dataset_root):
        st.info(_FLASHDET_INSTALL_MSG)

    # Manual count — always show this as fallback
    ti_names = _list_files(train_path, IMG_EXTENSIONS_ALL)
    vi_names = _list_files(val_path, IMG_EXTENSIONS_ALL)
    ti = len(ti_names)
    vi = len(vi_names)

    # Count YOLO labels if present
    from flashstudio.pages.data.fix.annotations import _yolo_label_dir

    train_label_dir = _yolo_label_dir(
        train_path, st.session_state.get("train_label_path", "")
    )
    train_labels = len(_list_files(train_label_dir, (".txt",)))

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Train Images", ti)
    with m2:
        st.metric("Val Images", vi)
    with m3:
        st.metric("Labels", train_labels if train_labels else "—")
    with m4:
        status = "OK" if ti else "Empty"
        if ti and train_labels and train_labels == ti:
            status = "Complete"
        elif ti and train_labels and train_labels != ti:
            status = f"Partial ({train_labels}/{ti})"
        st.metric("Status", status)

    if train_path and os.path.isdir(train_path):
        st.caption(f"Train: `{train_path}`")
    if val_path and os.path.isdir(val_path):
        st.caption(f"Val: `{val_path}`")

    # Sample preview
    if ti_names:
        with st.expander("Samples"):
            cols = st.columns(min(MAX_PREVIEW_COLS, len(ti_names)))
            for i, img_name in enumerate(ti_names[:MAX_PREVIEW_COLS]):
                with cols[i]:
                    try:
                        st.image(os.path.join(train_path, img_name), caption=img_name[:15], use_container_width=True)
                    except Exception:
                        pass
