"""Download tab — Quick Start, Native, and External dataset sections."""

import os

import streamlit as st

from flashstudio.constants import INFER_NUM_CLASSES
from flashstudio.pages.data._common import (
    _get_native_datasets, _dataset_already_downloaded, _is_archive_url,
    _get_data_root, _resolve_download_root,
    EXTERNAL_DATASETS, QUICK_START,
)
from flashstudio.pages.data.helpers import (
    _use_existing, _start_flashdet_download, _start_external_download,
    _download_is_active, _build_tiny_2class,
)


def _render_download_location():
    """Let the user choose where downloaded datasets are stored."""
    default_root = os.path.join(_get_data_root(), "datasets")
    if "_download_dir_input" not in st.session_state:
        st.session_state["_download_dir_input"] = _resolve_download_root()

    lc1, lc2 = st.columns([5, 1])
    with lc1:
        st.text_input(
            "Save datasets to",
            key="_download_dir_input",
            help=(
                "Each dataset is saved in a subfolder named after it. "
                "Relative paths resolve inside the project data folder."
            ),
        )
    with lc2:
        if st.button(
            "Reset", key="download_dir_reset", use_container_width=True,
            help="Use the project default folder",
        ):
            st.session_state["_download_dir_input"] = default_root
            st.session_state["download_output_dir"] = default_root
            st.rerun()

    raw = str(st.session_state.get("_download_dir_input") or "").strip()
    if raw:
        expanded = os.path.expanduser(raw)
        resolved = (
            expanded if os.path.isabs(expanded)
            else os.path.join(_get_data_root(), expanded)
        )
        resolved = os.path.abspath(resolved)
    else:
        resolved = default_root
    st.session_state["download_output_dir"] = resolved

    if os.path.exists(resolved) and not os.path.isdir(resolved):
        st.error("That path points to a file, not a folder. Choose another folder.")
    else:
        note = f"Datasets save to `{resolved}/<dataset name>`"
        if not os.path.isdir(resolved):
            note += " · folder created on first download"
        if _download_is_active():
            note += " · a running download keeps its current folder"
        st.caption(note)


def _render_download():
    active = _download_is_active()
    if active:
        st.caption("A download is already running. Use the cancel control above.")
    _render_download_location()
    # Quick start — compact single row each
    st.markdown("#### Quick Start")
    for idx, ds in enumerate(QUICK_START):
        did = ds["id"]
        existing = _dataset_already_downloaded(did)
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f'<div style="font-size: var(--fs-text-md);"><b>{ds["name"]}</b> · '
                f'{ds["imgs"]} imgs · {ds["cls"]} cls · {ds["sz"]}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            label = "Use" if existing else "Get"
            if st.button(
                label, key=f"qs_{idx}", use_container_width=True, type="primary",
                disabled=active and not existing,
            ):
                if existing:
                    _use_existing(existing, ds)
                else:
                    _start_flashdet_download(did, ds["name"], ds.get("cls"))

    # Tiny local sample built from the bundled COCO128 images.
    st.markdown("#### Tiny 2-class sample")
    t1, t2 = st.columns([5, 1])
    with t1:
        st.caption("**Tiny 2-Class** · person + car · 24 images · built offline")
    with t2:
        if st.button("Build", key="build_tiny2", use_container_width=True):
            _build_tiny_2class()

    # Native datasets
    native = _get_native_datasets()
    if native:
        with st.expander(f"FlashDet Native ({len(native)} datasets)", expanded=False):
            for i, ds in enumerate(native):
                did = ds.get("id", ds.get("name", "").lower().replace(" ", ""))
                existing = _dataset_already_downloaded(did)
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.caption(f"**{ds.get('name', did)}** · {ds.get('classes', '?')} cls")
                with c2:
                    if st.button(
                        "Use" if existing else "Get", key=f"nat_{i}",
                        use_container_width=True,
                        disabled=active and not existing,
                    ):
                        if existing:
                            _use_existing(existing, ds)
                        else:
                            _start_flashdet_download(
                                did,
                                ds.get("name", did),
                                ds.get("classes", INFER_NUM_CLASSES),
                            )

    # External
    with st.expander(f"External Datasets ({len(EXTERNAL_DATASETS)})", expanded=False):
        st.caption(
            "Downloads are normalized to COCO. Supported inputs: COCO, VOC, "
            "YOLO, KITTI, LabelMe, CSV, and Supervisely. After loading, use "
            "Use a subset for a 10% tiny version."
        )
        for i, ds in enumerate(EXTERNAL_DATASETS):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f'<div style="font-size: var(--fs-text-md);"><b>{ds["name"]}</b> · '
                    f'{ds["imgs"]} · {ds["cls"]} cls · {ds["sz"]} · {ds["fmt"]}</div>',
                    unsafe_allow_html=True,
                )
                if ds.get("desc"):
                    st.caption(ds["desc"])
                conv = str(ds.get("convert", "") or "")
                if conv and conv.lower() != "none":
                    st.caption(f"Auto-convert: **{conv} → COCO** after loading")
                elif not _is_archive_url(ds.get("url", "")):
                    st.caption("Manual download · no automatic annotation conversion")
            with c2:
                url = ds.get("url", "")
                is_manual = not _is_archive_url(url)
                if is_manual:
                    st.link_button("Open", url, use_container_width=True)
                else:
                    if st.button(
                        "Get", key=f"ext_{i}", use_container_width=True,
                        disabled=active,
                    ):
                        _start_external_download(ds)
