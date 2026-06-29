"""Download tab — Quick Start, Native, and External dataset sections."""

import streamlit as st

from flashstudio.constants import INFER_NUM_CLASSES
from flashstudio.pages.data._common import (
    _get_native_datasets, _dataset_already_downloaded,
    EXTERNAL_DATASETS, QUICK_START,
)
from flashstudio.pages.data.helpers import (
    _use_existing, _run_flashdet_download, _run_external_download,
)


def _render_download():
    # Quick start — compact single row each
    st.markdown("#### Quick Start")
    for idx, ds in enumerate(QUICK_START):
        did = ds["id"]
        existing = _dataset_already_downloaded(did)
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f'<div style="font-size:0.84rem;"><b>{ds["name"]}</b> · '
                f'{ds["imgs"]} imgs · {ds["cls"]} cls · {ds["sz"]}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            label = "Use" if existing else "Get"
            if st.button(label, key=f"qs_{idx}", use_container_width=True, type="primary"):
                if existing:
                    _use_existing(existing, ds)
                else:
                    st.session_state.update({"dataset_name": ds["name"], "dataset_classes": ds["cls"], "dataset_id": did})
                    _run_flashdet_download(did)

    st.divider()

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
                    if st.button("Use" if existing else "Get", key=f"nat_{i}", use_container_width=True):
                        if existing:
                            _use_existing(existing, ds)
                        else:
                            st.session_state.update({"dataset_name": ds.get("name", did), "dataset_id": did, "dataset_classes": ds.get("classes", INFER_NUM_CLASSES)})
                            _run_flashdet_download(did)

    # External
    with st.expander(f"External Datasets ({len(EXTERNAL_DATASETS)})", expanded=False):
        for i, ds in enumerate(EXTERNAL_DATASETS):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f'<div style="font-size:0.84rem;"><b>{ds["name"]}</b> · '
                    f'{ds["imgs"]} · {ds["cls"]} cls · {ds["sz"]} · {ds["fmt"]}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                url = ds.get("url", "")
                is_manual = not url or "VisDrone" in url
                if st.button("Link" if is_manual else "Get", key=f"ext_{i}", use_container_width=True):
                    if is_manual:
                        st.info(f"Manual: [{url}]({url})")
                    else:
                        st.session_state.update({"dataset_name": ds["name"], "dataset_classes": ds["cls"]})
                        _run_external_download(ds)
