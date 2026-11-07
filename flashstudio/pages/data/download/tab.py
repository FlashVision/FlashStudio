"""Download tab — Quick Start, Native, and External dataset sections."""

import streamlit as st

from flashstudio.constants import INFER_NUM_CLASSES
from flashstudio.pages.data._common import (
    _get_native_datasets, _dataset_already_downloaded, _is_archive_url,
    EXTERNAL_DATASETS, QUICK_START,
)
from flashstudio.pages.data.helpers import (
    _use_existing, _start_flashdet_download, _start_external_download,
    _download_is_active,
)


def _render_download():
    active = _download_is_active()
    if active:
        st.caption("A download is already running. Use the cancel control above.")
    # Quick start — compact single row each
    st.markdown("#### Quick Start")
    for idx, ds in enumerate(QUICK_START):
        did = ds["id"]
        existing = _dataset_already_downloaded(did)
        c1, c2 = st.columns([5, 1])
        with c1:
            st.markdown(
                f'<div style="font-size:0.88rem;"><b>{ds["name"]}</b> · '
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
            "**Get** downloads a direct archive automatically. "
            "**Link** opens the project page — download those files manually and use the Upload tab."
        )
        for i, ds in enumerate(EXTERNAL_DATASETS):
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f'<div style="font-size:0.88rem;"><b>{ds["name"]}</b> · '
                    f'{ds["imgs"]} · {ds["cls"]} cls · {ds["sz"]} · {ds["fmt"]}</div>',
                    unsafe_allow_html=True,
                )
            with c2:
                url = ds.get("url", "")
                is_manual = not _is_archive_url(url)
                if st.button(
                    "Link" if is_manual else "Get", key=f"ext_{i}",
                    use_container_width=True,
                    disabled=active and not is_manual,
                ):
                    if is_manual:
                        st.info(f"Manual download: [{url}]({url})")
                    else:
                        _start_external_download(ds)
