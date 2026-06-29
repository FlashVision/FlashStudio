"""Launch tab — create run, configure, and start training."""

import os
import streamlit as st

from flashstudio.utils.device import has_cuda
from flashstudio.pages.training._common import _get_save_dir
from flashstudio.pages.training.launch.preflight import _run_preflight_checks
from flashstudio.pages.training.launch.runner import _start_training, _generate_run_name
from flashstudio.pages.training.launch.controls import _stop_training, _pause_training, _resume_active_training
from flashstudio.pages.training.launch.dialogs import _render_clean_workspace_dialog, _render_resume_dialog


def _render_start_tab():
    """Tab: launch a new training run — professional step-based layout."""

    runs_root = _get_save_dir()
    if runs_root:
        os.makedirs(runs_root, exist_ok=True)

    if "run_name" not in st.session_state:
        st.session_state["run_name"] = _generate_run_name()

    run_created = st.session_state.get("run_created", False)
    nc = st.session_state.get("num_classes", 0)

    with st.container(border=True):
        st.markdown("#### 1. Create Run")
        st.text_input("Run Name", key="run_name")
        run_name = st.session_state.get("run_name", "untitled_run")
        full_run_path = os.path.join(runs_root, run_name)
        st.caption(f"Output: `{full_run_path}`")

        if not run_created:
            if st.button("Create Run", key="btn_create_run", use_container_width=True, type="primary"):
                from flashstudio.utils import flash
                if not run_name.strip():
                    flash("Run name cannot be empty", "error")
                    st.rerun()
                else:
                    os.makedirs(full_run_path, exist_ok=True)
                    st.session_state["run_created"] = True
                    st.session_state["save_dir_run"] = full_run_path
                    flash(f"Run `{run_name}` created", "success")
                    st.rerun()
        else:
            st.success(f"Run: **{run_name}**")

    run_name = st.session_state.get("run_name", "untitled_run")
    full_run_path = os.path.join(runs_root, run_name)

    if not run_created:
        st.info("Create a run first to configure and start training.")
        return

    with st.container(border=True):
        st.markdown("#### 2. Config")
        cc1, cc2, cc3, cc4, cc5, cc6, cc7, cc8 = st.columns(8)
        with cc1:
            from flashstudio.utils import get_state
            st.metric("Model", get_state("model_arch").replace("FlashDet-", ""))
        with cc2:
            st.metric("Dataset", (st.session_state.get("dataset_name") or "—")[:8])
        with cc3:
            st.metric("Classes", nc if nc else "—")
        with cc4:
            st.metric("Epochs", get_state("epochs"))
        with cc5:
            st.metric("Batch", get_state("batch_size"))
        with cc6:
            st.metric("Img Size", get_state("img_size"))
        with cc7:
            st.metric("LR", f"{get_state('lr'):.1e}")
        with cc8:
            st.metric("Device", "GPU" if has_cuda() else "CPU")

    with st.container(border=True):
        st.markdown("#### 3. Launch")
        checks = _run_preflight_checks()
        all_ok = True
        check_html = []
        for label, ok, msg in checks:
            status = "OK" if ok else "Failed"
            check_html.append(f'{label}: {status}')
            if not ok:
                all_ok = False
        st.markdown(
            '<div class="ds-card-stats">' + ' '.join(f'<span>{c}</span>' for c in check_html) + '</div>',
            unsafe_allow_html=True,
        )

        b1, b2, b3, b4 = st.columns(4)
        with b1:
            if not st.session_state.get("training_active"):
                if st.button("Start", use_container_width=True, type="primary",
                             key="btn_start_training", disabled=not all_ok):
                    st.session_state.update({"training_active": True, "training_status": "Running",
                                             "training_paused": False, "active_run_path": full_run_path})
                    _start_training()
            else:
                if st.button("Stop", use_container_width=True, key="btn_stop_training"):
                    _stop_training()
        with b2:
            if st.session_state.get("training_active"):
                if st.session_state.get("training_paused"):
                    if st.button("Resume", use_container_width=True, key="btn_resume_active"):
                        _resume_active_training()
                else:
                    if st.button("Pause", use_container_width=True, key="btn_pause_training"):
                        _pause_training()
            else:
                st.button("Pause", use_container_width=True, key="btn_pause_disabled", disabled=True)
        with b3:
            if st.button("Resume Ckpt", use_container_width=True, key="btn_resume_training", help="Resume from checkpoint"):
                st.session_state["show_resume_dialog"] = True
        with b4:
            if st.button("Clean", use_container_width=True, key="btn_clean_workspace"):
                st.session_state["show_clean_dialog"] = True

    if st.session_state.get("show_clean_dialog"):
        _render_clean_workspace_dialog()

    if st.session_state.get("show_resume_dialog"):
        _render_resume_dialog()
