"""Training launch dialogs — clean workspace, resume, folder management, config."""

import os
import yaml
import streamlit as st

from flashstudio.utils.filesystem import dir_size_str
from flashstudio.utils.config import build_training_config, apply_training_config
from flashstudio.constants import (
    CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_BEST_FP16,
    CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16, CKPT_LAST, CKPT_LAST_INFERENCE,
    TRAINING_LOG_CSV,
)
from flashstudio.pages.training._common import _get_save_dir


_build_config_dict = build_training_config
_apply_config_dict = apply_training_config
_get_folder_size_str = dir_size_str
_get_dir_size = dir_size_str


def _render_folder_dialogs(runs_root: str, existing_folders: list):
    """Render rename/delete dialogs for folder management."""
    import shutil
    from flashstudio.utils import flash

    if st.session_state.get("wf_rename_target"):
        target = st.session_state["wf_rename_target"]
        with st.container(border=True):
            rename_val = st.text_input("New name", value=target, key="rename_wf_input")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Rename", type="primary", key="do_rename_wf"):
                    if rename_val and rename_val != target:
                        old_path = os.path.join(runs_root, target)
                        new_path = os.path.join(runs_root, rename_val)
                        if os.path.exists(new_path):
                            flash(f"Rename failed: `{rename_val}` already exists", "error")
                        else:
                            os.rename(old_path, new_path)
                            if st.session_state.get("active_workflow_folder") == target:
                                st.session_state["active_workflow_folder"] = rename_val
                                st.session_state["run_name"] = rename_val
                            flash(f"Renamed `{target}` → `{rename_val}`", "success")
                            st.session_state.pop("wf_rename_target", None)
                            st.rerun()
            with rc2:
                if st.button("Cancel", key="cancel_rename_wf"):
                    st.session_state.pop("wf_rename_target", None)
                    st.rerun()

    if st.session_state.get("wf_delete_target"):
        target = st.session_state["wf_delete_target"]
        with st.container(border=True):
            st.error(f"Delete **{target}** and all contents?")
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("Delete", type="primary", key="confirm_delete_wf"):
                    shutil.rmtree(os.path.join(runs_root, target), ignore_errors=True)
                    if st.session_state.get("active_workflow_folder") == target:
                        st.session_state.pop("active_workflow_folder", None)
                    st.session_state.pop("wf_delete_target", None)
                    flash(f"Deleted `{target}`", "success")
                    st.rerun()
            with dc2:
                if st.button("Cancel", key="cancel_delete_wf"):
                    st.session_state.pop("wf_delete_target", None)
                    st.rerun()

    if st.session_state.get("show_delete_all_wf_folders"):
        with st.container(border=True):
            st.error(f"Delete ALL {len(existing_folders)} folders? This is permanent!")
            dc1, dc2 = st.columns(2)
            with dc1:
                if st.button("DELETE ALL", type="primary", key="confirm_delete_all_wf"):
                    for f in existing_folders:
                        shutil.rmtree(os.path.join(runs_root, f), ignore_errors=True)
                    st.session_state.pop("active_workflow_folder", None)
                    st.session_state.pop("show_delete_all_wf_folders", None)
                    st.rerun()
            with dc2:
                if st.button("Cancel", key="cancel_delete_all_wf"):
                    st.session_state.pop("show_delete_all_wf_folders", None)
                    st.rerun()


def _render_config_file_dialog(run_path: str):
    """Compact save/load config dialog."""
    with st.container(border=True):
        tab_save, tab_load = st.tabs(["Save", "Load"])
        with tab_save:
            config = _build_config_dict()
            yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
            with st.expander("YAML Preview"):
                st.code(yaml_str, language="yaml")
            sc1, sc2 = st.columns(2)
            with sc1:
                st.download_button("Download", yaml_str, file_name="config.yaml", mime="text/yaml",
                                   use_container_width=True, type="primary")
            with sc2:
                if st.button("Save to Run", use_container_width=True, key="save_config_to_run"):
                    os.makedirs(run_path, exist_ok=True)
                    with open(os.path.join(run_path, "config.yaml"), "w") as f:
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    st.success("Saved")
        with tab_load:
            load_method = st.radio("From", ["Upload", "Path", "Runs"], key="config_load_method", horizontal=True)
            config_to_load = None
            if load_method == "Upload":
                uploaded = st.file_uploader("YAML", type=["yaml", "yml"], key="config_file_upload")
                if uploaded:
                    try:
                        config_to_load = yaml.safe_load(uploaded.read().decode("utf-8"))
                    except Exception as e:
                        st.error(str(e)[:40])
            elif load_method == "Path":
                cp = st.text_input("Path", placeholder="/path/config.yaml", key="config_path_input")
                if cp and os.path.isfile(cp):
                    try:
                        with open(cp) as f:
                            config_to_load = yaml.safe_load(f.read())
                    except Exception as e:
                        st.error(str(e)[:40])
            else:
                sd = _get_save_dir()
                parent = sd if os.path.isdir(sd) else ""
                found = [(d, os.path.join(parent, d, "config.yaml"))
                         for d in (os.listdir(parent) if parent and os.path.isdir(parent) else [])
                         if os.path.isfile(os.path.join(parent, d, "config.yaml"))]
                if found:
                    sel = st.selectbox("Run", range(len(found)), format_func=lambda i: found[i][0], key="config_from_run_select")
                    with open(found[sel][1]) as f:
                        config_to_load = yaml.safe_load(f.read())
            if config_to_load:
                if st.button("Apply", type="primary", key="apply_loaded_config", use_container_width=True):
                    _apply_config_dict(config_to_load)
                    st.rerun()


def _render_clean_workspace_dialog():
    """Compact workspace cleanup."""
    with st.container(border=True):
        parent_dir = _get_save_dir()

        if not os.path.isdir(parent_dir):
            st.info("Nothing to clean.")
            if st.button("Close", key="close_clean"):
                st.session_state["show_clean_dialog"] = False
                st.rerun()
            return

        runs = [{"name": d, "path": os.path.join(parent_dir, d), "size": _get_dir_size(os.path.join(parent_dir, d)),
                 "has_best": (os.path.isfile(os.path.join(parent_dir, d, CKPT_BEST))
                              or os.path.isfile(os.path.join(parent_dir, d, CKPT_BEST_INFERENCE)))}
                for d in sorted(os.listdir(parent_dir)) if os.path.isdir(os.path.join(parent_dir, d))]
        if not runs:
            st.info("No runs."); return

        clean_mode = st.radio("Mode", ["Selected", "Keep best only", "Delete incomplete", "Full reset"],
                              key="clean_mode", horizontal=True)

        if clean_mode == "Selected":
            selected = st.multiselect("Runs", [r["name"] for r in runs], key="runs_to_delete")
            if selected and st.button("Delete", type="primary", key="do_delete_selected"):
                import shutil
                for n in selected:
                    shutil.rmtree(os.path.join(parent_dir, n), ignore_errors=True)
                st.session_state["show_clean_dialog"] = False; st.rerun()
        elif clean_mode == "Keep best only":
            if st.button("Clean", type="primary", key="do_keep_best"):
                for r in runs:
                    _cleanup_run_keep_best(r["path"])
                st.session_state["show_clean_dialog"] = False; st.rerun()
        elif clean_mode == "Delete incomplete":
            incomplete = [r for r in runs if not r["has_best"]]
            st.caption(f"{len(incomplete)} incomplete")
            if incomplete and st.button("Delete", type="primary", key="do_delete_incomplete"):
                import shutil
                for r in incomplete:
                    shutil.rmtree(r["path"])
                st.session_state["show_clean_dialog"] = False; st.rerun()
        elif clean_mode == "Full reset":
            if st.button("DELETE ALL", type="primary", key="do_delete_all"):
                import shutil
                shutil.rmtree(parent_dir); os.makedirs(parent_dir, exist_ok=True)
                st.session_state["show_clean_dialog"] = False; st.rerun()

        if st.button("Cancel", key="cancel_clean"):
            st.session_state["show_clean_dialog"] = False; st.rerun()


def _render_resume_dialog():
    """Compact resume dialog."""
    with st.container(border=True):
        parent_dir = _get_save_dir()
        resumable = []
        if os.path.isdir(parent_dir):
            for d in sorted(os.listdir(parent_dir), reverse=True):
                full = os.path.join(parent_dir, d)
                ckpt = os.path.join(full, CKPT_LAST)
                if not os.path.isfile(ckpt):
                    ckpt = os.path.join(full, CKPT_LAST_INFERENCE)
                if os.path.isdir(full) and os.path.isfile(ckpt):
                    resumable.append({"name": d, "path": full, "ckpt": ckpt})
        if not resumable:
            st.info("No resumable runs.")
            if st.button("Close", key="close_resume"):
                st.session_state["show_resume_dialog"] = False; st.rerun()
            return
        rc1, rc2, rc3 = st.columns([4, 1, 1])
        with rc1:
            selected = st.selectbox("Run", [r["name"] for r in resumable], key="resume_run_select", label_visibility="collapsed")
        with rc2:
            if st.button("Resume", type="primary", key="do_resume"):
                from flashstudio.pages.training.launch.runner import _start_training
                for r in resumable:
                    if r["name"] == selected:
                        st.session_state.update({"resume_training": True, "resume_path": r["ckpt"],
                                                 "active_run_path": r["path"], "training_active": True,
                                                 "training_status": "Resuming", "show_resume_dialog": False})
                        _start_training(); break
        with rc3:
            if st.button("Cancel", key="cancel_resume"):
                st.session_state["show_resume_dialog"] = False; st.rerun()


def _cleanup_run_keep_best(run_dir: str) -> int:
    """Remove non-essential files from a run, keeping final/best models and results."""
    from flashstudio.constants import ONNX_MODEL_FILE, ONNX_DATA_FILE, VIS_DIR_NAMES, GT_VERIFICATION_DIR
    keep = {
        CKPT_BEST, TRAINING_LOG_CSV,
        CKPT_BEST_INFERENCE, CKPT_BEST_FP16,
        CKPT_FINAL_INFERENCE, CKPT_FINAL_FP16,
        ONNX_MODEL_FILE, ONNX_DATA_FILE,
    }
    removed = 0
    for f in os.listdir(run_dir):
        fpath = os.path.join(run_dir, f)
        if os.path.isfile(fpath) and f not in keep and not f.endswith(".log"):
            os.remove(fpath)
            removed += 1
        elif os.path.isdir(fpath) and f in (*VIS_DIR_NAMES, GT_VERIFICATION_DIR):
            import shutil
            shutil.rmtree(fpath)
            removed += 1
    return removed
