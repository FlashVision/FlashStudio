"""FlashStudio — Project Management System.

Provides project isolation: each project has its own dataset, model config,
training runs, exports, and inference results. Users can create, switch,
duplicate, and delete projects.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Optional

import streamlit as st

PROJECTS_ROOT = os.path.join(os.path.expanduser("~"), ".flashstudio", "projects")
PROJECTS_INDEX = os.path.join(os.path.expanduser("~"), ".flashstudio", "projects.json")


def _ensure_dirs():
    os.makedirs(PROJECTS_ROOT, exist_ok=True)


def _load_index() -> dict:
    """Load the global project index."""
    _ensure_dirs()
    if os.path.isfile(PROJECTS_INDEX):
        with open(PROJECTS_INDEX, "r") as f:
            return json.load(f)
    return {"projects": [], "active": None}


def _save_index(index: dict):
    _ensure_dirs()
    with open(PROJECTS_INDEX, "w") as f:
        json.dump(index, f, indent=2)


def get_project_dir(project_id: str) -> str:
    return os.path.join(PROJECTS_ROOT, project_id)


def list_projects() -> list:
    """Return all projects with metadata."""
    index = _load_index()
    projects = []
    for p in index.get("projects", []):
        proj_dir = get_project_dir(p["id"])
        p["exists"] = os.path.isdir(proj_dir)
        if p["exists"]:
            p["size"] = _dir_size_str(proj_dir)
            runs_dir = os.path.join(proj_dir, "runs")
            p["num_runs"] = len(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else 0
        projects.append(p)
    return projects


def get_active_project() -> Optional[dict]:
    """Get the currently active project, or None."""
    index = _load_index()
    active_id = index.get("active")
    if not active_id:
        return None
    for p in index.get("projects", []):
        if p["id"] == active_id:
            return p
    return None


def create_project(name: str, description: str = "") -> dict:
    """Create a new project and set it as active."""
    index = _load_index()
    project_id = _slugify(name) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    proj_dir = get_project_dir(project_id)

    os.makedirs(proj_dir, exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "runs"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "exports"), exist_ok=True)
    os.makedirs(os.path.join(proj_dir, "inference_results"), exist_ok=True)

    project = {
        "id": project_id,
        "name": name,
        "description": description,
        "created": datetime.now().isoformat(),
        "last_modified": datetime.now().isoformat(),
    }

    # Save project config
    with open(os.path.join(proj_dir, "project.json"), "w") as f:
        json.dump(project, f, indent=2)

    index["projects"].append(project)
    index["active"] = project_id
    _save_index(index)

    return project


def switch_project(project_id: str):
    """Switch active project and load its state."""
    index = _load_index()
    index["active"] = project_id
    _save_index(index)
    _load_project_state(project_id)


def delete_project(project_id: str):
    """Delete a project and all its data."""
    index = _load_index()
    index["projects"] = [p for p in index["projects"] if p["id"] != project_id]
    if index.get("active") == project_id:
        index["active"] = index["projects"][0]["id"] if index["projects"] else None
    _save_index(index)

    proj_dir = get_project_dir(project_id)
    if os.path.isdir(proj_dir):
        shutil.rmtree(proj_dir)


def delete_all_projects():
    """Delete all projects and reset everything."""
    index = _load_index()
    for p in index.get("projects", []):
        proj_dir = get_project_dir(p["id"])
        if os.path.isdir(proj_dir):
            shutil.rmtree(proj_dir)
    _save_index({"projects": [], "active": None})


def save_project_state():
    """Save current session state to the active project."""
    index = _load_index()
    active_id = index.get("active")
    if not active_id:
        return

    proj_dir = get_project_dir(active_id)
    if not os.path.isdir(proj_dir):
        return

    # Save relevant session state keys
    state_keys = [
        "dataset_name", "dataset_classes", "dataset_id", "dataset_output_path",
        "train_img_path", "val_img_path", "train_ann_path", "val_ann_path",
        "ann_format", "class_names", "upload_num_classes",
        "model_arch", "arch_family", "epochs", "batch_size", "lr", "img_size",
        "finetune_strategy", "pretrain_option", "custom_weights",
        "lora_variant", "lora_rank", "lora_alpha", "lora_dropout", "lora_targets",
        "qlora", "qlora_dtype", "warmup_epochs", "patience", "num_workers",
        "amp", "aug_mosaic", "aug_mixup", "aug_copypaste",
        "activation_checkpointing", "activation_offloading", "optimizer_in_bwd",
        "use_8bit_optimizer", "compile_model", "ddp", "grad_accum",
        "chunked_loss", "chunk_size", "class_file",
        "save_dir", "run_name", "training_status",
        "infer_model_arch", "infer_conf", "infer_nms", "infer_img_size",
        "selected_solution",
    ]

    state = {}
    for key in state_keys:
        if key in st.session_state:
            val = st.session_state[key]
            if isinstance(val, (str, int, float, bool, list, dict, type(None))):
                state[key] = val

    state_file = os.path.join(proj_dir, "session_state.json")
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

    # Update last_modified
    for p in index["projects"]:
        if p["id"] == active_id:
            p["last_modified"] = datetime.now().isoformat()
            break
    _save_index(index)


def _load_project_state(project_id: str):
    """Load a project's saved state into session."""
    proj_dir = get_project_dir(project_id)
    state_file = os.path.join(proj_dir, "session_state.json")

    if not os.path.isfile(state_file):
        return

    with open(state_file, "r") as f:
        state = json.load(f)

    for key, val in state.items():
        st.session_state[key] = val


def duplicate_project(project_id: str, new_name: str) -> dict:
    """Duplicate an existing project with a new name."""
    src_dir = get_project_dir(project_id)
    new_project = create_project(new_name)
    dst_dir = get_project_dir(new_project["id"])

    # Copy data and config (not runs — those are large)
    for subdir in ("data",):
        src = os.path.join(src_dir, subdir)
        dst = os.path.join(dst_dir, subdir)
        if os.path.isdir(src):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)

    # Copy session state
    src_state = os.path.join(src_dir, "session_state.json")
    if os.path.isfile(src_state):
        shutil.copy2(src_state, os.path.join(dst_dir, "session_state.json"))

    return new_project


def get_project_stats(project_id: str) -> dict:
    """Get statistics for a project."""
    proj_dir = get_project_dir(project_id)
    stats = {"runs": 0, "best_map": None, "total_size": "0 KB", "has_export": False}

    runs_dir = os.path.join(proj_dir, "runs")
    if os.path.isdir(runs_dir):
        runs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        stats["runs"] = len(runs)

        # Find best mAP across runs
        best_map = 0
        for run in runs:
            results_file = os.path.join(runs_dir, run, "results.json")
            if os.path.isfile(results_file):
                with open(results_file) as f:
                    results = json.load(f)
                m = results.get("best_mAP50", 0)
                if m > best_map:
                    best_map = m
        if best_map > 0:
            stats["best_map"] = best_map

    exports_dir = os.path.join(proj_dir, "exports")
    if os.path.isdir(exports_dir) and os.listdir(exports_dir):
        stats["has_export"] = True

    stats["total_size"] = _dir_size_str(proj_dir)
    return stats


def render_project_selector():
    """Render the project management UI in the sidebar or top of page."""
    index = _load_index()
    active_id = index.get("active")
    projects = index.get("projects", [])

    if not projects:
        return False  # No projects exist — show creation UI

    # Quick selector
    project_names = {p["id"]: p["name"] for p in projects}
    active_idx = 0
    for i, p in enumerate(projects):
        if p["id"] == active_id:
            active_idx = i
            break

    selected_idx = st.selectbox(
        "Active Project",
        range(len(projects)),
        index=active_idx,
        format_func=lambda i: f"📁 {projects[i]['name']}",
        key="project_selector",
    )

    if projects[selected_idx]["id"] != active_id:
        save_project_state()
        switch_project(projects[selected_idx]["id"])
        st.rerun()

    return True


def render_project_manager_page():
    """Full project management page — create, switch, delete, manage projects."""
    st.markdown(
        "<div style='text-align:center; padding:1rem 0;'>"
        "<span style='font-size:2rem;'>📋</span> "
        "<b style='font-size:1.4rem;'>Project Manager</b>"
        "</div>",
        unsafe_allow_html=True,
    )

    index = _load_index()
    projects = index.get("projects", [])
    active_id = index.get("active")

    # ─── Create New Project ───
    with st.container(border=True):
        st.markdown("### ➕ Create New Project")
        col_name, col_desc, col_btn = st.columns([2, 3, 1])
        with col_name:
            new_name = st.text_input("Project Name", placeholder="e.g. PPE Detection v2",
                                     key="new_project_name")
        with col_desc:
            new_desc = st.text_input("Description (optional)",
                                     placeholder="Safety equipment detection for construction sites",
                                     key="new_project_desc")
        with col_btn:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("Create", type="primary", key="create_project_btn",
                         use_container_width=True, disabled=not new_name):
                proj = create_project(new_name, new_desc)
                st.success(f"✅ Project **{new_name}** created!")
                _clear_session_for_new_project()
                st.rerun()

    # ─── Project List ───
    if not projects:
        st.info("No projects yet. Create your first project above to get started.")
        return

    st.markdown(f"### 📁 Your Projects ({len(projects)})")

    for p in sorted(projects, key=lambda x: x.get("last_modified", ""), reverse=True):
        is_active = p["id"] == active_id
        proj_dir = get_project_dir(p["id"])
        stats = get_project_stats(p["id"])

        border_style = "border-left: 3px solid #7C3AED;" if is_active else ""
        with st.container(border=True):
            col_info, col_stats, col_actions = st.columns([3, 2, 2])

            with col_info:
                active_badge = " **[ACTIVE]**" if is_active else ""
                st.markdown(f"**{p['name']}**{active_badge}")
                if p.get("description"):
                    st.caption(p["description"])
                created = p.get("created", "")[:10]
                modified = p.get("last_modified", "")[:10]
                st.caption(f"Created: {created} · Modified: {modified}")

            with col_stats:
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Runs", stats["runs"])
                with sc2:
                    map_str = f"{stats['best_map']:.3f}" if stats["best_map"] else "—"
                    st.metric("Best mAP", map_str)
                with sc3:
                    st.metric("Size", stats["total_size"])

            with col_actions:
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    if not is_active:
                        if st.button("📂", key=f"switch_{p['id']}", help="Switch to this project"):
                            save_project_state()
                            switch_project(p["id"])
                            st.rerun()
                with ac2:
                    if st.button("📋", key=f"dup_{p['id']}", help="Duplicate project"):
                        dup = duplicate_project(p["id"], f"{p['name']} (copy)")
                        st.success(f"Duplicated as '{dup['name']}'")
                        st.rerun()
                with ac3:
                    if st.button("🗑️", key=f"del_{p['id']}", help="Delete project"):
                        st.session_state[f"confirm_delete_{p['id']}"] = True

            # Delete confirmation
            if st.session_state.get(f"confirm_delete_{p['id']}"):
                st.error(f"⚠️ Delete project **{p['name']}** and ALL its data? This cannot be undone.")
                dc1, dc2, dc3 = st.columns([1, 1, 3])
                with dc1:
                    if st.button("Yes, Delete", key=f"confirm_del_{p['id']}", type="primary"):
                        delete_project(p["id"])
                        st.session_state.pop(f"confirm_delete_{p['id']}", None)
                        st.rerun()
                with dc2:
                    if st.button("Cancel", key=f"cancel_del_{p['id']}"):
                        st.session_state.pop(f"confirm_delete_{p['id']}", None)
                        st.rerun()

    # ─── Danger Zone ───
    st.divider()
    with st.expander("⚠️ Danger Zone", expanded=False):
        st.error("These actions are irreversible!")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.button("🗑️ Delete ALL Projects", key="delete_all_projects"):
                st.session_state["confirm_delete_all"] = True

        if st.session_state.get("confirm_delete_all"):
            st.warning(f"This will delete **{len(projects)} projects** and all training data!")
            if st.button("CONFIRM: Delete Everything", type="primary", key="confirm_delete_all_btn"):
                delete_all_projects()
                _clear_session_for_new_project()
                st.session_state.pop("confirm_delete_all", None)
                st.rerun()
            if st.button("Cancel", key="cancel_delete_all"):
                st.session_state.pop("confirm_delete_all", None)
                st.rerun()


def _clear_session_for_new_project():
    """Clear session state keys when switching to a new/blank project."""
    keys_to_clear = [
        "dataset_name", "dataset_classes", "dataset_id", "dataset_output_path",
        "train_img_path", "val_img_path", "train_ann_path", "val_ann_path",
        "training_active", "training_status", "training_pid",
        "infer_img_results", "video_results", "exported_files",
        "run_name", "detected_format",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.session_state["current_step"] = 0


def _slugify(text: str) -> str:
    """Convert text to a safe directory name."""
    import re
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text)
    return text[:30]


def _dir_size_str(path: str) -> str:
    """Get human-readable directory size."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    if total > 1024 * 1024 * 1024:
        return f"{total / (1024**3):.1f} GB"
    elif total > 1024 * 1024:
        return f"{total / (1024**2):.0f} MB"
    return f"{total / 1024:.0f} KB"
