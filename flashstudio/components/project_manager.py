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
        try:
            with open(PROJECTS_INDEX, "r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "projects" in data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"projects": [], "active": None}


def _save_index(index: dict):
    _ensure_dirs()
    with open(PROJECTS_INDEX, "w") as f:
        json.dump(index, f, indent=2)


def get_project_dir(project_id: str) -> str:
    return os.path.join(PROJECTS_ROOT, project_id)


def get_project_data_dir(project_id: str) -> str:
    """Directory that owns this project's uploaded/downloaded datasets."""
    return os.path.join(get_project_dir(project_id), "data")


def get_project_runs_dir(project_id: str) -> str:
    """Directory that owns this project's training runs."""
    return os.path.join(get_project_dir(project_id), "runs")


def get_project_exports_dir(project_id: str) -> str:
    return os.path.join(get_project_dir(project_id), "exports")


def get_active_project_data_dir() -> Optional[str]:
    project = get_active_project()
    return get_project_data_dir(project["id"]) if project else None


def get_active_project_runs_dir() -> Optional[str]:
    project = get_active_project()
    return get_project_runs_dir(project["id"]) if project else None


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
    _clear_session_for_new_project()
    index = _load_index()
    index["active"] = project_id
    _save_index(index)
    load_project_state(project_id)
    _rebuild_config_mirror()
    st.session_state["_project_state_loaded"] = True


def delete_project(project_id: str):
    """Delete a project and all its data."""
    index = _load_index()
    was_active = index.get("active") == project_id
    if was_active:
        _clear_session_for_new_project()
    index["projects"] = [p for p in index["projects"] if p["id"] != project_id]
    if was_active:
        index["active"] = index["projects"][0]["id"] if index["projects"] else None
    _save_index(index)

    proj_dir = get_project_dir(project_id)
    if os.path.isdir(proj_dir):
        shutil.rmtree(proj_dir)

    if was_active:
        if index["active"]:
            load_project_state(index["active"])
            _rebuild_config_mirror()
        st.session_state["_project_state_loaded"] = bool(index["active"])


def delete_all_projects():
    """Delete all projects and reset everything."""
    _clear_session_for_new_project()
    index = _load_index()
    for p in index.get("projects", []):
        proj_dir = get_project_dir(p["id"])
        if os.path.isdir(proj_dir):
            shutil.rmtree(proj_dir)
    _save_index({"projects": [], "active": None})
    st.session_state["_project_state_loaded"] = False


def save_project_state():
    """Save current session state to the active project."""
    index = _load_index()
    active_id = index.get("active")
    if not active_id:
        return

    proj_dir = get_project_dir(active_id)
    if not os.path.isdir(proj_dir):
        return

    from flashstudio.utils.config import CONFIG_KEYS

    # Save relevant session state keys.  CONFIG_KEYS is the single source of
    # truth for model/training config; extra keys cover data + inference state.
    state_keys = tuple(dict.fromkeys(CONFIG_KEYS + (
        "dataset_classes", "dataset_id", "dataset_output_path",
        "train_ann_path", "val_ann_path", "detected_format",
        "class_file", "resume_path", "training_status", "exported_files",
        "infer_model_arch", "infer_conf", "infer_nms", "infer_img_size",
        "infer_device", "selected_solution",
    )))

    state = {}
    for key in state_keys:
        if key in st.session_state:
            val = st.session_state[key]
            if isinstance(val, (str, int, float, bool, list, dict, type(None))):
                state[key] = val

    state_file = os.path.join(proj_dir, "session_state.json")
    tmp_file = state_file + ".tmp"
    with open(tmp_file, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_file, state_file)

    # Update last_modified
    for p in index["projects"]:
        if p["id"] == active_id:
            p["last_modified"] = datetime.now().isoformat()
            break
    _save_index(index)


def _rebuild_config_mirror():
    """Rebuild ``_config_mirror`` from the current project's loaded state."""
    from flashstudio.utils.config import MIRROR_KEYS

    mirror = {k: st.session_state[k] for k in MIRROR_KEYS if k in st.session_state}
    if mirror:
        st.session_state["_config_mirror"] = mirror
    else:
        st.session_state.pop("_config_mirror", None)


def load_project_state(project_id: str):
    """Load a project's saved state into session.

    Also seeds the project's default save directory so runs live inside the
    project instead of a shared global workspace.  Explicit custom save paths
    are preserved.
    """
    from flashstudio.constants import DEFAULT_SAVE_DIR

    proj_dir = get_project_dir(project_id)
    runs_dir = get_project_runs_dir(project_id)
    os.makedirs(runs_dir, exist_ok=True)
    state_file = os.path.join(proj_dir, "session_state.json")

    state = {}
    if os.path.isfile(state_file):
        try:
            with open(state_file, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                state = loaded
        except (json.JSONDecodeError, OSError):
            state = {}

    for key, val in state.items():
        st.session_state[key] = val

    # A subprocess cannot survive a Streamlit session reload, so an active
    # status restored from disk is stale.
    if st.session_state.get("training_status") in ("Running", "Paused", "Resuming"):
        st.session_state["training_status"] = "Stopped"

    saved_save_dir = state.get("save_dir")
    if not saved_save_dir or saved_save_dir == DEFAULT_SAVE_DIR:
        st.session_state["save_dir"] = runs_dir

    # The monitor caches a workspace path; force it to re-read the new value.
    st.session_state.pop("monitor_workspace", None)


def duplicate_project(project_id: str, new_name: str) -> dict:
    """Duplicate an existing project with a new name."""
    index = _load_index()
    prev_active = index.get("active")
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

    # Copy session state, remapping project-local paths onto the duplicate.
    src_state = os.path.join(src_dir, "session_state.json")
    if os.path.isfile(src_state):
        try:
            with open(src_state, "r") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            state = {}
        src_data = get_project_data_dir(project_id)
        dst_data = get_project_data_dir(new_project["id"])
        for key in (
            "train_img_path", "val_img_path", "dataset_output_path",
            "train_ann_path", "val_ann_path", "train_label_path", "val_label_path",
        ):
            value = state.get(key)
            if isinstance(value, str) and value.startswith(src_data):
                state[key] = dst_data + value[len(src_data):]
        state["save_dir"] = get_project_runs_dir(new_project["id"])
        # Runs/exports are not copied, so stale artifact paths must not follow.
        state.pop("exported_files", None)
        with open(os.path.join(dst_dir, "session_state.json"), "w") as f:
            json.dump(state, f, indent=2)

    # Restore previously active project so duplication doesn't switch context
    if prev_active and prev_active != new_project["id"]:
        idx = _load_index()
        idx["active"] = prev_active
        _save_index(idx)

    return new_project


def get_project_stats(project_id: str) -> dict:
    """Get statistics for a project."""
    import re
    import csv
    import glob
    from flashstudio.constants import (
        DEFAULT_SAVE_DIR, TRAINING_LOG_CSV, TRAINING_LOG_GLOB,
    )

    proj_dir = get_project_dir(project_id)
    stats = {"runs": 0, "best_map": None, "total_size": "0 KB", "has_export": False}

    saved_save_dir = ""
    state_file = os.path.join(proj_dir, "session_state.json")
    if os.path.isfile(state_file):
        try:
            with open(state_file, "r") as f:
                saved_save_dir = json.load(f).get("save_dir", "")
        except (json.JSONDecodeError, OSError):
            saved_save_dir = ""

    # Project runs first, then the project's configured save dir, then the
    # legacy global workspace.  Deduplicate so a run is never counted twice.
    candidate_dirs = [get_project_runs_dir(project_id), saved_save_dir, DEFAULT_SAVE_DIR]
    seen_dirs = set()
    runs_dirs = []
    for d in candidate_dirs:
        if not d:
            continue
        real = os.path.realpath(os.path.abspath(d))
        if real in seen_dirs or not os.path.isdir(real):
            continue
        seen_dirs.add(real)
        runs_dirs.append(real)

    best_map = 0.0
    seen_runs = set()

    def _run_best_map(run_dir: str) -> float:
        csv_path = os.path.join(run_dir, TRAINING_LOG_CSV)
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, "r") as f:
                    rows = list(csv.DictReader(f))
                vals = [float(r.get("mAP@0.5", 0)) for r in rows if r.get("mAP@0.5", "").strip()]
                if vals:
                    return max(vals)
            except (OSError, ValueError, TypeError):
                pass
        logs = glob.glob(os.path.join(run_dir, TRAINING_LOG_GLOB))
        for lf in logs:
            try:
                with open(lf, "r", encoding="utf-8", errors="replace") as f:
                    for ln in f.readlines()[-20:]:
                        bm = re.search(r"Best mAP@0\.5:\s*([\d.]+)", ln)
                        if bm:
                            return float(bm.group(1))
            except OSError:
                pass
        return 0.0

    for runs_dir in runs_dirs:
        for run in os.listdir(runs_dir):
            run_path = os.path.join(runs_dir, run)
            if not os.path.isdir(run_path):
                continue
            try:
                entries = os.listdir(run_path)
            except OSError:
                continue
            if not (
                any(e.startswith("train_") and e.endswith(".log") for e in entries)
                or TRAINING_LOG_CSV in entries
                or any(e.endswith(".pth") for e in entries)
            ):
                continue
            real_run = os.path.realpath(run_path)
            if real_run in seen_runs:
                continue
            seen_runs.add(real_run)
            best_map = max(best_map, _run_best_map(run_path))
            if not stats["has_export"]:
                if any(f.endswith(".onnx") for f in entries):
                    stats["has_export"] = True

    stats["runs"] = len(seen_runs)
    if best_map > 0:
        stats["best_map"] = best_map

    exports_dir = get_project_exports_dir(project_id)
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
    active_idx = 0
    for i, p in enumerate(projects):
        if p["id"] == active_id:
            active_idx = i
            break

    selected_idx = st.selectbox(
        "Active Project",
        range(len(projects)),
        index=active_idx,
        format_func=lambda i: projects[i]['name'],
        key="project_selector",
    )

    if projects[selected_idx]["id"] != active_id:
        save_project_state()
        switch_project(projects[selected_idx]["id"])
        st.rerun()

    return True


def render_project_manager_page():
    """Full project management page — create, switch, delete, manage projects."""
    from flashstudio.components.styles import render_page_header
    render_page_header("", "Projects", "Create, switch, duplicate, or delete projects")

    index = _load_index()
    projects = index.get("projects", [])
    active_id = index.get("active")

    # ─── Create New Project ───
    with st.container(border=True):
        st.markdown("#### Create project")
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
                project = create_project(new_name, new_desc)
                _clear_session_for_new_project()
                load_project_state(project["id"])
                _rebuild_config_mirror()
                st.session_state["_project_state_loaded"] = True
                st.success(f"Project **{new_name}** created!")
                st.rerun()

    # ─── Project List ───
    if not projects:
        st.info("No projects yet. Create your first project above to get started.")
        return

    st.markdown(f"#### Projects ({len(projects)})")

    for p in sorted(projects, key=lambda x: x.get("last_modified", ""), reverse=True):
        is_active = p["id"] == active_id
        stats = get_project_stats(p["id"])

        with st.container(border=True):
            col_info, col_stats, col_actions = st.columns([3, 2, 2])

            with col_info:
                import html as _html
                name_html = _html.escape(str(p.get("name", "Untitled")))
                active_badge = ' <span class="ds-badge">Active</span>' if is_active else ""
                st.markdown(f"**{name_html}**{active_badge}", unsafe_allow_html=True)
                if p.get("description"):
                    st.caption(_html.escape(str(p["description"])))
                created = _html.escape(str(p.get("created", ""))[:10])
                modified = _html.escape(str(p.get("last_modified", ""))[:10])
                st.caption(f"Created: {created} · Modified: {modified}")

            with col_stats:
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Runs", stats["runs"])
                with sc2:
                    map_str = f"{stats['best_map']:.3f}" if stats["best_map"] is not None else "—"
                    st.metric("Best mAP", map_str)
                with sc3:
                    st.metric("Size", stats["total_size"])

            with col_actions:
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    if not is_active:
                        if st.button("Open", key=f"switch_{p['id']}", help="Switch to this project"):
                            save_project_state()
                            switch_project(p["id"])
                            st.rerun()
                with ac2:
                    if st.button("Duplicate", key=f"dup_{p['id']}", help="Duplicate project"):
                        dup = duplicate_project(p["id"], f"{p['name']} (copy)")
                        st.success(f"Duplicated as '{dup['name']}'")
                        st.rerun()
                with ac3:
                    if st.button("Delete", key=f"del_{p['id']}", help="Delete project"):
                        st.session_state[f"confirm_delete_{p['id']}"] = True

            # Delete confirmation
            if st.session_state.get(f"confirm_delete_{p['id']}"):
                st.error(f"Delete project **{p['name']}** and ALL its data? This cannot be undone.")
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
    with st.expander("Danger zone", expanded=False):
        st.error("These actions are irreversible!")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if st.button("Delete all projects", key="delete_all_projects"):
                st.session_state["confirm_delete_all"] = True

        if st.session_state.get("confirm_delete_all"):
            st.warning(f"This will delete **{len(projects)} projects** and all training data!")
            if st.button("Delete everything", type="primary", key="confirm_delete_all_btn"):
                delete_all_projects()
                _clear_session_for_new_project()
                st.session_state.pop("confirm_delete_all", None)
                st.rerun()
            if st.button("Cancel", key="cancel_delete_all"):
                st.session_state.pop("confirm_delete_all", None)
                st.rerun()


def _clear_session_for_new_project():
    """Clear all per-project session state when switching/creating a project."""
    import signal

    # Stop a dataset download that belongs to the old project.
    if st.session_state.get("_dataset_download"):
        try:
            from flashstudio.pages.data.helpers import _cancel_dataset_download

            _cancel_dataset_download()
        except Exception:
            pass

    # Stop a training process that belongs to the old project.
    proc = st.session_state.get("_training_process")
    pid = st.session_state.get("training_pid")
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                proc.terminate()
            except Exception:
                pass
    elif pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    fh = st.session_state.pop("_log_file_handle", None)
    if fh and not fh.closed:
        try:
            fh.close()
        except Exception:
            pass

    from flashstudio.utils.config import CONFIG_KEYS

    keys_to_clear = tuple(dict.fromkeys(CONFIG_KEYS + (
        # Dataset
        "dataset_classes", "dataset_id", "dataset_output_path",
        "train_ann_path", "val_ann_path", "detected_format",
        "train_label_path", "val_label_path", "_dataset_download",
        # Training runtime
        "training_active", "training_status", "training_pid", "training_error",
        "training_log_file", "training_paused", "training_start_time",
        "active_run_path", "save_dir_run", "run_created",
        "_run_name_at_create", "_training_process",
        # Inference inputs/results/zones
        "infer_weights_file", "infer_weights_path", "_infer_weights_tmp_path",
        "infer_images", "infer_video", "infer_stream_url", "infer_input_type",
        "infer_img_results", "video_results", "exported_files",
        "zone_line_points", "zone_polygons", "zone_draw_points",
        "zone_closed", "zone_draw_mode",
        # Monitor/dialog UI state
        "monitor_workspace", "monitor_auto_refresh", "selected_run_idx",
        "show_clean_dialog", "show_resume_dialog", "show_delete_run",
        "_flash_messages",
    )))

    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.session_state.pop("_config_mirror", None)
    st.session_state.pop("_project_state_loaded", None)
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
