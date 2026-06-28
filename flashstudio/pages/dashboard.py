"""FlashStudio — Dashboard."""

import os
import time
import glob as glob_module
import streamlit as st
from flashstudio.components.styles import render_page_header
from flashstudio.utils.device import get_gpu_info
from flashstudio.constants import (
    IMG_EXTENSIONS, DEFAULT_SAVE_DIR, MAX_DISPLAY_RUNS,
    CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_FINAL_INFERENCE,
    CKPT_FINAL_FP16, TRAINING_LOG_CSV,
    COLOR_SUCCESS, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED, COLOR_BORDER, COLOR_BG_HIGHLIGHT,
    GPU_NAME_TRUNCATE, DATASET_NAME_TRUNCATE,
)


def render_dashboard():
    from flashstudio.components.project_manager import get_active_project, get_project_stats
    from flashstudio.utils import show_flashes

    render_page_header("", "Dashboard")
    show_flashes()

    gpu = get_gpu_info()
    project = get_active_project()
    stats = get_project_stats(project["id"]) if project else None

    # ── Project info banner ──
    if project:
        proj_name = project.get("name", "Untitled")
        proj_desc = project.get("description", "")
        created = project.get("created", "")[:10]
        modified = project.get("last_modified", "")[:10]
        st.markdown(
            f'<div style="background:{COLOR_BG_HIGHLIGHT};border-left:4px solid {COLOR_SUCCESS};padding:0.8rem 1rem;'
            f'border-radius:0 8px 8px 0;margin-bottom:0.8rem;">'
            f'<div style="font-size:1.1rem;font-weight:700;color:{COLOR_TEXT_PRIMARY};">{proj_name}</div>'
            f'{"<div style=\"font-size:0.85rem;color:" + COLOR_TEXT_SECONDARY + ";margin-top:0.2rem;\">" + proj_desc + "</div>" if proj_desc else ""}'
            f'<div style="font-size:0.78rem;color:{COLOR_TEXT_MUTED};margin-top:0.3rem;">Created: {created} · Modified: {modified}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Key metrics — large numbers ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        with st.container(border=True):
            gpu_name = gpu["name"][:GPU_NAME_TRUNCATE] if gpu["available"] else "CPU only"
            vram = f"{gpu['memory_total']:.0f} GB" if gpu["available"] else "—"
            st.markdown(
                f'<div style="text-align:center;padding:0.4rem 0;">'
                f'<div style="font-size:0.78rem;color:{COLOR_TEXT_SECONDARY};font-weight:500;">GPU</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{COLOR_TEXT_PRIMARY};">{gpu_name}</div>'
                f'<div style="font-size:0.75rem;color:{COLOR_TEXT_MUTED};">VRAM: {vram}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with m2:
        with st.container(border=True):
            ds_name = (st.session_state.get("dataset_name") or "Not loaded")[:DATASET_NAME_TRUNCATE]
            train_path = st.session_state.get("train_img_path", "")
            img_count = 0
            if train_path and os.path.isdir(train_path):
                img_count = len([f for f in os.listdir(train_path) if f.lower().endswith(IMG_EXTENSIONS)])
            st.markdown(
                f'<div style="text-align:center;padding:0.4rem 0;">'
                f'<div style="font-size:0.78rem;color:{COLOR_TEXT_SECONDARY};font-weight:500;">Dataset</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{COLOR_TEXT_PRIMARY};">{ds_name}</div>'
                f'<div style="font-size:0.75rem;color:{COLOR_TEXT_MUTED};">{img_count} images</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with m3:
        with st.container(border=True):
            from flashstudio.utils import get_state
            model = get_state("model_arch")
            model_short = model.replace("FlashDet-", "")
            epochs = get_state("epochs")
            st.markdown(
                f'<div style="text-align:center;padding:0.4rem 0;">'
                f'<div style="font-size:0.78rem;color:{COLOR_TEXT_SECONDARY};font-weight:500;">Model</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{COLOR_TEXT_PRIMARY};">{model_short}</div>'
                f'<div style="font-size:0.75rem;color:{COLOR_TEXT_MUTED};">{epochs} epochs</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    with m4:
        with st.container(border=True):
            best_map = f"{stats['best_map']:.3f}" if stats and stats["best_map"] is not None else "—"
            runs_count = stats["runs"] if stats else 0
            st.markdown(
                f'<div style="text-align:center;padding:0.4rem 0;">'
                f'<div style="font-size:0.78rem;color:{COLOR_TEXT_SECONDARY};font-weight:500;">Best mAP</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{COLOR_SUCCESS if best_map != "—" else COLOR_TEXT_PRIMARY};">{best_map}</div>'
                f'<div style="font-size:0.75rem;color:{COLOR_TEXT_MUTED};">{runs_count} runs</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Two panels: Pipeline Status | Quick Actions ──
    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown("#### Pipeline Status")
            _pipeline_status()

    with right:
        with st.container(border=True):
            st.markdown("#### Quick Actions")
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                if st.button("Upload Data", use_container_width=True, key="dash_goto_data"):
                    st.session_state["current_step"] = 1
                    st.rerun()
                if st.button("Configure Model", use_container_width=True, key="dash_goto_model"):
                    st.session_state["current_step"] = 2
                    st.rerun()
            with ac2:
                if st.button("Start Training", use_container_width=True, type="primary", key="dash_goto_train"):
                    st.session_state["current_step"] = 3
                    st.rerun()
                if st.button("Export Model", use_container_width=True, key="dash_goto_export"):
                    st.session_state["current_step"] = 4
                    st.rerun()
            with ac3:
                if st.button("Run Inference", use_container_width=True, key="dash_goto_infer"):
                    st.session_state["current_step"] = 5
                    st.rerun()
                if st.button("Projects", use_container_width=True, key="dash_goto_proj"):
                    st.session_state["show_project_manager"] = True
                    st.rerun()

    # ── Recent Runs ──
    with st.container(border=True):
        st.markdown("#### Recent Runs")
        _runs()


def _pipeline_status():
    ds = st.session_state.get("dataset_name")
    model = st.session_state.get("model_arch")
    train_status = st.session_state.get("training_status", "")
    exported = st.session_state.get("exported_files")

    steps = [
        ("Data", ds is not None, ds or "Not loaded"),
        ("Model", model is not None, model.replace("FlashDet-", "") if model else "Not set"),
        ("Training", train_status in ("Complete", "Running"), train_status or "Not started"),
        ("Export", exported is not None, "Ready" if exported else "Pending"),
    ]

    for label, done, detail in steps:
        color = COLOR_SUCCESS if done else "#D1D5DB"
        text_color = COLOR_TEXT_PRIMARY if done else COLOR_TEXT_MUTED
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.6rem;padding:0.35rem 0;'
            f'border-bottom:1px solid {COLOR_BORDER};">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0;"></div>'
            f'<div style="flex:1;">'
            f'<span style="font-size:0.85rem;font-weight:600;color:{text_color};">{label}</span>'
            f'<span style="font-size:0.78rem;color:{COLOR_TEXT_MUTED};margin-left:0.5rem;">{detail}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


def _runs():
    from flashstudio.components.project_manager import get_active_project, get_project_dir

    candidates = []
    active_proj = get_active_project()
    if active_proj:
        candidates.append(os.path.join(get_project_dir(active_proj["id"]), "runs"))
    candidates += [
        DEFAULT_SAVE_DIR,
        os.path.join(os.getcwd(), "..", "FlashDet", "workspace"),
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "FlashDet", "workspace"),
    ]
    ws = None
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(c):
            ws = c; break

    if not ws:
        st.caption("No workspace found. Start training to create runs.")
        return

    dirs = sorted(
        [d for d in os.listdir(ws) if os.path.isdir(os.path.join(ws, d))],
        key=lambda d: os.path.getmtime(os.path.join(ws, d)), reverse=True,
    )[:MAX_DISPLAY_RUNS]

    if not dirs:
        st.caption("No runs yet. Go to Training to start your first run.")
        return

    import re
    for name in dirs:
        rd = os.path.join(ws, name)
        has_best = (os.path.isfile(os.path.join(rd, CKPT_BEST))
                    or os.path.isfile(os.path.join(rd, CKPT_BEST_INFERENCE)))
        has_final = (os.path.isfile(os.path.join(rd, CKPT_FINAL_INFERENCE))
                     or os.path.isfile(os.path.join(rd, CKPT_FINAL_FP16)))
        status = "Done" if (has_best or has_final) else "—"
        info = ""
        csv_path = os.path.join(rd, TRAINING_LOG_CSV)
        if os.path.isfile(csv_path):
            try:
                import csv as _csv
                with open(csv_path, "r") as _f:
                    rows = list(_csv.DictReader(_f))
                if rows:
                    vals = [float(r.get("mAP@0.5", 0)) for r in rows if r.get("mAP@0.5", "").strip()]
                    if vals:
                        info = f"mAP {max(vals):.3f}"
            except Exception:
                pass
        if not info:
            log_files = glob_module.glob(os.path.join(rd, "train_*.log"))
            if log_files:
                try:
                    best_log = max(log_files, key=lambda p: os.path.getsize(p))
                    with open(best_log, "r", encoding="utf-8", errors="replace") as _f:
                        for _ln in _f.readlines()[-10:]:
                            bm = re.search(r"Best mAP@0\.5:\s*([\d.]+)", _ln)
                            if bm:
                                info = f"mAP {bm.group(1)}"
                except OSError:
                    pass

        mtime = os.path.getmtime(rd)
        date_str = time.strftime("%b %d, %H:%M", time.localtime(mtime))
        status_color = COLOR_SUCCESS if (has_best or has_final) else COLOR_TEXT_MUTED

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'font-size:0.85rem;padding:0.4rem 0;border-bottom:1px solid {COLOR_BORDER};">'
            f'<div>'
            f'<span style="font-weight:600;color:{COLOR_TEXT_PRIMARY};">{name[:25]}</span>'
            f'<span style="color:{COLOR_TEXT_MUTED};font-size:0.75rem;margin-left:0.5rem;">{date_str}</span>'
            f'</div>'
            f'<div style="display:flex;gap:0.8rem;align-items:center;">'
            f'<span style="color:{COLOR_TEXT_SECONDARY};">{info}</span>'
            f'<span style="color:{status_color};font-weight:600;">{status}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
