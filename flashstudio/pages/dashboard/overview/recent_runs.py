"""Dashboard — recent runs section."""

import os
import re
import time
import glob as glob_module
import streamlit as st
from flashstudio.pages.dashboard._common import (
    DEFAULT_SAVE_DIR, MAX_DISPLAY_RUNS,
    CKPT_BEST, CKPT_BEST_INFERENCE, CKPT_FINAL_INFERENCE,
    CKPT_FINAL_FP16, TRAINING_LOG_CSV,
    COLOR_SUCCESS, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED, COLOR_BORDER,
)


def render_recent_runs():
    from flashstudio.components.project_manager import get_active_project, get_project_runs_dir

    user_save_dir = st.session_state.get("save_dir", "")

    candidates = []
    active_proj = get_active_project()
    if active_proj:
        candidates.append(get_project_runs_dir(active_proj["id"]))
    if user_save_dir:
        candidates.append(user_save_dir)
    candidates.append(DEFAULT_SAVE_DIR)

    runs = []
    seen_dirs = set()
    seen_runs = set()

    def _looks_like_run(run_dir: str) -> bool:
        try:
            entries = os.listdir(run_dir)
        except OSError:
            return False
        return (
            any(e.startswith("train_") and e.endswith(".log") for e in entries)
            or TRAINING_LOG_CSV in entries
            or any(e.endswith(".pth") for e in entries)
        )

    for c in candidates:
        if not c:
            continue
        real = os.path.realpath(os.path.abspath(c))
        if real in seen_dirs or not os.path.isdir(real):
            continue
        seen_dirs.add(real)
        for d in os.listdir(real):
            rd = os.path.join(real, d)
            if not os.path.isdir(rd):
                continue
            real_run = os.path.realpath(rd)
            if real_run in seen_runs:
                continue
            if not _looks_like_run(rd):
                continue
            seen_runs.add(real_run)
            runs.append((d, rd))

    runs.sort(key=lambda item: os.path.getmtime(item[1]), reverse=True)
    runs = runs[:MAX_DISPLAY_RUNS]

    if not runs:
        st.caption("No runs yet. Create your first training run to see metrics here.")
        if st.button("Open Training", key="dash_open_training",
                     use_container_width=False):
            st.session_state["current_step"] = 3
            st.rerun()
        return

    for name, rd in runs:
        has_best = (os.path.isfile(os.path.join(rd, CKPT_BEST))
                    or os.path.isfile(os.path.join(rd, CKPT_BEST_INFERENCE)))
        has_final = (os.path.isfile(os.path.join(rd, CKPT_FINAL_INFERENCE))
                     or os.path.isfile(os.path.join(rd, CKPT_FINAL_FP16)))
        status = "Done" if (has_best or has_final) else "\u2014"
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
            from flashstudio.constants import TRAINING_LOG_GLOB
            log_files = glob_module.glob(os.path.join(rd, TRAINING_LOG_GLOB))
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
        import html
        name_html = html.escape(str(name)[:25])
        date_html = html.escape(date_str)
        info_html = html.escape(str(info))

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'font-size:0.85rem;padding:0.4rem 0;border-bottom:1px solid {COLOR_BORDER};">'
            f'<div>'
            f'<span style="font-weight:600;color:{COLOR_TEXT_PRIMARY};">{name_html}</span>'
            f'<span style="color:{COLOR_TEXT_MUTED};font-size:0.75rem;margin-left:0.5rem;">{date_html}</span>'
            f'</div>'
            f'<div style="display:flex;gap:0.8rem;align-items:center;">'
            f'<span style="color:{COLOR_TEXT_SECONDARY};">{info_html}</span>'
            f'<span style="color:{status_color};font-weight:600;">{status}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
