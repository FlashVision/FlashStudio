"""Dashboard — guided next steps."""

import os

import streamlit as st

from flashstudio.pages.dashboard._common import model_display_name


def render_quick_actions():
    from flashstudio.utils import get_state

    dataset = bool(
        st.session_state.get("train_img_path")
        and os.path.isdir(str(st.session_state.get("train_img_path")))
    )
    model = bool(get_state("model_framework"))
    train_status = str(st.session_state.get("training_status") or "")
    trained = train_status in ("Complete", "Running")
    exported = bool(st.session_state.get("exported_files"))
    inferred = bool(
        st.session_state.get("infer_img_results")
        or st.session_state.get("video_results")
    )

    steps = [
        ("Load a dataset", "Import a ZIP or point at a COCO folder.", 1, dataset),
        ("Configure the model", f"Active model: {model_display_name()}", 2, model),
        ("Train the model", "Launch a run and watch metrics live.", 3, trained),
        ("Export for deployment", "ONNX, TorchScript, fusion, or INT8.", 4, exported),
        ("Run inference", "Images, video, or a live stream.", 5, inferred),
    ]
    done = sum(1 for _label, _detail, _step, ok in steps if ok)
    st.caption(f"{done} of {len(steps)} steps complete")
    st.progress(done / len(steps))

    next_marked = False
    for label, detail, step, ok in steps:
        left, right = st.columns([4, 1])
        with left:
            color = "#059669" if ok else "#D1D5DB"
            label_style = "color:var(--fs-text);" if ok else "color:var(--fs-body);"
            st.markdown(
                f'<div style="display:flex;gap:.6rem;align-items:flex-start;padding:.32rem 0;">'
                f'<span style="width:9px;height:9px;border-radius:50%;background:{color};'
                f'margin-top:.35rem;flex:0 0 9px;"></span>'
                f'<div><div style="font-size:.84rem;font-weight:700;{label_style}">{label}</div>'
                f'<div style="font-size:.73rem;color:var(--fs-muted);">{detail}</div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with right:
            is_next = not ok and not next_marked
            if is_next:
                next_marked = True
            if st.button(
                "Open" if not is_next else "Start",
                key=f"dash_step_{step}",
                type="primary" if is_next else "secondary",
                use_container_width=True,
            ):
                st.session_state["current_step"] = step
                st.rerun()
