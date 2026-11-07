"""Dashboard — guided next steps."""

import os

import streamlit as st


def render_quick_actions():
    from flashstudio.utils import get_state

    dataset = bool(
        st.session_state.get("train_img_path")
        and os.path.isdir(str(st.session_state.get("train_img_path")))
    )
    model = bool(get_state("model_framework"))
    train_status = str(st.session_state.get("training_status") or "")
    trained = train_status == "Complete"
    exported = bool(st.session_state.get("exported_files"))
    inferred = bool(
        st.session_state.get("infer_img_results")
        or st.session_state.get("video_results")
    )

    steps = [
        ("Load a dataset", 1, dataset),
        ("Configure the model", 2, model),
        ("Train the model", 3, trained),
        ("Export for deployment", 4, exported),
        ("Run inference", 5, inferred),
    ]
    done = sum(1 for _label, _step, ok in steps if ok)
    st.caption(f"{done} of {len(steps)} steps complete")
    st.progress(done / len(steps))

    next_marked = False
    for label, step, ok in steps:
        left, right = st.columns([4, 1])
        with left:
            color = "#059669" if ok else "#D1D5DB"
            label_style = "color:var(--fs-text);" if ok else "color:var(--fs-body);"
            st.markdown(
                f'<div style="display:flex;gap:.6rem;align-items:flex-start;padding:.32rem 0;">'
                f'<span style="width:9px;height:9px;border-radius:50%;background:{color};'
                f'margin-top:.35rem;flex:0 0 9px;"></span>'
                f'<div><div style="font-size: var(--fs-text-md);font-weight:700;{label_style}">{label}</div></div>'
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
                if step == 1:
                    st.session_state["_data_tab_next"] = (
                        "Preview" if dataset else "Upload"
                    )
                st.rerun()
