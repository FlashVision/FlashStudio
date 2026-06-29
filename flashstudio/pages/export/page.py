"""FlashStudio — Export (zero-scroll, split layout)."""

import os
import streamlit as st
from flashstudio.constants import (
    EXPORT_OPSET_DEFAULT,
    CKPT_BEST_INFERENCE, CKPT_BEST, CKPT_LAST_INFERENCE, CKPT_LAST,
)


def render_export_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.utils import show_flashes
    render_page_header("", "Export")
    show_flashes()

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            from flashstudio.pages.export.weights.settings import render_export_settings
            wp, img_sz, export_fmt = render_export_settings()

            if st.button(f"Export {export_fmt}", use_container_width=True, type="primary"):
                from flashstudio.pages.export.weights.exporter import _run_export
                _run_export(wp, img_sz, st.session_state.get("export_opset", EXPORT_OPSET_DEFAULT),
                            st.session_state.get("export_dynamic", True),
                            export_fmt.lower())

    with right:
        with st.container(border=True):
            st.markdown("#### Output")
            exported = st.session_state.get("exported_files", [])
            if not exported:
                st.caption("Auto-saved weights from training:")
                for n, d in [(CKPT_BEST_INFERENCE, "Best, inference-ready"),
                              (CKPT_BEST, "Best, full checkpoint"),
                              (CKPT_LAST_INFERENCE, "Final, inference-ready"),
                              (CKPT_LAST, "Final, full checkpoint")]:
                    st.markdown(f'<span style="font-size:0.82rem;color:#4B5563;">• `{n}` — {d}</span>', unsafe_allow_html=True)
                st.caption("Manual:")
                st.code("flashdet export --model model.pth --output model.onnx", language="bash")
            else:
                for exp in exported:
                    st.markdown(f"**{exp['format']}** · `{os.path.basename(exp.get('path', ''))}`")
                    if exp.get("size"):
                        st.caption(exp["size"])
                    st.success("OK") if exp.get("success") else st.error("Failed")
                    if exp.get("success") and os.path.isfile(exp.get("path", "")):
                        with open(exp["path"], "rb") as f:
                            st.download_button("Download", f, file_name=os.path.basename(exp["path"]),
                                               key=f"dl_{exp['format']}", use_container_width=True)
                with st.expander("Post-export"):
                    st.code("trtexec --onnx=model.onnx --saveEngine=model.engine --fp16", language="bash")
