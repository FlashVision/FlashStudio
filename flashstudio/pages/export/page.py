"""FlashStudio — Ship/Export page (framework-first)."""

import os

import streamlit as st

from flashstudio.constants import (
    CKPT_BEST_INFERENCE, CKPT_BEST, CKPT_LAST_INFERENCE, CKPT_LAST,
)


def _render_export_status(framework: str):
    """One-line readiness strip: checkpoint, artifacts, and the next action."""
    from flashstudio.pages.export._common import weights_with_mtime
    from flashstudio.pages.export.weights.settings import _get_export_save_dir

    weights = [
        path
        for path, _mtime in weights_with_mtime(framework, _get_export_save_dir())
    ]
    exported = st.session_state.get("exported_files") or []
    count = len(exported)

    if weights:
        checkpoint = os.path.basename(weights[0])
        if len(weights) > 1:
            checkpoint += f" (+{len(weights) - 1})"
        checkpoint_html = f"<b>{checkpoint}</b>"
    else:
        checkpoint_html = "<b>none</b>"

    if count:
        next_step = "download it below, then run it in Inference"
    elif weights:
        next_step = "choose a format and press Export"
    else:
        next_step = "train a model or download a starter checkpoint"

    files = f"{count} file" if count == 1 else f"{count} files"
    st.markdown(
        f'<div class="info-bar">'
        f"Checkpoint: {checkpoint_html} · "
        f"Exported: <b>{files}</b> · "
        f"Next: {next_step}"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_export_page():
    from flashstudio.components.styles import render_page_header
    from flashstudio.pages.export._common import EXPORT_FRAMEWORKS, FRAMEWORK_NOTES
    from flashstudio.utils import get_state, show_flashes

    render_page_header(
        "", "Export", "Convert, fuse, quantize, and benchmark trained weights",
    )
    show_flashes()

    if "export_framework" not in st.session_state:
        framework = str(get_state("model_framework") or "FlashDet")
        st.session_state["export_framework"] = (
            framework if framework in EXPORT_FRAMEWORKS else "FlashDet"
        )

    framework = st.radio(
        "Framework", EXPORT_FRAMEWORKS, horizontal=True, key="export_framework",
        help="Each framework has its own checkpoints and deployment options.",
    )
    previous = st.session_state.get("_export_framework_last")
    if previous and previous != framework:
        for key in (
            "exported_files", "opt_fusion", "opt_onnx", "opt_int8",
            "opt_benchmark", "ul_opt_benchmark",
        ):
            st.session_state.pop(key, None)
    st.session_state["_export_framework_last"] = framework
    st.caption(FRAMEWORK_NOTES.get(framework, ""))
    from flashstudio.components.dependencies import render_dependency_hint

    render_dependency_hint(
        framework, "converting and optimizing checkpoints",
        key_suffix=f"export_{framework.lower()}",
    )

    _render_export_status(framework)
    st.caption(
        "**Convert** creates the deployment file. **Optimize** fuses, quantizes, "
        "or benchmarks that file."
    )

    tab_convert, tab_optimize = st.tabs(["Convert", "Optimize"])
    with tab_convert:
        _render_convert(framework)
    with tab_optimize:
        from flashstudio.pages.export.optimization.tab import _render_optimization_tab

        _render_optimization_tab(framework)


def _render_convert(framework: str):
    from flashstudio.pages.export.weights.settings import render_export_settings

    render_export_settings(framework)

    with st.container(border=True):
        st.markdown("#### Output")
        exported = st.session_state.get("exported_files", [])
        if not exported:
            if framework == "Ultralytics":
                st.caption("Ultralytics weight names after training:")
                items = [
                    ("model_best_inference.pt", "Best · YOLO"),
                    ("model_last_inference.pt", "Latest · YOLO"),
                    ("best.pt", "Ultralytics best"),
                    ("last.pt", "Ultralytics last"),
                ]
            else:
                st.caption(f"{framework} weight names after training:")
                items = [
                    (CKPT_BEST_INFERENCE, "Best · inference"),
                    (CKPT_BEST, "Best · checkpoint"),
                    (CKPT_LAST_INFERENCE, "Latest · inference"),
                    (CKPT_LAST, "Latest · checkpoint"),
                ]
            for name, detail in items:
                st.markdown(
                    f'<span style="font-size:.84rem;color:var(--fs-body);">'
                    f'• `{name}` — {detail}</span>',
                    unsafe_allow_html=True,
                )
            st.caption("Manual export:")
            st.code(
                f"# {framework} export\n"
                "Choose a checkpoint and format above, then press Export.",
                language="bash",
            )
        else:
            for exp in exported:
                st.markdown(
                    f"**{exp['format']}** · "
                    f"`{os.path.basename(exp.get('path', ''))}`"
                )
                details = []
                if exp.get("architecture"):
                    details.append(str(exp["architecture"]).upper())
                if exp.get("input_size"):
                    details.append(f"{exp['input_size']}×{exp['input_size']}")
                if exp.get("opset"):
                    details.append(f"opset {exp['opset']}")
                if exp.get("size"):
                    details.append(str(exp["size"]))
                if details:
                    st.caption(" · ".join(details))
                if exp.get("path"):
                    st.caption(f"`{exp['path']}`")
                if exp.get("success"):
                    st.success("Ready")
                else:
                    st.error("Failed")
                if exp.get("success") and os.path.isfile(exp.get("path", "")):
                    with open(exp["path"], "rb") as f:
                        st.download_button(
                            "Download", f,
                            file_name=os.path.basename(exp["path"]),
                            key=f"dl_{exp['format']}",
                            use_container_width=True,
                        )
            with st.expander("Post-export"):
                st.code(
                    "trtexec --onnx=model.onnx --saveEngine=model.engine --fp16",
                    language="bash",
                )
