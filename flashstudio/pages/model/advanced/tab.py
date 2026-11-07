"""FlashStudio — Advanced options tab."""

import streamlit as st


def _tab_adv():
    st.markdown("#### Memory, distributed & IO")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Memory")
            ac1, ac2 = st.columns(2)
            with ac1:
                st.checkbox("Activation checkpoint", False, key="activation_checkpointing")
                st.checkbox("Activation offload", False, key="activation_offloading")
            with ac2:
                st.checkbox("Optimizer in backward", False, key="optimizer_in_bwd")
                st.checkbox("torch.compile", False, key="compile_model")
            st.checkbox("8-bit optimizer", False, key="use_8bit_optimizer")
            st.checkbox("Chunked loss", False, key="chunked_loss",
                        help="Not used by the bundled FlashDet Trainer backend; ignored when enabled.")
            if st.session_state.get("chunked_loss"):
                st.number_input("Chunk size", 256, 4096, 1024, key="chunk_size")
                st.caption("Chunked loss is ignored by the bundled FlashDet Trainer backend.")
            st.markdown("#### Performance")
            st.checkbox("Channels-last (NHWC)", False, key="channels_last",
                        help="Convert the model to channels-last memory format.")
            st.checkbox("cuDNN benchmark", False, key="cudnn_benchmark",
                        help="Autotune cuDNN kernels (CUDA only).")
            st.checkbox("TF32 matmul", False, key="tf32",
                        help="Allow TF32 on Ampere+ GPUs (CUDA only).")

    with col2:
        with st.container(border=True):
            st.markdown("#### Distributed & IO")
            st.checkbox("Multi-GPU (DDP)", False, key="ddp")

            nc = st.session_state.get("num_classes", 0)
            cls_names = st.session_state.get("class_names", "")
            if isinstance(cls_names, list):
                cls_names = "\n".join(cls_names)
            if nc and cls_names.strip():
                names_list = [c.strip() for c in cls_names.strip().split("\n") if c.strip()]
                preview = ", ".join(names_list[:5])
                if len(names_list) > 5:
                    preview += f" ... (+{len(names_list) - 5})"
                st.caption(f"Classes ({nc}): {preview}")
            else:
                st.caption("Classes: not set — go to Data → Upload")

            st.text_input("Class file (.txt)", placeholder="Auto-generated from dataset classes",
                          key="class_file", help="Leave empty to auto-generate from dataset classes")

            from flashstudio.utils import DEFAULTS
            from flashstudio.components.project_manager import get_active_project_runs_dir
            default_save = get_active_project_runs_dir() or DEFAULTS["save_dir"]
            current_save = st.session_state.get("save_dir", "")
            if not str(current_save).strip() or current_save == DEFAULTS["save_dir"]:
                st.session_state["save_dir"] = default_save
            st.text_input("Save directory", key="save_dir", placeholder=default_save)
            rc1, rc2 = st.columns(2)
            with rc1:
                st.checkbox("Save best only", False, key="save_best",
                            help="Delete resumable 'latest' checkpoints after training.")
            with rc2:
                st.checkbox("Resume", False, key="resume_training")
            if st.session_state.get("resume_training"):
                st.text_input("Checkpoint", placeholder="checkpoint_last.pth", key="resume_path")
