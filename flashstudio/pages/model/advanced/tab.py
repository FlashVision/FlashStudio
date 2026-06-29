"""FlashStudio — Advanced options tab."""

import streamlit as st


def _tab_adv():
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### Memory")
            ac1, ac2 = st.columns(2)
            with ac1:
                st.checkbox("Act. Checkpoint", False, key="activation_checkpointing")
                st.checkbox("Act. Offload", False, key="activation_offloading")
            with ac2:
                st.checkbox("Optim in Bwd", False, key="optimizer_in_bwd")
                st.checkbox("torch.compile", False, key="compile_model")
            st.checkbox("8-bit Optimizer", False, key="use_8bit_optimizer")
            st.checkbox("Chunked Loss", False, key="chunked_loss")
            if st.session_state.get("chunked_loss"):
                st.number_input("Chunk Size", 256, 4096, 1024, key="chunk_size")

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

            st.text_input("Class File (.txt)", placeholder="auto-generated from dataset classes",
                          key="class_file", help="Leave empty to auto-generate from dataset classes")

            from flashstudio.utils import DEFAULTS
            if "save_dir" not in st.session_state:
                st.session_state["save_dir"] = DEFAULTS["save_dir"]
            st.text_input("Save Dir", key="save_dir")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.checkbox("Save Best Only", True, key="save_best")
            with rc2:
                st.checkbox("Resume", False, key="resume_training")
            if st.session_state.get("resume_training"):
                st.text_input("Checkpoint", placeholder="checkpoint_last.pth", key="resume_path")
