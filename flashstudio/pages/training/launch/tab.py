"""Launch tab — create run, configure, and start training."""

import html as _html
import os

import streamlit as st

from flashstudio.utils.device import has_cuda
from flashstudio.pages.training._common import (
    _get_save_dir, _training_display_status,
)
from flashstudio.pages.training.launch.preflight import (
    _other_active_runs, _run_preflight_checks,
)
from flashstudio.pages.training.launch.runner import (
    _generate_run_name, _sanitize_run_name,
)
from flashstudio.pages.training.launch.dialogs import (
    _render_clean_workspace_dialog, _render_resume_dialog,
)


def _request_training_start(run_path):
    """Queue a start from a widget callback (runs before the page renders).

    Doing the state change in ``on_click`` avoids calling ``st.rerun`` from
    inside a tab body, which used to leave a second copy of the tab component
    in the DOM.
    """
    import time as _time

    st.session_state.update({
        "training_paused": False,
        "active_run_path": run_path,
        "training_start_time": _time.time(),
        "training_status": "Starting",
        "_start_training_pending": run_path,
    })


def _queue_training_action(action, payload=None):
    """Queue stop/pause/resume so the rerun happens before tabs render."""
    st.session_state["_training_action"] = (action, payload)


def _create_run(run_path, run_name):
    """Create the run folder and mark it ready from a widget callback."""
    from flashstudio.utils import flash

    if not str(run_name).strip():
        flash("Run name cannot be empty", "error")
        return
    os.makedirs(run_path, exist_ok=True)
    st.session_state["run_created"] = True
    st.session_state["_run_name_at_create"] = run_name
    st.session_state["save_dir_run"] = run_path
    if not st.session_state.get("training_active"):
        st.session_state["training_status"] = "Not started"
        st.session_state["training_paused"] = False
        st.session_state.pop("training_error", None)
        st.session_state.pop("training_start_time", None)
    flash(f"Run `{run_name}` created", "success")


def _set_cfg(key, value):
    """Update a config key in session state and the mirror."""
    st.session_state[key] = value
    mirror = st.session_state.get("_config_mirror")
    if isinstance(mirror, dict):
        mirror[key] = value
        st.session_state["_config_mirror"] = mirror


def _model_label() -> str:
    from flashstudio.utils import get_state

    framework = str(get_state("model_framework") or "FlashDet")
    if framework == "Ultralytics":
        series = str(get_state("yolo_series") or "YOLOv8")
        size = str(get_state("yolo_size") or "n").upper()
        return f"{series}-{size}"
    if framework == "YOLOX":
        return "YOLOX"
    arch = str(get_state("model_arch") or "FlashDet-Pico")
    return arch.replace("FlashDet-", "")


def _render_start_weights(backend: str):
    """Offer a pretrained download when no starting checkpoint is configured."""
    from flashstudio.utils import get_state
    from flashstudio.utils.weight_hub import (
        download_flashdet, download_ultralytics,
        flashdet_weights, ultralytics_weight_name,
    )

    pretrain = str(get_state("pretrain_option") or "Random")
    custom = str(get_state("custom_weights") or "")
    if pretrain == "Custom" and custom and os.path.isfile(os.path.abspath(custom)):
        st.caption(f"Starting checkpoint · `{os.path.basename(custom)}`")
        return

    st.markdown("#### Starting weights")
    if backend == "Ultralytics":
        family = str(get_state("arch_family") or "YOLOv8")
        if not family.startswith("YOLOv"):
            st.warning("Choose YOLOv8/v9/v10/v11 in Model → Architecture first.")
            return
        size = str(get_state("yolo_size") or "n")
        stem = ultralytics_weight_name(family, size)
        if st.button(f"Download {stem}.pt", key="launch_dl_ul",
                     use_container_width=True):
            try:
                path = download_ultralytics(stem)
                _set_cfg("pretrain_option", "Custom")
                _set_cfg("custom_weights", path)
                from flashstudio.utils import flash
                flash(f"Downloaded {stem}.pt", "success")
                st.rerun()
            except Exception as exc:
                st.error(f"Download failed: {exc}")
    else:
        entries = flashdet_weights()
        if not entries:
            st.caption("No FlashDet registry checkpoints are available.")
            return
        labels = {
            entry["name"]: (
                f"{entry['name']} · {entry.get('num_classes') or '?'} classes"
                + (" · cached" if entry.get("exists") else "")
            )
            for entry in entries
        }
        name = st.selectbox("Registry checkpoint", list(labels),
                            format_func=labels.get, key="launch_dl_fd_name")
        if st.button("Download and use", key="launch_dl_fd",
                     use_container_width=True):
            try:
                path = download_flashdet(name)
                _set_cfg("pretrain_option", "Custom")
                _set_cfg("custom_weights", path)
                from flashstudio.utils import flash
                flash(f"Downloaded {name}", "success")
                st.rerun()
            except Exception as exc:
                st.error(f"Download failed: {exc}")


def _render_preflight():
    checks = _run_preflight_checks()
    all_ok = True
    checks_html = []
    for label, ok, msg in checks:
        label_html = _html.escape(str(label))
        msg_html = _html.escape(str(msg))
        if not ok:
            checks_html.append(
                f'<span class="ds-check ds-fail">✕ {label_html} — {msg_html}</span>'
            )
            all_ok = False
        elif label in ("GPU", "Ignored options") and msg:
            checks_html.append(
                f'<span class="ds-check ds-note">! {label_html} — {msg_html}</span>'
            )
        else:
            detail = f" · {msg_html}" if msg else ""
            checks_html.append(
                f'<span class="ds-check ds-ok">✓ {label_html}{detail}</span>'
            )
    st.markdown(
        '<div class="ds-check-row">' + "".join(checks_html) + "</div>",
        unsafe_allow_html=True,
    )
    return all_ok


def _render_start_tab(empty_hint=None):
    """Create a run, configure it, and launch training.

    ``empty_hint`` optionally replaces the "create a run" notice shown before a
    run exists — the Training page uses it to surface the selected run's live
    summary right where the blocker used to be.
    """
    runs_root = _get_save_dir()
    if runs_root:
        os.makedirs(runs_root, exist_ok=True)

    if "run_name" not in st.session_state:
        # Streamlit can drop a widget key when its widget is not rendered for
        # one run (the Start button queues a run that exits before the Launch
        # tab is painted). Restore the created run name instead of generating
        # a new one, which used to reset run_created and hide Stop/Pause.
        st.session_state["run_name"] = (
            st.session_state.get("_run_name_at_create")
            or _generate_run_name()
        )

    prev_name = st.session_state.get("_run_name_at_create", "")
    current_name = st.session_state.get("run_name", "")
    if prev_name and current_name != prev_name:
        st.session_state["run_created"] = False
        # A different run is being configured; do not keep the previous run's
        # "Complete" status on the Launch panel.
        if not st.session_state.get("training_active"):
            st.session_state["training_status"] = "Not started"

    run_created = st.session_state.get("run_created", False)
    nc = st.session_state.get("num_classes", 0)

    with st.container(border=True):
        st.markdown("#### 1 · Name this run")
        name_col, button_col = st.columns([3, 1])
        with name_col:
            st.text_input(
                "Run name",
                key="run_name",
                label_visibility="collapsed",
                placeholder="e.g. traffic-night-v2",
            )
        run_name = st.session_state.get("run_name", "untitled_run")
        safe_run_name = _sanitize_run_name(run_name)
        full_run_path = os.path.join(runs_root, safe_run_name)

        with button_col:
            if not run_created:
                st.button(
                    "Create run", key="btn_create_run",
                    use_container_width=True, type="primary",
                    on_click=_create_run,
                    args=(full_run_path, run_name),
                )
            else:
                st.success("Created")
        st.caption(f"Output · `{full_run_path}`")
        if run_created:
            st.caption(f"Active run · **{run_name}**")

    if not run_created:
        if empty_hint:
            empty_hint()
        else:
            st.info("Create a run to start training.")
        return

    left, right = st.columns([3, 2])

    with left:
        with st.container(border=True):
            st.markdown("#### 2 · Training setup")
            from flashstudio.utils import get_state
            if "train_backend" not in st.session_state:
                st.session_state["train_backend"] = (
                    "Ultralytics"
                    if str(get_state("model_framework") or "FlashDet") == "Ultralytics"
                    else "FlashDet"
                )
            backend = st.selectbox(
                "Training backend",
                ["FlashDet", "Ultralytics"],
                index=0 if str(st.session_state.get("train_backend", "FlashDet")) != "Ultralytics" else 1,
                key="train_backend",
                help=(
                    "FlashDet trains any architecture in the Model page. "
                    "Ultralytics converts COCO to YOLO labels and trains "
                    "YOLOv8/9/10/11 with the Ultralytics engine."
                ),
            )
            if backend == "Ultralytics":
                st.caption("COCO labels are converted to YOLO `.txt` in the run folder.")
            from flashstudio.components.dependencies import render_dependency_hint

            render_dependency_hint(
                backend, "training",
                key_suffix=f"launch_{backend.lower()}",
            )

            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                st.metric("Model", _model_label())
            with cc2:
                st.metric("Dataset", st.session_state.get("dataset_name") or "—")
            with cc3:
                st.metric("Classes", nc if nc else "—")
            with cc4:
                st.metric("Epochs", get_state("epochs"))
            cc5, cc6, cc7, cc8 = st.columns(4)
            with cc5:
                st.metric("Batch", get_state("batch_size"))
            with cc6:
                st.metric("Image", get_state("img_size"))
            with cc7:
                st.metric("LR", f"{get_state('lr'):.1e}")
            with cc8:
                st.metric("Device", "GPU" if has_cuda() else "CPU")

            # A created run keeps its live progress visible in the setup panel,
            # not only in the run summary below the launch controls.
            from flashstudio.pages.training.progress import _render_training_progress
            _render_training_progress(full_run_path, show_eta=False)

            st.divider()
            _render_start_weights(backend)

    with right:
        with st.container(border=True):
            st.markdown("#### 3 · Launch")
            status = _training_display_status()
            active = bool(st.session_state.get("training_active"))
            if active:
                pid = st.session_state.get("training_pid")
                paused = st.session_state.get("training_paused")
                st.success(f"{'Paused' if paused else 'Running'} · PID {pid}")
            else:
                label = "Last run" if status in ("Complete", "Stopped", "Failed") else "Status"
                st.caption(f"{label} · **{status}**")

            all_ok = _render_preflight()
            if not all_ok:
                st.caption("Resolve the failed checks to enable Start.")
                # Offer a direct stop action for the run blocking this launch.
                for idx, other in enumerate(_other_active_runs()):
                    pid = other["pids"][0] if other.get("pids") else "?"
                    st.button(
                        f"Stop {other['name']} (PID {pid})",
                        key=f"stop_other_run_{idx}",
                        use_container_width=True,
                        on_click=_queue_training_action,
                        args=("stop_run", other["path"]),
                    )

            if not st.session_state.get("training_active"):
                st.button(
                    "Start training", use_container_width=True, type="primary",
                    key="btn_start_training", disabled=not all_ok,
                    on_click=_request_training_start,
                    args=(full_run_path,),
                )
            else:
                st.button(
                    "Stop training", use_container_width=True,
                    key="btn_stop_training",
                    on_click=_queue_training_action,
                    args=("stop",),
                )

            b1, b2, b3 = st.columns(3)
            with b1:
                if st.session_state.get("training_active"):
                    if st.session_state.get("training_paused"):
                        st.button(
                            "Resume", use_container_width=True,
                            key="btn_resume_active",
                            on_click=_queue_training_action,
                            args=("resume",),
                        )
                    else:
                        st.button(
                            "Pause", use_container_width=True,
                            key="btn_pause_training",
                            on_click=_queue_training_action,
                            args=("pause",),
                        )
                else:
                    st.button("Pause", use_container_width=True, key="btn_pause_disabled",
                              disabled=True)
            with b2:
                if st.button("Resume from checkpoint", use_container_width=True,
                             key="btn_resume_training",
                             help="Resume a run from a saved checkpoint"):
                    st.session_state["show_resume_dialog"] = True
            with b3:
                if st.button("Clean", use_container_width=True,
                             key="btn_clean_workspace", help="Clean the workspace"):
                    st.session_state["show_clean_dialog"] = True

    if st.session_state.get("show_clean_dialog"):
        _render_clean_workspace_dialog()
    if st.session_state.get("show_resume_dialog"):
        _render_resume_dialog()
