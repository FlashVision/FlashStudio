"""Helpers for tracking and locating training subprocesses.

The Streamlit session can lose its ``Popen`` handle (app restart, project
switch, browser reload). Writing the PID into the run folder makes the running
job recoverable, and matching ``/proc`` command lines is a fallback for runs
that were started before the PID file existed.
"""

import os

PID_FILE = "train.pid"
MANIFEST_FILE = "train_manifest.json"


def _proc_stat(pid):
    try:
        with open(f"/proc/{pid}/stat", "r") as fh:
            return fh.read()
    except OSError:
        return ""


def pid_is_alive(pid):
    """Return True only for a live process (zombies count as dead)."""
    parts = _proc_stat(pid).split()
    if len(parts) > 2:
        if parts[2] == "Z":
            return False
        return True
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def pid_is_paused(pid):
    """Return True when the process is stopped (SIGSTOP)."""
    parts = _proc_stat(pid).split()
    return len(parts) > 2 and parts[2] == "T"


def _ppid(pid):
    parts = _proc_stat(pid).split()
    try:
        return int(parts[3])
    except (IndexError, ValueError):
        return None


def _cmdline_args(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            data = fh.read()
    except OSError:
        return []
    return [
        part.decode("utf-8", "replace")
        for part in data.split(b"\x00")
        if part
    ]


def read_pid_file(run_dir):
    if not run_dir:
        return None
    try:
        with open(os.path.join(run_dir, PID_FILE), "r") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def write_pid_file(run_dir, pid):
    if not run_dir:
        return
    try:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, PID_FILE), "w") as fh:
            fh.write(str(pid))
    except OSError:
        pass


def clear_pid_file(run_dir):
    if not run_dir:
        return
    try:
        os.remove(os.path.join(run_dir, PID_FILE))
    except OSError:
        pass


def find_run_training_pids(run_dir):
    """Return live trainer-launcher root PIDs for *run_dir*.

    Uses the run's PID file first, then falls back to matching the run's
    ``train_manifest.json`` in ``/proc/*/cmdline`` so jobs launched before the
    PID file existed can still be recovered and stopped.
    """
    if not run_dir or not os.path.isdir(run_dir):
        return []

    found = []
    pid = read_pid_file(run_dir)
    if pid and pid_is_alive(pid):
        found.append(pid)

    manifest = os.path.abspath(os.path.join(run_dir, MANIFEST_FILE))
    if not os.path.isfile(manifest):
        return found

    for candidate in running_trainer_manifests().get(manifest, []):
        if candidate not in found:
            found.append(candidate)
    return found


def running_trainer_manifests():
    """Map manifest path -> live launcher root PIDs for every trainer process."""
    grouped = {}
    try:
        proc_entries = os.listdir("/proc")
    except OSError:
        return {}

    for name in proc_entries:
        if not name.isdigit():
            continue
        candidate = int(name)
        args = _cmdline_args(candidate)
        if (
            not any("trainer_launcher" in arg for arg in args)
            or not pid_is_alive(candidate)
        ):
            continue
        manifest = next(
            (arg for arg in args if arg.endswith(MANIFEST_FILE)),
            "",
        )
        if manifest:
            grouped.setdefault(os.path.abspath(manifest), set()).add(candidate)

    result = {}
    for manifest, pids in grouped.items():
        roots = [pid for pid in pids if _ppid(pid) not in pids]
        result[manifest] = sorted(roots or pids)
    return result
