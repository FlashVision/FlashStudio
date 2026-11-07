"""Tests for the cancellable background download worker."""

import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def _serve(directory):
    handler = partial(_QuietHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _wait_status(path, predicate, timeout=15):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        try:
            with open(path, "r", encoding="utf-8") as f:
                last = json.load(f)
            if predicate(last):
                return last
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    return last


def _run_worker(config, timeout=30):
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [sys.executable, "-m", "flashstudio.utils.download_worker", config],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )


def test_external_worker_downloads_and_extracts(tmp_path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    with zipfile.ZipFile(serve_dir / "data.zip", "w") as zf:
        zf.writestr("dataset/train/images/a.jpg", b"jpeg")

    server, port = _serve(serve_dir)
    try:
        out = tmp_path / "out"
        status = tmp_path / "status.json"
        config = tmp_path / "config.json"
        config.write_text(json.dumps({
            "mode": "external",
            "url": f"http://127.0.0.1:{port}/data.zip",
            "name": "test",
            "output_dir": str(out),
            "status_path": str(status),
            "cancel_path": str(tmp_path / "cancel"),
        }))
        result = _run_worker(str(config))
        assert result.returncode == 0, result.stderr
        data = json.loads(status.read_text())
        assert data["state"] == "done"
        assert (out / "dataset" / "train" / "images" / "a.jpg").is_file()
    finally:
        server.shutdown()


def test_worker_cancels_slow_download(tmp_path):
    class SlowHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(50 * 1024 * 1024))
            self.end_headers()
            try:
                for _ in range(50):
                    self.wfile.write(b"x" * (1024 * 1024))
                    self.wfile.flush()
                    time.sleep(0.1)
            except Exception:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        status = tmp_path / "status.json"
        cancel = tmp_path / "cancel"
        config = tmp_path / "config.json"
        config.write_text(json.dumps({
            "mode": "external",
            "url": f"http://127.0.0.1:{port}/slow.zip",
            "name": "slow",
            "output_dir": str(tmp_path / "out"),
            "status_path": str(status),
            "cancel_path": str(cancel),
        }))
        env = os.environ.copy()
        env["PYTHONPATH"] = ROOT
        proc = subprocess.Popen(
            [sys.executable, "-m", "flashstudio.utils.download_worker", str(config)],
            cwd=ROOT, env=env,
        )
        data = _wait_status(
            str(status),
            lambda item: item.get("state") == "downloading"
            and (item.get("bytes") or 0) > 0,
        )
        assert data.get("state") == "downloading"
        cancel.write_text("cancel")
        assert proc.wait(timeout=10) == 130
        data = json.loads(status.read_text())
        assert data["state"] == "cancelled"
    finally:
        server.shutdown()


def test_start_and_cancel_download_job(tmp_path, mock_session_state, monkeypatch):
    import streamlit as st

    from flashstudio.pages.data import helpers

    server, port = _serve(tmp_path)
    try:
        monkeypatch.setattr(helpers, "_get_data_root", lambda: str(tmp_path))
        state = helpers._start_dataset_download(
            "external",
            str(tmp_path / "out"),
            url=f"http://127.0.0.1:{port}/missing.zip",
            name="test",
        )
        assert state is not None
        assert helpers._download_is_active()
        helpers._cancel_dataset_download()
        assert helpers._download_state() is None
        assert not helpers._download_is_active()
    finally:
        server.shutdown()
