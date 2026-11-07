"""Background dataset download worker.

The Streamlit process starts this module in a subprocess so the UI stays
responsive and a download can be cancelled at any time.

Usage::

    python -m flashstudio.utils.download_worker /path/to/config.json
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import time
import traceback
import zipfile
from urllib.request import Request, urlopen

from flashstudio.utils.dataset_sources import (
    archive_extension, as_url_list, get_dataset_urls,
)

READ_SIZE = 1024 * 1024
STATUS_WRITE_INTERVAL = 0.4


class DownloadCancelled(RuntimeError):
    pass


class StatusWriter:
    def __init__(self, path: str):
        self.path = path
        self._last = 0.0
        self._data = {
            "state": "starting",
            "message": "Preparing download...",
            "bytes": 0,
            "total": 0,
            "percent": None,
            "result": "",
            "error": "",
            "pid": os.getpid(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def write(self, force: bool = False, **changes) -> None:
        now = time.monotonic()
        self._data.update(changes)
        if not force and now - self._last < STATUS_WRITE_INTERVAL:
            return
        self._last = now
        self._data["updated"] = time.time()
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)
        except OSError:
            pass


def _remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _archive_ok(path: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        if path.endswith(".zip"):
            with zipfile.ZipFile(path, "r") as zf:
                zf.namelist()
            return True
        with tarfile.open(path, "r:*") as tf:
            tf.getmembers()
        return True
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return False


def _download(
    urls: list[str],
    dest: str,
    writer: StatusWriter,
    cancel_path: str,
    label: str,
) -> str:
    if os.path.isfile(dest):
        if _archive_ok(dest):
            writer.write(force=True, message=f"Using cached {label}")
            return dest
        _remove(dest)

    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    _remove(part)
    errors = []

    for url in urls:
        try:
            request = Request(
                url, headers={"User-Agent": "FlashStudio/0.2 dataset downloader"}
            )
            with urlopen(request, timeout=45) as response:
                total = int(response.headers.get("Content-Length") or 0)
                written = 0
                writer.write(
                    force=True,
                    state="downloading",
                    message=f"Downloading {label}...",
                    bytes=0,
                    total=total,
                    percent=0.0 if total else None,
                    url=url,
                )
                with open(part, "wb") as f:
                    while True:
                        if os.path.exists(cancel_path):
                            raise DownloadCancelled("Download cancelled")
                        chunk = response.read(READ_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
                        writer.write(
                            state="downloading",
                            message=f"Downloading {label}...",
                            bytes=written,
                            total=total,
                            percent=(written / total) if total else None,
                            url=url,
                        )
            if os.path.getsize(part) < 1:
                raise OSError("empty download")
            os.replace(part, dest)
            return dest
        except DownloadCancelled:
            _remove(part)
            raise
        except Exception as exc:
            errors.append(f"{url}\n{exc}")
            _remove(part)

    raise RuntimeError("Download failed for all mirrors:\n" + "\n".join(errors))


def _safe_target(base: str, name: str) -> str:
    base = os.path.realpath(base)
    target = os.path.realpath(os.path.join(base, name))
    if target != base and not target.startswith(base + os.sep):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return target


def _extract_external(archive_path: str, out_dir: str, writer: StatusWriter) -> None:
    os.makedirs(out_dir, exist_ok=True)
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.infolist()
            for member in members:
                _safe_target(out_dir, member.filename)
            total = max(len(members), 1)
            for i, member in enumerate(members, 1):
                zf.extract(member, out_dir)
                writer.write(
                    state="extracting",
                    message=f"Extracting {i}/{total} files...",
                    percent=i / total,
                    bytes=i,
                    total=total,
                )
        return

    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        for member in members:
            _safe_target(out_dir, member.name)
            if member.issym() or member.islnk():
                link = os.path.realpath(
                    os.path.join(os.path.dirname(os.path.join(out_dir, member.name)),
                                 member.linkname)
                )
                base = os.path.realpath(out_dir)
                if link != base and not link.startswith(base + os.sep):
                    raise ValueError(f"Unsafe archive link: {member.name!r}")
        total = max(len(members), 1)
        for i, member in enumerate(members, 1):
            try:
                tf.extract(member, out_dir, filter="data")
            except TypeError:  # Python < 3.12
                tf.extract(member, out_dir)
            writer.write(
                state="extracting",
                message=f"Extracting {i}/{total} files...",
                percent=i / total,
                bytes=i,
                total=total,
            )


def _run_flashdet(cfg: dict, writer: StatusWriter) -> str:
    from flashdet import download_dataset

    dataset_id = cfg["dataset_id"]
    output_dir = cfg["output_dir"]
    cache_dir = cfg.get("cache_dir") or os.path.join(
        os.path.expanduser("~"), ".cache", "flashdet"
    )
    cancel_path = cfg["cancel_path"]
    download_dir = os.path.join(cache_dir, dataset_id)
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    urls = get_dataset_urls(dataset_id)
    if not urls:
        raise RuntimeError(f"Dataset {dataset_id!r} is not in the flashdet registry.")

    file_count = len(urls)
    for index, (key, value) in enumerate(urls.items(), 1):
        candidates = as_url_list(value)
        if not candidates:
            continue
        ext = archive_extension(candidates[0]) or "tar"
        archive = os.path.join(download_dir, f"{key}.{ext}")
        writer.write(
            force=True,
            state="downloading",
            message=f"Downloading {key} ({index}/{file_count})...",
            bytes=0,
            total=0,
            percent=None,
            file_index=index,
            file_count=file_count,
        )
        _download(candidates, archive, writer, cancel_path, label=key)

    if os.path.exists(cancel_path):
        raise DownloadCancelled("Download cancelled")

    writer.write(
        force=True,
        state="extracting",
        message="Extracting and preparing dataset...",
        percent=None,
        bytes=0,
        total=0,
    )
    result = download_dataset(
        dataset_id=dataset_id, output_dir=output_dir, cache_dir=cache_dir
    )
    writer.write(
        force=True,
        state="done",
        message="Dataset ready",
        percent=1.0,
        result=result,
    )
    return result


def _run_external(cfg: dict, writer: StatusWriter) -> str:
    url = cfg["url"]
    name = cfg.get("name", "dataset")
    out_dir = cfg["output_dir"]
    cancel_path = cfg["cancel_path"]
    ext = archive_extension(url) or "zip"
    archive = out_dir.rstrip("/\\") + f".download.{ext}"
    try:
        _download([url], archive, writer, cancel_path, label=name)
        if os.path.exists(cancel_path):
            raise DownloadCancelled("Download cancelled")
        writer.write(
            force=True,
            state="extracting",
            message="Extracting archive...",
            percent=0.0,
            bytes=0,
            total=0,
        )
        _extract_external(archive, out_dir, writer)
    finally:
        _remove(archive)
    writer.write(
        force=True,
        state="done",
        message="Archive extracted",
        percent=1.0,
        result=out_dir,
    )
    return out_dir


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m flashstudio.utils.download_worker <config.json>")
        return 2
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = json.load(f)

    writer = StatusWriter(cfg["status_path"])
    writer.write(force=True, state="starting", message="Preparing download...")
    try:
        if cfg.get("mode") == "flashdet":
            _run_flashdet(cfg, writer)
        elif cfg.get("mode") == "external":
            _run_external(cfg, writer)
        else:
            raise ValueError(f"Unknown download mode: {cfg.get('mode')!r}")
        return 0
    except DownloadCancelled:
        writer.write(force=True, state="cancelled", message="Download cancelled")
        return 130
    except Exception as exc:
        writer.write(
            force=True,
            state="error",
            message=str(exc) or exc.__class__.__name__,
            error=traceback.format_exc()[-4000:],
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
