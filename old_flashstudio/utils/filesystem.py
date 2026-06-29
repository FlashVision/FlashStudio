"""Filesystem utilities — shared helpers for directory operations."""

import os
import shutil


def dir_size_bytes(path: str) -> int:
    """Calculate total size of a directory in bytes."""
    total = 0
    try:
        for dirpath, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(dirpath, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total


def dir_size_str(path: str) -> str:
    """Get human-readable directory size string."""
    total = dir_size_bytes(path)
    if total > 1_073_741_824:
        return f"{total / 1_073_741_824:.1f} GB"
    elif total > 1_048_576:
        return f"{total / 1_048_576:.0f} MB"
    elif total > 1024:
        return f"{total / 1024:.0f} KB"
    return f"{total} B"


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns the path."""
    os.makedirs(path, exist_ok=True)
    return path


def safe_rmtree(path: str) -> bool:
    """Safely remove a directory tree. Returns True if removed."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            return True
    except OSError:
        pass
    return False


def list_subdirs(path: str, sort_by_mtime: bool = True) -> list:
    """List immediate subdirectories, optionally sorted by modification time (newest first)."""
    if not os.path.isdir(path):
        return []
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if sort_by_mtime:
        dirs.sort(key=lambda d: os.path.getmtime(os.path.join(path, d)), reverse=True)
    return dirs


def count_files(path: str, extensions: tuple = None) -> int:
    """Count files in a directory, optionally filtered by extension."""
    if not os.path.isdir(path):
        return 0
    if extensions:
        return sum(1 for f in os.scandir(path) if f.is_file() and f.name.endswith(extensions))
    return sum(1 for f in os.scandir(path) if f.is_file())
