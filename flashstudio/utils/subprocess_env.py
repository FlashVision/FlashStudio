"""Environment setup for FlashStudio child processes.

FlashDet logs status symbols such as ``✓`` (U+2713).  When a child process
writes to a redirected log file on Windows, Python guesses the ANSI code page
(``cp1252``) for stdout and the write raises ``UnicodeEncodeError``, killing
training during dataset verification.  Forcing UTF-8 I/O for every child
process avoids that class of failure.
"""

from __future__ import annotations

import os


def utf8_subprocess_env(package_root: str = "") -> dict:
    """Return an ``os.environ`` copy configured for UTF-8 child output."""
    env = os.environ.copy()
    if package_root:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            package_root + (os.pathsep + existing if existing else "")
        )
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env
