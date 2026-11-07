"""Launch FlashStudio UI — works locally and inside Google Colab."""

import atexit
import platform
import subprocess
import sys
import os
import re
import tempfile
import time as _time
from pathlib import Path

from flashstudio.utils.device import is_colab

_STREAMLIT_CONFIG = """\
[theme]
primaryColor = "#7C3AED"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FB"
textColor = "#1A1A2E"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[client]
showSidebarNavigation = false

[browser]
gatherUsageStats = false
"""


def _get_app_path() -> str:
    return str(Path(__file__).parent / "app.py")


def _ensure_streamlit_config():
    """Write the default Streamlit config if one doesn't already exist."""
    config_dir = Path.home() / ".streamlit"
    config_file = config_dir / "config.toml"
    if config_file.exists():
        return
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(_STREAMLIT_CONFIG)


def launch(port: int = 8501, share: bool = True, ngrok_token: str | None = None):
    """Launch FlashStudio Streamlit app.

    Args:
        port: Port to run the Streamlit server on.
        share: If True and running in Colab, creates a public tunnel
               (works on mobile too).
        ngrok_token: Optional ngrok auth token. If not provided, uses
                     NGROK_TOKEN env var.
    """
    _ensure_streamlit_config()
    app_path = _get_app_path()

    if is_colab():
        return _launch_colab(app_path, port, share, ngrok_token)
    else:
        _launch_local(app_path, port)
        return None


# ---------------------------------------------------------------------------
# Tunnel helpers
# ---------------------------------------------------------------------------

# Keep tunnel processes referenced for the lifetime of the session. If these
# are garbage collected or terminated, the trycloudflare.com hostname starts
# returning Cloudflare error 1033 ("tunnel cannot resolve").
_LIVE_TUNNELS: list = []


def _shutdown_tunnels():
    for proc in list(_LIVE_TUNNELS):
        try:
            if proc.poll() is None:
                proc.terminate()
        except OSError:
            pass


atexit.register(_shutdown_tunnels)


def _cloudflared_binary() -> str | None:
    """Return a usable cloudflared binary path, downloading it if needed."""
    import shutil

    existing = shutil.which("cloudflared")
    if existing:
        return existing

    binary = "/usr/local/bin/cloudflared"
    if os.path.isfile(binary):
        return binary

    machine = platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
    url = ("https://github.com/cloudflare/cloudflared/releases/latest/download/"
           f"cloudflared-linux-{arch}")

    try:
        subprocess.check_call(
            ["wget", "-q", url, "-O", binary],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        try:
            import urllib.request
            urllib.request.urlretrieve(url, binary)
        except Exception:
            return None

    try:
        os.chmod(binary, 0o755)
    except OSError:
        return None
    return binary


def _start_cloudflared_tunnel(port: int) -> str | None:
    """Start a free Cloudflare quick tunnel (no sign-up required).

    The tunnel process is kept alive for the rest of the session so the URL
    stays usable. Returns the public https URL or None on failure.
    """
    binary = _cloudflared_binary()
    if not binary:
        return None

    fd, log_name = tempfile.mkstemp(prefix="flashstudio-cloudflared-",
                                    suffix=".log")
    os.close(fd)
    log_path = Path(log_name)
    try:
        with open(log_path, "wb") as log_file:
            proc = subprocess.Popen(
                [binary, "tunnel", "--no-autoupdate", "--protocol", "http2",
                 "--url", f"http://127.0.0.1:{port}"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
    except OSError:
        log_path.unlink(missing_ok=True)
        return None

    _LIVE_TUNNELS.append(proc)

    url_re = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = _time.monotonic() + 30
    while _time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            output = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            output = ""
        match = url_re.search(output)
        if match:
            return match.group(0)
        _time.sleep(0.5)

    # Timed out or cloudflared exited — clean up so we don't leak a dead tunnel.
    if proc.poll() is None:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    if proc in _LIVE_TUNNELS:
        _LIVE_TUNNELS.remove(proc)
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        pass
    return None


def _start_ngrok_tunnel(port: int, ngrok_token: str) -> str | None:
    """Start an ngrok tunnel. Requires an auth token."""
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "pyngrok"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            from pyngrok import ngrok, conf
        except Exception:
            return None
    try:
        conf.get_default().auth_token = ngrok_token
        return str(ngrok.connect(port, "http"))
    except Exception:
        return None


def _get_colab_proxy_url(port: int) -> str | None:
    """Get Colab's built-in proxy URL (only works in the same browser session)."""
    try:
        from google.colab.output import eval_js
        result = eval_js(f"google.colab.kernel.proxyPort({port})")
        if result and str(result).startswith("http"):
            return str(result)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Colab launcher
# ---------------------------------------------------------------------------

def _launch_colab(app_path: str, port: int, share: bool, ngrok_token: str | None):
    """Launch in Google Colab with a public URL that works on any device."""
    import urllib.request

    print("  Starting FlashStudio...", end="", flush=True)

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.port", str(port),
         "--server.headless", "true",
         "--server.enableCORS", "false",
         "--server.enableXsrfProtection", "false"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    ready = False
    for _ in range(30):
        _time.sleep(1)
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            err_msg = (stderr.decode("utf-8", errors="replace")[-500:]
                       if stderr else "Unknown error")
            print(f"\n\n  FlashStudio failed to start:\n  {err_msg}")
            return proc
        try:
            urllib.request.urlopen(
                f"http://localhost:{port}/_stcore/health", timeout=2)
            ready = True
            break
        except Exception:
            print(".", end="", flush=True)

    if ready:
        print(" Ready!")
    else:
        print(" (still loading, URL may take a moment)")

    # --- Obtain a public URL ---
    # Priority: cloudflared (free, works everywhere including mobile)
    #         > ngrok (needs token) > Colab proxy (desktop-only)
    public_url = None
    tunnel_type = None

    if share:
        # 1) Free Cloudflare tunnel — works on mobile, no sign-up
        print("  Setting up public URL (works on mobile)...", end="", flush=True)
        public_url = _start_cloudflared_tunnel(port)
        if public_url:
            tunnel_type = "cloudflared"
            print(" Done!")

        # 2) ngrok — if a token is available
        if not public_url:
            token = ngrok_token or os.environ.get("NGROK_TOKEN", "")
            if token:
                public_url = _start_ngrok_tunnel(port, token)
                if public_url:
                    tunnel_type = "ngrok"
                    print(" Done!")

        if not public_url:
            print(" Skipped (using Colab proxy)")

    # 3) Colab's built-in proxy (fallback — desktop browser only)
    if not public_url:
        public_url = _get_colab_proxy_url(port)

    # --- Display results ---
    print(f"\n{'=' * 60}")
    print("  FlashStudio is running!")
    if public_url:
        print(f"  Open: {public_url}")
        if tunnel_type:
            print("  (works on mobile & any device)")
        else:
            print("  ⚠ This URL only works in this browser.")
            print("    On mobile? Re-run with: launch(share=True)")
    else:
        print(f"  Port: {port}")
        print("")
        print("  Run this in a new cell to get the URL:")
        print("    from google.colab.output import eval_js")
        print(f"    print(eval_js('google.colab.kernel.proxyPort({port})'))")
    print(f"{'=' * 60}\n")

    # Show clickable link
    if public_url:
        try:
            from IPython.display import display, HTML
            mobile_badge = ""
            if tunnel_type:
                mobile_badge = (
                    '<span style="display:inline-block;font-size:11px;'
                    'padding:2px 8px;background:#10B981;color:white;'
                    'border-radius:4px;margin-left:8px;vertical-align:middle;">'
                    'Mobile Ready</span>'
                )
            display(HTML(
                f'<div style="margin:8px 0;">'
                f'<a href="{public_url}" target="_blank" '
                f'style="display:inline-block;font-size:16px;padding:10px 24px;'
                f'background:#7C3AED;color:white;border-radius:8px;'
                f'text-decoration:none;font-weight:600;">'
                f'Open FlashStudio</a>{mobile_badge}</div>'
            ))
        except Exception:
            pass

    # Embed Streamlit directly in the Colab cell output
    try:
        from google.colab import output
        output.serve_kernel_port_as_iframe(port, height=800)
    except Exception:
        if not public_url:
            try:
                from IPython.display import display, HTML
                display(HTML(
                    f'<div style="padding:12px;background:#FEF3C7;'
                    f'border-radius:8px;margin:8px 0;font-size:14px;">'
                    f'Run in a <b>new cell</b>:<br>'
                    f'<code style="background:#F3F4F6;padding:4px 8px;'
                    f'border-radius:4px;">'
                    f'from google.colab import output; '
                    f'output.serve_kernel_port_as_iframe({port}, height=800)'
                    f'</code></div>'
                ))
            except Exception:
                pass

    return proc


def _launch_local(app_path: str, port: int):
    """Launch locally with streamlit."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.port", str(port)],
    )
