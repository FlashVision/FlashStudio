"""Launch FlashStudio UI — works locally and inside Google Colab."""

import subprocess
import sys
import os
import re
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

def _start_cloudflared_tunnel(port: int) -> str | None:
    """Start a free Cloudflare tunnel (no sign-up required).

    Returns the public https URL or None on failure.
    """
    binary = "/usr/local/bin/cloudflared"
    if not os.path.isfile(binary):
        try:
            subprocess.check_call(
                ["wget", "-q",
                 "https://github.com/cloudflare/cloudflared/releases/latest"
                 "/download/cloudflared-linux-amd64",
                 "-O", binary],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.chmod(binary, 0o755)
        except Exception:
            return None

    tunnel_proc = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://localhost:{port}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    url_re = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    deadline = _time.monotonic() + 20
    while _time.monotonic() < deadline:
        line = tunnel_proc.stderr.readline().decode("utf-8", errors="replace")
        if not line:
            if tunnel_proc.poll() is not None:
                return None
            _time.sleep(0.2)
            continue
        match = url_re.search(line)
        if match:
            return match.group(0)

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
    colab_proxy_url = _get_colab_proxy_url(port)
    if not public_url:
        public_url = colab_proxy_url

    # --- Display results ---
    print(f"\n{'=' * 60}")
    print("  FlashStudio is running!")
    if public_url:
        print(f"  Open: {public_url}")
        if tunnel_type:
            print(f"  (works on mobile & any device)")
        else:
            print(f"  ⚠ This URL only works in this browser.")
            print(f"    On mobile? Re-run with: launch(share=True)")
    else:
        print(f"  Port: {port}")
        print("")
        print("  Run this in a new cell to get the URL:")
        print(f"    from google.colab.output import eval_js")
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
