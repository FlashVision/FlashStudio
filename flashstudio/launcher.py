"""Launch FlashStudio UI — works locally and inside Google Colab."""

import subprocess
import sys
import os
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
        share: If True and running in Colab, creates a public ngrok tunnel.
        ngrok_token: Optional ngrok auth token. If not provided, uses NGROK_TOKEN env var.
    """
    _ensure_streamlit_config()
    app_path = _get_app_path()

    if is_colab():
        return _launch_colab(app_path, port, share, ngrok_token)
    else:
        _launch_local(app_path, port)
        return None


def _launch_colab(app_path: str, port: int, share: bool, ngrok_token: str | None):
    """Launch in Google Colab with automatic tunnel detection."""
    import time as _time
    import urllib.request
    import urllib.error

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

    # Wait until Streamlit is actually responding (up to 30s)
    ready = False
    for i in range(30):
        _time.sleep(1)
        if proc.poll() is not None:
            _, stderr = proc.communicate()
            err_msg = stderr.decode("utf-8", errors="replace")[-500:] if stderr else "Unknown error"
            print(f"\n\n  FlashStudio failed to start:\n  {err_msg}")
            return proc
        try:
            urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=2)
            ready = True
            break
        except Exception:
            print(".", end="", flush=True)

    if ready:
        print(" Ready!")
    else:
        print(" (still loading, URL may take a moment)")

    public_url = None

    # Method 1: Colab's built-in proxy (simplest, no install needed)
    try:
        from google.colab.output import eval_js
        result = eval_js(f"google.colab.kernel.proxyPort({port})")
        if result and str(result).startswith("http"):
            public_url = str(result)
    except Exception:
        pass

    # Method 2: ngrok (only if token is provided)
    if not public_url:
        token = ngrok_token or os.environ.get("NGROK_TOKEN", "")
        if token:
            try:
                from pyngrok import ngrok, conf
                conf.get_default().auth_token = token
                public_url = str(ngrok.connect(port, "http"))
            except ImportError:
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", "-q", "pyngrok"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    from pyngrok import ngrok, conf
                    conf.get_default().auth_token = token
                    public_url = str(ngrok.connect(port, "http"))
                except Exception:
                    pass
            except Exception:
                pass

    print(f"\n{'=' * 60}")
    print("  FlashStudio is running!")
    if public_url:
        print(f"  Open: {public_url}")
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
            display(HTML(
                f'<a href="{public_url}" target="_blank" '
                f'style="display:inline-block;font-size:16px;padding:10px 24px;'
                f'background:#7C3AED;color:white;border-radius:8px;'
                f'text-decoration:none;font-weight:600;margin:8px 0;">'
                f'Open FlashStudio</a>'
            ))
        except Exception:
            pass

    # Embed Streamlit directly in the Colab cell (works on mobile too)
    try:
        from google.colab import output
        output.serve_kernel_port_as_iframe(port, height=800)
    except Exception:
        if not public_url:
            try:
                from IPython.display import display, HTML
                display(HTML(
                    f'<div style="padding:12px;background:#FEF3C7;border-radius:8px;'
                    f'margin:8px 0;font-size:14px;">'
                    f'Run in a <b>new cell</b>:<br>'
                    f'<code style="background:#F3F4F6;padding:4px 8px;border-radius:4px;">'
                    f'from google.colab import output; '
                    f'output.serve_kernel_port_as_iframe({port}, height=800)</code>'
                    f'</div>'
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
