"""Launch FlashStudio UI — works locally and inside Google Colab."""

import subprocess
import sys
import os
from pathlib import Path

from flashstudio.utils.device import is_colab


def _get_app_path() -> str:
    return str(Path(__file__).parent / "app.py")


def launch(port: int = 8501, share: bool = True, ngrok_token: str | None = None):
    """Launch FlashStudio Streamlit app.

    Args:
        port: Port to run the Streamlit server on.
        share: If True and running in Colab, creates a public ngrok tunnel.
        ngrok_token: Optional ngrok auth token. If not provided, uses NGROK_TOKEN env var.
    """
    app_path = _get_app_path()

    if is_colab():
        _launch_colab(app_path, port, share, ngrok_token)
    else:
        _launch_local(app_path, port)


def _launch_colab(app_path: str, port: int, share: bool, ngrok_token: str | None):
    """Launch in Google Colab with ngrok tunnel."""
    try:
        from pyngrok import ngrok, conf
    except ImportError:
        print("Installing pyngrok...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyngrok"])
        from pyngrok import ngrok, conf

    token = ngrok_token or os.environ.get("NGROK_TOKEN", "")
    if token:
        conf.get_default().auth_token = token

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.port", str(port),
         "--server.headless", "true",
         "--server.enableCORS", "false",
         "--server.enableXsrfProtection", "false"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if share:
        public_url = ngrok.connect(port, "http")
        print(f"\n{'=' * 60}")
        print("  FlashStudio is running!")
        print(f"  Local:  http://localhost:{port}")
        print(f"  Public: {public_url}")
        print(f"{'=' * 60}\n")
    else:
        print(f"\n  FlashStudio running at http://localhost:{port}\n")

    return proc


def _launch_local(app_path: str, port: int):
    """Launch locally with streamlit."""
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path,
         "--server.port", str(port)],
    )
