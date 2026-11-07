"""FlashStudio CLI entrypoint."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="FlashStudio — Training & Inference UI for FlashDet")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on")
    parser.add_argument("--no-share", action="store_true", help="Disable ngrok sharing in Colab")
    parser.add_argument("--ngrok-token", type=str, default=None, help="ngrok auth token")
    args = parser.parse_args()

    from flashstudio.launcher import launch
    launch(port=args.port, share=not args.no_share, ngrok_token=args.ngrok_token)


if __name__ == "__main__":
    main()
