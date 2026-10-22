"""Entry point: `python -m app [--host 127.0.0.1] [--port 7860] [--no-browser] [--device auto]`."""

from __future__ import annotations

import argparse
import threading
import webbrowser


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app", description="Visual saliency prediction GUI.")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default: 127.0.0.1, local only)")
    parser.add_argument("--port", type=int, default=7860, help="port (default: 7860)")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser on start")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                        help="inference device (default: auto = cuda > mps > cpu)")
    parser.add_argument("--weights-dir", default=None,
                        help="folder with the .safetensors files (default: weights/, or $SALIENCY_WEIGHTS_DIR)")
    args = parser.parse_args(argv)

    import uvicorn

    from app.server import create_app

    app = create_app(device=args.device, weights_dir=args.weights_dir)
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0", "::") else args.host
    url = f"http://{host}:{args.port}/"
    print(f"Saliency GUI on {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
