"""Build and optionally serve the offline engineering lifetime dashboard."""
from __future__ import annotations

import argparse
import functools
import http.server
import json
from pathlib import Path

from build_engineering_demo import DEFAULT_EXPERIMENT_OUTPUT, DEFAULT_OUTPUT, build_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-output", type=Path, default=DEFAULT_EXPERIMENT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--no-serve", action="store_true", help="Build only; do not start a web server.")
    args = parser.parse_args()

    report = build_dashboard(args.experiment_output, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.no_serve:
        return

    directory = args.output.resolve()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"Engineering dashboard: http://{args.host}:{args.port}/", flush=True)
        print("Press Ctrl+C to stop the local server.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Engineering dashboard server stopped.", flush=True)


if __name__ == "__main__":
    main()
