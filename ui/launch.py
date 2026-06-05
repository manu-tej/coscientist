"""Launch the run explorer over one DB or a whole directory of runs.

Usage:
    python -m ui.launch [path] [--port 7860]

`path` may be a single run SQLite or a directory of them (default: bench_runs/).
The explorer lets you pick any question/run and drill into its hypotheses,
every tournament match (Elo movement + transcripts), a timeline, and the eval
reports — auto-refreshing every 3s so live runs update in place.
"""
from __future__ import annotations

import argparse
import sys

from ui.app import build_app
from ui import explore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AI Co-Scientist run explorer")
    p.add_argument("path", nargs="?", default="bench_runs",
                   help="a run .db file or a directory of them (default: bench_runs/)")
    p.add_argument("--reports", default="bench_runs", help="dir holding *.json eval reports")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args(argv)

    runs = explore.list_runs(args.path)
    if not runs:
        print(f"No runs found under {args.path}.", file=sys.stderr)
        return 1
    print(f"Explorer: {len(runs)} run(s) from {args.path}")
    app = build_app(runs, reports_dir=args.reports)
    app.launch(server_name="127.0.0.1", server_port=args.port,
               share=args.share, inbrowser=True, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
