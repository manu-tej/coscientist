"""Launch the live Hypothesis Explorer against any captured/active run DB.

Usage:
    python -m ui.launch [db_path] [--run-id RUN_ID] [--port 7860]

Defaults to coscientist.db; point it at a bench run, e.g.
    python -m ui.launch bench_runs/rediscover_da-1-3_4.db
The explorer auto-refreshes every 3s, so it tracks a run live as it executes.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

from core.state import StateStore
from ui.app import build_app


def _pick_run(db_path: str, run_id: str | None) -> tuple[str | None, str | None]:
    """Return (run_id, goal). If run_id is None, pick the most recent run in the DB."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if run_id:
            row = conn.execute(
                "SELECT run_id, goal FROM configs WHERE run_id = ?", (run_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT run_id, goal FROM configs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
    finally:
        conn.close()
    return (row["run_id"], row["goal"]) if row else (None, None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live Hypothesis Explorer for a co-scientist run")
    p.add_argument("db_path", nargs="?", default="coscientist.db",
                   help="path to a run SQLite (e.g. bench_runs/rediscover_da-1-3_4.db)")
    p.add_argument("--run-id", default=None, help="specific run_id (default: most recent)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a public gradio share link")
    args = p.parse_args(argv)

    run_id, goal = _pick_run(args.db_path, args.run_id)
    if not run_id:
        print(f"No run found in {args.db_path} (no configs row).", file=sys.stderr)
        return 1
    print(f"Explorer → run {run_id} in {args.db_path}")
    print(f"goal: {(goal or '')[:120]}")

    store = StateStore(args.db_path)
    app = build_app(store, run_id, supervisor_handle={})
    app.launch(server_name="127.0.0.1", server_port=args.port,
               share=args.share, inbrowser=True, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
