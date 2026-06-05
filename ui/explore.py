"""Rich read-only queries over a run's SQLite, for the detailed explorer:
every tournament match (Elo movement + transcript), full hypothesis detail
(text + reviews + match record), and a chronological timeline.

All functions take a db_path + run_id and open their own short-lived connection,
so the explorer can switch freely between question runs.
"""
from __future__ import annotations

import glob
import json
import os
import sqlite3


def list_eval_reports(path: str = "bench_runs") -> list[str]:
    """JSON eval reports (concordance / rediscovery), newest first."""
    return sorted(glob.glob(os.path.join(path, "*.json")), key=os.path.getmtime, reverse=True)


def load_eval_report(report_path: str) -> dict | None:
    try:
        with open(report_path) as fh:
            return json.load(fh)
    except Exception:
        return None


def list_runs(path: str) -> list[dict]:
    """Discover runs. `path` may be a single .db file or a directory of them.
    Returns [{label, db_path, run_id, goal}] sorted by db mtime (newest first)."""
    if os.path.isdir(path):
        dbs = sorted(glob.glob(os.path.join(path, "*.db")), key=os.path.getmtime, reverse=True)
    else:
        dbs = [path]
    runs: list[dict] = []
    for db in dbs:
        try:
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            for r in conn.execute("SELECT run_id, goal FROM configs ORDER BY created_at DESC"):
                tag = os.path.splitext(os.path.basename(db))[0]
                runs.append({
                    "label": f"{tag}  ·  {(r['goal'] or '')[:60]}",
                    "db_path": db,
                    "run_id": r["run_id"],
                    "goal": r["goal"] or "",
                })
            conn.close()
        except Exception:
            continue
    return runs


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def match_history(db_path: str, run_id: str) -> list[dict]:
    """Every tournament match in chronological order, with both hypotheses'
    summaries, the winner, Elo before→after for each, type, and transcript."""
    c = _conn(db_path)
    try:
        sums = {r["id"]: r["summary"] for r in
                c.execute("SELECT id, summary FROM hypotheses WHERE run_id=?", (run_id,))}
        rows = c.execute(
            "SELECT * FROM tournament_matches WHERE run_id=? ORDER BY created_at, id",
            (run_id,),
        ).fetchall()
    finally:
        c.close()
    out: list[dict] = []
    for i, m in enumerate(rows, 1):
        h1_won = m["winner_id"] == m["h1_id"]
        out.append({
            "n": i,
            "h1": sums.get(m["h1_id"], m["h1_id"][:8]),
            "h2": sums.get(m["h2_id"], m["h2_id"][:8]),
            "h1_won": h1_won,
            "type": m["match_type"],
            "e1_before": round(m["elo_before_h1"], 1), "e1_after": round(m["elo_after_h1"], 1),
            "e2_before": round(m["elo_before_h2"], 1), "e2_after": round(m["elo_after_h2"], 1),
            "transcript": m["debate_transcript"] or "",
            "created_at": str(m["created_at"]),
        })
    return out


def hypotheses_detailed(db_path: str, run_id: str) -> list[dict]:
    """Active hypotheses ranked by Elo, each with full text, reviews, and the
    matches it played (opponent + win/loss + Elo delta)."""
    c = _conn(db_path)
    try:
        hyps = c.execute(
            "SELECT * FROM hypotheses WHERE run_id=? AND status='active' "
            "ORDER BY elo_rating DESC", (run_id,)).fetchall()
        sums = {r["id"]: r["summary"] for r in
                c.execute("SELECT id, summary FROM hypotheses WHERE run_id=?", (run_id,))}
        matches = c.execute(
            "SELECT * FROM tournament_matches WHERE run_id=? ORDER BY created_at", (run_id,)).fetchall()
        reviews_by_h: dict[str, list] = {}
        for r in c.execute(
            "SELECT rv.* FROM reviews rv JOIN hypotheses h ON rv.hypothesis_id=h.id "
            "WHERE h.run_id=? ORDER BY rv.tier", (run_id,)):
            reviews_by_h.setdefault(r["hypothesis_id"], []).append(
                {"tier": r["tier"], "verdict": r["verdict"], "critique": r["critique"]})
    finally:
        c.close()
    out: list[dict] = []
    for h in hyps:
        played = []
        for m in matches:
            if h["id"] in (m["h1_id"], m["h2_id"]):
                is_h1 = m["id"] and h["id"] == m["h1_id"]
                opp = sums.get(m["h2_id"] if is_h1 else m["h1_id"], "?")
                won = m["winner_id"] == h["id"]
                before = m["elo_before_h1"] if is_h1 else m["elo_before_h2"]
                after = m["elo_after_h1"] if is_h1 else m["elo_after_h2"]
                played.append({"opponent": opp, "won": won,
                               "delta": round(after - before, 1)})
        out.append({
            "id": h["id"], "elo": round(h["elo_rating"], 1), "summary": h["summary"],
            "method": h["generation_method"], "source": h["source"],
            "text": h["text"], "reviews": reviews_by_h.get(h["id"], []),
            "matches": played,
        })
    return out


def timeline(db_path: str, run_id: str) -> list[dict]:
    """Merged chronological event log: generation, matches, reviews, meta-reviews."""
    c = _conn(db_path)
    events: list[dict] = []
    try:
        sums = {r["id"]: r["summary"] for r in
                c.execute("SELECT id, summary FROM hypotheses WHERE run_id=?", (run_id,))}
        for r in c.execute(
            "SELECT created_at, summary, generation_method FROM hypotheses WHERE run_id=?", (run_id,)):
            events.append({"t": str(r["created_at"]), "kind": "generate",
                           "text": f"{r['generation_method']} · {r['summary'][:70]}"})
        for r in c.execute(
            "SELECT created_at, match_type, winner_id, h1_id, h2_id "
            "FROM tournament_matches WHERE run_id=?", (run_id,)):
            w = sums.get(r["winner_id"], "?")[:50]
            events.append({"t": str(r["created_at"]), "kind": "match",
                           "text": f"{r['match_type']} → winner: {w}"})
        for r in c.execute(
            "SELECT rv.created_at, rv.tier FROM reviews rv JOIN hypotheses h "
            "ON rv.hypothesis_id=h.id WHERE h.run_id=?", (run_id,)):
            events.append({"t": str(r["created_at"]), "kind": "review",
                           "text": f"tier-{r['tier']} review"})
        for r in c.execute(
            "SELECT created_at, tick FROM meta_reviews WHERE run_id=?", (run_id,)):
            events.append({"t": str(r["created_at"]), "kind": "meta",
                           "text": f"meta-review (tick {r['tick']})"})
    finally:
        c.close()
    events.sort(key=lambda e: e["t"])
    return events
