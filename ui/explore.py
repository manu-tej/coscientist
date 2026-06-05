"""Rich read-only queries over a run's SQLite, for the detailed explorer:
every tournament match (Elo movement + transcript), full hypothesis detail
(text + reviews + match record), and a chronological timeline.

All functions take a db_path + run_id and open their own short-lived connection,
so the explorer can switch freely between question runs.
"""
from __future__ import annotations

import functools
import glob
import json
import os
import re
import sqlite3


@functools.lru_cache(maxsize=1)
def _gpqa_gold() -> dict:
    """{goal_id: gold_letter} for GPQA-bio (cached HF load; {} if unavailable)."""
    try:
        from bench.datasets.gpqa import load_gpqa_hf
        return {g.id: g.gold_answer for g in load_gpqa_hf(all_subjects=True)}
    except Exception:
        return {}


@functools.lru_cache(maxsize=1)
def _rediscover_gold() -> dict:
    """{task_id: BenchGoal} for the curated BiomniBench goldset."""
    try:
        from bench.goalset import load_goalset
        return {g.id: g for g in load_goalset("bench/datasets/biomnibench_goldset.jsonl")}
    except Exception:
        return {}


def resolve_gold(db_path: str) -> dict:
    """Map a run DB (by its filename's task id) to its ground truth.
    Returns {kind, task, gold_answer?|gold_entities?+gold_finding?}."""
    base = os.path.splitext(os.path.basename(db_path))[0]
    m = re.match(r"[a-zA-Z0-9]+_(.+?)_\d+$", base)
    task = m.group(1) if m else base
    if task.startswith("gpqa"):
        return {"kind": "gpqa", "task": task, "gold_answer": _gpqa_gold().get(task)}
    if task.startswith("da-"):
        g = _rediscover_gold().get(task)
        if g:
            return {"kind": "rediscover", "task": task,
                    "gold_entities": g.gold_entities,
                    "gold_finding": g.metadata.get("gold_finding", "")}
    return {"kind": None, "task": task}


def active_hypotheses(db_path: str, run_id: str) -> list[dict]:
    """All active hypotheses (text + elo) ranked by Elo, top first."""
    c = _conn(db_path)
    try:
        rows = c.execute(
            "SELECT text, summary, elo_rating FROM hypotheses "
            "WHERE run_id=? AND status='active' ORDER BY elo_rating DESC", (run_id,)).fetchall()
    finally:
        c.close()
    return [{"text": r["text"], "summary": r["summary"], "elo": round(r["elo_rating"], 1)} for r in rows]


def list_eval_reports(path: str = "bench_runs") -> list[str]:
    """JSON eval reports (concordance / rediscovery), newest first."""
    return sorted(glob.glob(os.path.join(path, "*.json")), key=os.path.getmtime, reverse=True)


def load_eval_report(report_path: str) -> dict | None:
    try:
        with open(report_path) as fh:
            return json.load(fh)
    except Exception:
        return None


# Boilerplate that prefixes every GPQA goal — strip it so the dropdown shows the
# actual question, not an identical "Research goal: Determine…" for every run.
_GOAL_BOILER = re.compile(
    r"^\s*research goal:.*?question:\s*", re.IGNORECASE | re.DOTALL)


def goal_excerpt(goal: str, n: int = 64) -> str:
    """A short, distinguishing slice of a run's goal for the run picker."""
    t = _GOAL_BOILER.sub("", goal or "").strip()
    t = re.sub(r"\s+", " ", t)
    return (t[: n - 1] + "…") if len(t) > n else t


# Benchmark family of a run, keyed off its DB filename prefix. Each family is a
# distinct eval with its own ground-truth shape, so the picker groups by these.
_FAMILIES = [
    ("GPQA-bio (concordance)", re.compile(r"^concordance_gpqa", re.IGNORECASE)),
    ("BiomniBench (rediscovery)", re.compile(r"^rediscover_da", re.IGNORECASE)),
]


def run_family(db_path: str) -> str:
    """Group label for a run, from its filename (e.g. 'GPQA-bio (concordance)')."""
    base = os.path.basename(db_path)
    for name, pat in _FAMILIES:
        if pat.match(base):
            return name
    return "Other"


def list_runs(path: str) -> list[dict]:
    """Discover runs. `path` may be a single .db file or a directory of them.
    Returns [{label, db_path, run_id, goal, n_hyps}] with populated runs first
    (then newest-first); each label carries the run tag, hypothesis count, and a
    distinguishing goal excerpt so every run is identifiable at a glance."""
    if os.path.isdir(path):
        dbs = sorted(glob.glob(os.path.join(path, "*.db")), key=os.path.getmtime, reverse=True)
    else:
        dbs = [path]
    runs: list[dict] = []
    for rank, db in enumerate(dbs):  # rank preserves the mtime order as a tiebreak
        try:
            conn = sqlite3.connect(db)
            conn.row_factory = sqlite3.Row
            for r in conn.execute("SELECT run_id, goal FROM configs ORDER BY created_at DESC"):
                n = conn.execute(
                    "SELECT COUNT(*) FROM hypotheses WHERE run_id=? AND status='active'",
                    (r["run_id"],)).fetchone()[0]
                tag = os.path.splitext(os.path.basename(db))[0]
                runs.append({
                    "label": f"{tag}  ·  {n}h  ·  {goal_excerpt(r['goal'] or '')}",
                    "db_path": db,
                    "run_id": r["run_id"],
                    "goal": r["goal"] or "",
                    "n_hyps": n,
                    "family": run_family(db),
                    "_rank": rank,
                })
            conn.close()
        except Exception:
            continue
    # Populated runs first (a fresh visitor never lands on an empty/timed-out run),
    # then preserve newest-first within each group.
    runs.sort(key=lambda r: (r["n_hyps"] == 0, r["_rank"]))
    for r in runs:
        r.pop("_rank", None)
    return runs


def runs_by_family(runs: list[dict]) -> dict[str, list[dict]]:
    """Group runs into {family: [runs]}, families ordered as declared
    (known families first, 'Other' last); run order within a family preserved."""
    order = [name for name, _ in _FAMILIES] + ["Other"]
    grouped: dict[str, list[dict]] = {}
    for r in runs:
        grouped.setdefault(r["family"], []).append(r)
    return {fam: grouped[fam] for fam in order if fam in grouped}


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    return c


def match_history(db_path: str, run_id: str) -> list[dict]:
    """Every tournament match in chronological order, with both hypotheses'
    summaries, the winner, Elo before→after for each, type, and transcript."""
    c = _conn(db_path)
    try:
        texts = {r["id"]: r["text"] for r in
                 c.execute("SELECT id, text FROM hypotheses WHERE run_id=?", (run_id,))}
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
            "h1": texts.get(m["h1_id"], m["h1_id"][:8]),
            "h2": texts.get(m["h2_id"], m["h2_id"][:8]),
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
