"""Audit why a concordance tournament is, or is not, informative.

The headline verdict (``bench/concordance.py``) fits ``correct ~ Elo``
conditionally by tournament, and a tournament carries information only when
correctness varies AND Elo varies within it. That test is necessary but blind
to *why* a tournament passes or fails it. On the 2026-06-04 GPQA-bio runs it
passed strata whose only "wrong" rows were hypotheses persisted with empty
text, and strata whose only Elo spread came from a single match. This module
classifies each run database on both axes and refits the headline under the
exclusions those defects call for, all offline from the run databases.

Rows are the ``status='active'`` hypotheses of the latest run in each
database, exactly as ``bench.runner.read_run`` and ``concordance_from_runs``
select them, so the "all rows" refit reproduces the reported number.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from bench.concordance import ScoredHypothesis, _conditional_logistic_correct_on_elo
from bench.datasets.gpqa import score_answer

_TAG_RE = re.compile(r"concordance_(?P<qid>.+)_(?P<seed>\d+)$")


@dataclass
class AuditRow:
    scored: ScoredHypothesis
    empty: bool


@dataclass
class TournamentAudit:
    tag: str
    question_id: str
    run_id: str
    gold_answer: str
    n_active: int
    n_correct: int
    n_empty: int
    elo_distinct: int
    n_matches: int
    n_matches_among_active: int
    n_matches_with_empty_side: int
    outcome: str
    elo: str
    rows: list[AuditRow] = field(default_factory=list)

    @property
    def informative(self) -> bool:
        return self.outcome == "varies" and self.elo == "varies"


@dataclass
class Refit:
    label: str
    n_rows: int
    n_strata: int
    stats: dict


def audit_run(db_path: str, *, question_id: str, gold_answer: str, tag: str | None = None) -> TournamentAudit:
    """Classify one run database. Reads the latest run, active hypotheses only."""
    resolved = Path(db_path).resolve()
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cfg = conn.execute("SELECT run_id FROM configs ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        if cfg is None:
            raise ValueError(f"{db_path}: no persisted config")
        run_id = cfg["run_id"]
        hyps = conn.execute(
            "SELECT id, text, elo_rating FROM hypotheses WHERE run_id=? AND status='active'", (run_id,)
        ).fetchall()
        matches = conn.execute(
            "SELECT h1_id, h2_id FROM tournament_matches WHERE run_id=?", (run_id,)
        ).fetchall()
        empty_ids = {r["id"] for r in conn.execute(
            "SELECT id FROM hypotheses WHERE run_id=? AND length(text)=0", (run_id,))}
    finally:
        conn.close()

    rows = [
        AuditRow(
            scored=ScoredHypothesis(
                elo=float(h["elo_rating"]), correct=score_answer(h["text"], gold_answer),
                question_id=question_id, stratum=tag or run_id,
            ),
            empty=(h["id"] in empty_ids),
        )
        for h in hyps
    ]
    active_ids = {h["id"] for h in hyps}
    n = len(rows)
    n_correct = sum(r.scored.correct for r in rows)
    n_empty = sum(r.empty for r in rows)
    elo_distinct = len({r.scored.elo for r in rows})
    n_matches = len(matches)
    among_active = sum(1 for m in matches if m["h1_id"] in active_ids and m["h2_id"] in active_ids)
    with_empty = sum(1 for m in matches if m["h1_id"] in empty_ids or m["h2_id"] in empty_ids)

    if n == 0:
        outcome = "no active hypotheses"
    elif 0 < n_correct < n:
        outcome = "varies"
    elif n_correct == n:
        outcome = "all correct"
    elif n_empty == n:
        outcome = "all empty text"
    else:
        outcome = "all wrong"

    if elo_distinct > 1:
        elo = "varies"
    elif n_matches == 0:
        elo = "tied: no matches"
    else:
        elo = f"tied: {n_matches} match{'es' if n_matches != 1 else ''}, none among active hypotheses"

    return TournamentAudit(
        tag=tag or run_id, question_id=question_id, run_id=run_id, gold_answer=gold_answer,
        n_active=n, n_correct=n_correct, n_empty=n_empty, elo_distinct=elo_distinct,
        n_matches=n_matches, n_matches_among_active=among_active,
        n_matches_with_empty_side=with_empty, outcome=outcome, elo=elo, rows=rows,
    )


def refit_under_exclusions(audits: list[TournamentAudit]) -> list[Refit]:
    """Refit the headline conditional logistic on successively stricter row sets."""
    plans = [
        ("all rows", lambda a, r: True),
        ("drop empty-text rows", lambda a, r: not r.empty),
        ("drop empty-text rows and single-match tournaments", lambda a, r: not r.empty and a.n_matches > 1),
    ]
    out: list[Refit] = []
    for label, keep in plans:
        rows = [r.scored for a in audits for r in a.rows if keep(a, r)]
        strata = {r.stratum for r in rows}
        out.append(Refit(label=label, n_rows=len(rows), n_strata=len(strata),
                         stats=_conditional_logistic_correct_on_elo(rows)))
    return out


def tag_from_path(db_path: str) -> tuple[str, str] | None:
    """('gpqa-biology-10', 'concordance_gpqa-biology-10_2') from a run DB path."""
    stem = Path(db_path).stem
    m = _TAG_RE.match(stem)
    return (m.group("qid"), stem) if m else None


def format_table(audits: list[TournamentAudit]) -> str:
    head = (f"{'tournament':30} {'gold':4} {'act':>3} {'cor':>3} {'emp':>3} {'elo#':>4} "
            f"{'match':>5} {'m.act':>5} {'m.emp':>5}  outcome / elo")
    lines = [head]
    for a in audits:
        lines.append(
            f"{a.tag:30} {a.gold_answer:4} {a.n_active:3} {a.n_correct:3} {a.n_empty:3} {a.elo_distinct:4} "
            f"{a.n_matches:5} {a.n_matches_among_active:5} {a.n_matches_with_empty_side:5}  "
            f"{a.outcome} / {a.elo}{'   INFORMATIVE' if a.informative else ''}"
        )
    return "\n".join(lines)


def format_refits(refits: list[Refit]) -> str:
    lines = []
    for f in refits:
        s = f.stats
        lines.append(
            f"{f.label:52} rows={f.n_rows:3} strata={f.n_strata:2}  fitted: "
            f"{s['logistic_n_questions']} strata / {s['logistic_n_rows']:2} rows  "
            f"coef={s['logistic_coef']:+.2f}/SD  p={s['logistic_p']:.3f}"
            f"{'  SEPARATED' if s.get('logistic_separated') else ''}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--runs", default="bench_runs/concordance_gpqa-biology-*.db",
                   help="glob of concordance run databases (one tournament each)")
    args = p.parse_args(argv)
    paths = sorted(glob.glob(args.runs))
    if not paths:
        print(f"no run databases match {args.runs!r}")
        return 1
    # Gold letters come from the cached dataset; never reach the network for an audit.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from bench.datasets.gpqa import load_gpqa_hf
    gold = {g.id: g.gold_answer for g in load_gpqa_hf(all_subjects=True)}
    audits = []
    for path in paths:
        parsed = tag_from_path(path)
        if parsed is None or parsed[0] not in gold:
            print(f"skip {path}: not a concordance run with a known gold answer")
            continue
        qid, tag = parsed
        audits.append(audit_run(path, question_id=qid, gold_answer=gold[qid], tag=tag))
    print(format_table(audits))
    inf = [a for a in audits if a.informative]
    print(f"\ninformative by the headline's test: {len(inf)}/{len(audits)} tournaments, "
          f"{sum(a.n_active for a in inf)} of {sum(a.n_active for a in audits)} rows\n")
    print(format_refits(refit_under_exclusions(audits)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
