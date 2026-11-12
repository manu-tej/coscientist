"""bench/tournament_audit.py: why is a concordance tournament (un)informative?

Each test builds a real run database through StateStore so the audit's SQL is
checked against the schema the system actually writes.
"""
from __future__ import annotations

import uuid

import pytest

from bench.tournament_audit import (
    TournamentAudit,
    audit_run,
    refit_under_exclusions,
)
from core.models import Hypothesis, ResearchPlanConfig, TournamentMatch
from core.state import StateStore


def _hyp(run_id: str, text: str, *, elo: float = 1200.0, status: str = "active") -> Hypothesis:
    return Hypothesis(
        id=str(uuid.uuid4()), run_id=run_id, text=text, summary=text[:40],
        generation_method="literature", source="system", elo_rating=elo, status=status,
    )


async def _run_db(tmp_path, run_id: str, hyps: list[Hypothesis], matches=()) -> str:
    db_path = str(tmp_path / f"{run_id}.db")
    store = StateStore(db_path)
    await store.init_db()
    await store.save_config(ResearchPlanConfig(
        run_id=run_id, goal="g", preferences="p", attributes=["a"],
        constraints="c", safety_approved=True,
    ))
    for h in hyps:
        await store.save_hypothesis(h)
    for m in matches:
        await store.save_match_and_elos(m)
    return db_path


def _match(run_id: str, h1: Hypothesis, h2: Hypothesis, *, winner: Hypothesis, k: float = 16.0) -> TournamentMatch:
    d = k if winner is h1 else -k
    return TournamentMatch(
        id=str(uuid.uuid4()), run_id=run_id, h1_id=h1.id, h2_id=h2.id,
        winner_id=winner.id, match_type="single_turn",
        elo_before_h1=h1.elo_rating, elo_before_h2=h2.elo_rating,
        elo_after_h1=h1.elo_rating + d, elo_after_h2=h2.elo_rating - d,
    )


async def test_all_correct_tournament_is_a_ceiling_not_a_tournament_defect(tmp_path):
    run_id = "r1"
    hyps = [_hyp(run_id, f"Reasoning {i}. Answer: B") for i in range(3)]
    db = await _run_db(tmp_path, run_id, hyps, [_match(run_id, hyps[0], hyps[1], winner=hyps[0])])

    audit = audit_run(db, question_id="q", gold_answer="B", tag="q_0")

    assert isinstance(audit, TournamentAudit)
    assert audit.informative is False
    assert audit.n_active == 3 and audit.n_correct == 3
    assert audit.outcome == "all correct"
    assert audit.elo == "varies"
    assert audit.n_matches == 1


async def test_empty_text_hypotheses_are_counted_separately_from_wrong_answers(tmp_path):
    run_id = "r2"
    hyps = [_hyp(run_id, ""), _hyp(run_id, ""), _hyp(run_id, "Answer: D"), _hyp(run_id, "Answer: A")]
    db = await _run_db(tmp_path, run_id, hyps, [_match(run_id, hyps[0], hyps[1], winner=hyps[0])])

    audit = audit_run(db, question_id="q", gold_answer="D", tag="q_0")

    assert audit.n_active == 4
    assert audit.n_empty == 2
    assert audit.n_correct == 1
    assert audit.outcome == "varies"
    # the informativeness test, as the headline fit applies it, is fooled here:
    assert audit.informative is True
    # but the audit exposes that every Elo move came from an empty-vs-empty match
    assert audit.n_matches_with_empty_side == 1


async def test_elo_is_flat_when_the_only_match_was_between_rejected_hypotheses(tmp_path):
    run_id = "r3"
    active = [_hyp(run_id, "Answer: C"), _hyp(run_id, "Answer: A"), _hyp(run_id, "Answer: A")]
    rejected = [_hyp(run_id, "Answer: B", status="rejected"), _hyp(run_id, "Answer: B", status="rejected")]
    db = await _run_db(tmp_path, run_id, active + rejected,
                       [_match(run_id, rejected[0], rejected[1], winner=rejected[0])])

    audit = audit_run(db, question_id="q", gold_answer="C", tag="q_0")

    assert audit.outcome == "varies"
    assert audit.elo == "tied: 1 match, none among active hypotheses"
    assert audit.informative is False
    assert audit.n_matches == 1 and audit.n_matches_among_active == 0


async def test_audit_reads_the_latest_config_run_in_a_shared_db(tmp_path):
    db = await _run_db(tmp_path, "old", [_hyp("old", "Answer: A")])
    store = StateStore(db)
    await store.save_config(ResearchPlanConfig(
        run_id="new", goal="g", preferences="p", attributes=["a"], constraints="c", safety_approved=True,
    ))
    for h in [_hyp("new", "Answer: B"), _hyp("new", "Answer: B")]:
        await store.save_hypothesis(h)

    audit = audit_run(db, question_id="q", gold_answer="B", tag="q_0")

    assert audit.run_id == "new"
    assert audit.n_active == 2


async def test_refit_reports_what_survives_each_exclusion(tmp_path):
    # T1: clean tournament, 2 matches, mixed correctness, Elo spread.
    a = [_hyp("t1", "Answer: B", elo=1216), _hyp("t1", "Answer: B", elo=1184),
         _hyp("t1", "Answer: A", elo=1216), _hyp("t1", "Answer: A", elo=1184)]
    t1 = await _run_db(tmp_path, "t1", a, [_match("t1", a[0], a[1], winner=a[0]), _match("t1", a[2], a[3], winner=a[2])])
    # T2: informative only through empty rows scored wrong and one empty-vs-empty match.
    b = [_hyp("t2", ""), _hyp("t2", ""), _hyp("t2", "Answer: B")]
    t2 = await _run_db(tmp_path, "t2", b, [_match("t2", b[0], b[1], winner=b[0])])
    # T3: single match, mixed correctness.
    c = [_hyp("t3", "Answer: B"), _hyp("t3", "Answer: A"), _hyp("t3", "Answer: A")]
    t3 = await _run_db(tmp_path, "t3", c, [_match("t3", c[0], c[1], winner=c[1])])
    audits = [audit_run(p, question_id=q, gold_answer="B", tag=t)
              for p, q, t in ((t1, "q1", "q1_0"), (t2, "q2", "q2_0"), (t3, "q3", "q3_0"))]

    fits = refit_under_exclusions(audits)

    labels = [f.label for f in fits]
    assert labels == ["all rows", "drop empty-text rows", "drop empty-text rows and single-match tournaments"]
    assert [f.n_rows for f in fits] == [10, 8, 4]          # T2 keeps its one real row
    assert [f.n_strata for f in fits] == [3, 3, 1]        # T2 still has a row; T3 has one match
    assert [f.stats["logistic_n_questions"] for f in fits] == [3, 2, 1]


def test_refit_on_no_audits_is_empty_not_an_error():
    fits = refit_under_exclusions([])
    assert [f.n_rows for f in fits] == [0, 0, 0]
