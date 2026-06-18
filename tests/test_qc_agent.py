"""Tests for the QC agent's selection parsing and graceful degradation.

No live LLM: a stub backend returns canned text, and we assert the agent parses,
clips to budget, validates names, and falls back cleanly on error / no backend.
"""
from __future__ import annotations

import pytest

from qc.agent import DEFAULT_BATTERY, QCScientist, _parse_checks

VALID = ["missingness", "univariate_outlier", "covariate_shift", "batch_effect", "leakage"]


class StubBackend:
    def __init__(self, reply: str):
        self.reply = reply

    async def call(self, system, user, **kw):
        return self.reply


class FailBackend:
    async def call(self, system, user, **kw):
        raise RuntimeError("simulated 429 rate limit")


# --- parsing ----------------------------------------------------------------

def test_parse_json_array():
    out = _parse_checks('["batch_effect", "covariate_shift"] because sites differ', VALID, 2)
    assert out == ["batch_effect", "covariate_shift"]


def test_parse_filters_invalid_and_clips_budget():
    out = _parse_checks('["leakage", "not_a_check", "missingness", "batch_effect"]', VALID, 2)
    assert out == ["leakage", "missingness"]


def test_parse_prose_fallback_uses_mention_order():
    out = _parse_checks("I'd run leakage first, then maybe batch_effect.", VALID, 2)
    assert out == ["leakage", "batch_effect"]


def test_parse_empty_when_nothing_valid():
    assert _parse_checks("run something sensible", VALID, 2) == []


# --- agent behaviour --------------------------------------------------------

async def test_fallback_without_backend():
    sel = await QCScientist(backend=None, budget=2).select_checks("anything")
    assert sel.source == "fallback"
    assert sel.checks == DEFAULT_BATTERY[:2]


async def test_llm_selection_parsed():
    sci = QCScientist(backend=StubBackend('["batch_effect","covariate_shift"]'), budget=2)
    sel = await sci.select_checks("two sites, recalibrated instruments")
    assert sel.source == "llm"
    assert sel.checks == ["batch_effect", "covariate_shift"]


async def test_backend_error_degrades_to_fallback():
    sel = await QCScientist(backend=FailBackend(), budget=2).select_checks("desc")
    assert sel.source == "fallback"
    assert sel.checks == DEFAULT_BATTERY[:2]


async def test_unparseable_reply_degrades():
    sel = await QCScientist(backend=StubBackend("sure, I'll handle it"), budget=2).select_checks("desc")
    assert sel.source == "fallback"


async def test_budget_is_respected():
    sci = QCScientist(backend=StubBackend('["missingness","leakage","batch_effect"]'), budget=1)
    sel = await sci.select_checks("desc")
    assert len(sel.checks) == 1


async def test_rank_pads_to_full_permutation():
    sci = QCScientist(backend=StubBackend('["leakage","batch_effect"]'))
    sel = await sci.rank_checks("desc", available=VALID)
    assert sel.checks[:2] == ["leakage", "batch_effect"]
    assert sorted(sel.checks) == sorted(VALID)          # all checks present, no dups
    assert len(sel.checks) == len(VALID)


async def test_rank_fallback_is_full_permutation():
    sel = await QCScientist(backend=None).rank_checks("desc", available=VALID)
    assert sorted(sel.checks) == sorted(VALID)
    assert sel.source == "fallback"


async def test_report_without_backend_lists_flags():
    from qc.detectors import Flag
    f = Flag("leakage", "col9", 0.99, 0.001, 0.001, 0.49, "high", "AUC=0.99")
    report = await QCScientist(backend=None).write_report("desc", [f])
    assert "leakage" in report and "col9" in report
