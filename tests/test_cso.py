"""Tests for the multi-program CSO portfolio loop (spec §8) — all offline.

The CSO agent's LLM path is exercised with stub backends; the quantitative core
(allocation, marginal rNPV, explore→exploit) is deterministic by construction.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from company import engine
from company.cso import (
    CSOAgent,
    deterministic_plan,
    explore_exploit_beta,
    marginal_rnpv_per_credit,
    propose_allocation,
    propose_new_programs,
)
from company.models import ProgramStatus, Stage

PAH = "pulmonary arterial hypertension"


def _portfolio_with(*specs):
    """specs: (name, disease, estimated_value, confidence)"""
    pf = engine.new_company("Helix", PAH, credit_budget=500.0, seed=3)
    for name, disease, value, conf in specs:
        p = engine.add_program(pf, name, disease, estimated_value=value)
        p.confidence = conf
    return pf


# --- allocation math --------------------------------------------------------

def test_marginal_rnpv_rewards_value_and_confidence():
    pf = _portfolio_with(("RICH", PAH, 900.0, 0.9), ("POOR", PAH, 100.0, 0.9))
    rich, poor = pf.programs
    assert marginal_rnpv_per_credit(rich) > marginal_rnpv_per_credit(poor)
    # confidence feeds through PoS, so a shakier program earns fewer credits
    shaky = _portfolio_with(("SHAKY", PAH, 900.0, 0.2)).programs[0]
    assert marginal_rnpv_per_credit(rich) > marginal_rnpv_per_credit(shaky)


def test_completed_program_has_no_marginal_value():
    p = _portfolio_with(("DONE", PAH, 900.0, 0.9)).programs[0]
    p.stage = Stage.COMPLETED
    assert marginal_rnpv_per_credit(p) == 0.0


def test_explore_exploit_beta_rises_then_caps():
    assert explore_exploit_beta(0) == 1.0
    assert explore_exploit_beta(4) > explore_exploit_beta(0)
    assert explore_exploit_beta(99) == 4.0            # capped


def test_allocation_concentrates_as_cycles_advance():
    pf = _portfolio_with(("STRONG", PAH, 900.0, 0.9), ("WEAK", PAH, 200.0, 0.5))
    strong, weak = pf.programs
    early = propose_allocation(pf.programs, 100.0, cycle=0)
    late = propose_allocation(pf.programs, 100.0, cycle=6)
    # the leader's share grows as the schedule shifts explore → exploit
    assert late[strong.id] > early[strong.id]
    assert late[weak.id] < early[weak.id]
    assert round(sum(early.values())) == 100


def test_allocation_ignores_killed_and_completed():
    pf = _portfolio_with(("LIVE", PAH, 500.0, 0.8), ("DEAD", PAH, 500.0, 0.8))
    live, dead = pf.programs
    dead.status = ProgramStatus.KILLED
    alloc = propose_allocation(pf.programs, 50.0)
    assert live.id in alloc and dead.id not in alloc


def test_allocation_includes_held_programs_and_excludes_pending_gates():
    pf = _portfolio_with(
        ("HELD", PAH, 900.0, 0.9),
        ("PENDING", PAH, 800.0, 0.8),
        ("ACTIVE", PAH, 100.0, 0.6),
    )
    held, pending, active = pf.programs
    held.status = ProgramStatus.HELD

    allocation = propose_allocation(
        pf.programs,
        100.0,
        pending_ids={pending.id},
    )

    assert held.id in allocation
    assert active.id in allocation
    assert pending.id not in allocation
    assert sum(allocation.values()) == 100.0


def test_pending_program_does_not_displace_launch_eligible_gpu_leader():
    pf = _portfolio_with(
        ("PENDING-LEADER", PAH, 1000.0, 0.9),
        ("RUNNING", PAH, 100.0, 0.6),
    )
    pending, running = pf.programs
    pf.pending[pending.id] = engine.run_stage(pending, cycle=1)

    assert engine._is_leader(pf, running) is True


# --- new-program proposals --------------------------------------------------

def test_proposes_only_uncovered_diseases_by_value():
    pf = _portfolio_with(("PAH-1", PAH, 400.0, 0.7))
    props = propose_new_programs(pf.programs, k=3)
    diseases = [d for d, _ in props]
    assert PAH not in diseases                        # already covered
    assert diseases == sorted(diseases, key=lambda d: -dict(props)[d])   # value-ranked


# --- the plan ---------------------------------------------------------------

def test_plan_ranks_gates_above_funding():
    pf = _portfolio_with(
        ("PAH-1", PAH, 400.0, 0.0),
        ("PAH-2", PAH, 300.0, 0.7),
    )
    engine.run_program_stage(pf, pf.programs[0])       # creates a pending gate
    plan = deterministic_plan(pf.programs, pf.pending, 100.0, cycle=1)
    assert plan.actions[0].kind == "gate"             # gates are the urgent decision
    assert {a.kind for a in plan.actions} >= {"gate", "fund", "found"}
    assert plan.source == "deterministic"


def test_plan_is_reproducible():
    def build():
        pf = _portfolio_with(("A", PAH, 400.0, 0.7), ("B", "multiple myeloma", 900.0, 0.6))
        return [(a.kind, a.target, a.detail) for a in
                deterministic_plan(pf.programs, {}, 200.0, cycle=2).actions]
    assert build() == build()


# --- the agent --------------------------------------------------------------

class _StubBackend:
    def __init__(self, reply): self.reply = reply
    async def call(self, system, user, **kw): return self.reply


class _BoomBackend:
    async def call(self, system, user, **kw): raise RuntimeError("rate limited")


def _plan(backend, pf, cycle=1):
    return asyncio.run(CSOAgent(backend).propose_plan(pf.programs, pf.pending, 100.0, cycle=cycle))


def test_agent_without_backend_is_deterministic():
    pf = _portfolio_with(("A", PAH, 400.0, 0.7))
    assert _plan(None, pf).source == "deterministic"


def test_agent_reorders_by_llm_ranking():
    pf = _portfolio_with(("A", PAH, 400.0, 0.7), ("B", "multiple myeloma", 900.0, 0.6))
    base = deterministic_plan(pf.programs, pf.pending, 100.0, cycle=1)
    reversed_targets = list(reversed(base.targets()))
    plan = _plan(_StubBackend(str(reversed_targets).replace("'", '"')), pf)
    assert plan.source == "llm"
    assert plan.targets() == reversed_targets


def test_agent_falls_back_when_backend_errors():
    pf = _portfolio_with(("A", PAH, 400.0, 0.7))
    plan = _plan(_BoomBackend(), pf)
    assert plan.source == "deterministic" and "LLM error" in plan.summary


def test_agent_falls_back_on_unparseable_reply():
    pf = _portfolio_with(("A", PAH, 400.0, 0.7))
    plan = _plan(_StubBackend("I have no opinion whatsoever."), pf)
    assert plan.source == "deterministic" and "unparseable" in plan.summary


def test_cso_cli_closes_llm_backend(monkeypatch, capsys):
    from company import cli
    from tools import llm

    class ClosableBackend(_StubBackend):
        def __init__(self):
            super().__init__('["A"]')
            self.closed = False

        async def aclose(self):
            self.closed = True

    pf = _portfolio_with(("A", PAH, 400.0, 0.7))
    backend = ClosableBackend()
    loaded = []
    monkeypatch.setattr(cli, "_load", lambda args: pf)
    monkeypatch.setattr(llm, "load_backend_environment", lambda: loaded.append(True))
    monkeypatch.setattr(llm, "make_backend", lambda cfg: backend)

    cli.cmd_cso(SimpleNamespace(llm=True, state="unused.json"))

    assert loaded == [True]
    assert backend.closed is True
    assert "CSO portfolio plan" in capsys.readouterr().out


# --- multi-program loop resilience -----------------------------------------

def test_quarter_holds_unaffordable_programs_instead_of_crashing():
    pf = engine.new_company("Helix", PAH, credit_budget=5.0, seed=1)   # 3 × 2cr > 5
    for n in ("A", "B", "C"):
        engine.add_program(pf, n, PAH)
    ran = engine.run_quarter(pf, auto=True)
    assert len(ran) == 2                                   # two funded, one held
    held = [p for p in pf.programs if p.status is ProgramStatus.HELD]
    assert len(held) == 1
    assert "insufficient credits" in held[0].history[-1]
    assert pf.company.credit_spent <= pf.company.credit_budget
