"""The offline provider is the only free, network-free way to run the full pipeline.

If it breaks, a fresh clone has no runnable path at all, so its contract is pinned here:
it must satisfy every parser in the repo and stay deterministic.
"""
from core.config_parser import _extract
from agents.reflection import _parse_verdict
from agents.ranking import _parse_winner
from agents.generation import _parse_hypothesis_output
from core.models import ResearchPlanConfig
from tools.llm import PROVIDERS, make_backend
from tools.offline import OfflineBackend


CFG = {"provider": "offline", "offline": {"seed": 0}}


def test_offline_is_a_registered_provider(monkeypatch):
    assert "offline" in PROVIDERS
    # COSCIENTIST_PROVIDER outranks cfg["provider"], and a developer .env may set it,
    # so pin the environment rather than depending on test ordering.
    monkeypatch.delenv("COSCIENTIST_PROVIDER", raising=False)
    assert isinstance(make_backend(CFG), OfflineBackend)
    # An explicit override must win regardless of environment.
    monkeypatch.setenv("COSCIENTIST_PROVIDER", "claude-code")
    assert isinstance(
        make_backend(CFG, provider_override="offline"), OfflineBackend
    )


def test_offline_never_claims_another_model():
    """A run must not be able to attribute offline output to a real model."""
    backend = OfflineBackend(model_strong="claude-opus-4-8", model_fast="gpt-5")
    assert backend.model_strong == "offline-deterministic-v1"
    assert backend.model_fast == "offline-deterministic-v1"


async def test_offline_output_satisfies_every_parser():
    backend = OfflineBackend()
    text = await backend.call("system", "Goal: Explain ALS mechanisms")

    # config_parser: safety + plan fields
    assert _extract("DECISION", text).strip() == "APPROVED"
    assert _extract("PREFERENCES", text)
    assert _extract("ATTRIBUTES", text)
    assert _extract("CONSTRAINTS", text)

    # reflection / ranking
    assert _parse_verdict(text) in {"passed", "flagged", "rejected"}
    assert _parse_winner(text, "h1", "h2") in {"h1", "h2"}

    # generation, plus the turn-loop termination signals
    config = ResearchPlanConfig(
        run_id="r", goal="g", preferences="p", attributes=["Novelty"],
        constraints="c", safety_approved=True,
    )
    hypothesis = _parse_hypothesis_output(text, config, "debate")
    assert hypothesis.summary
    assert "HYPOTHESIS" in text          # generation debate/assumptions loops
    assert "better idea:" in text        # ranking loop


async def test_offline_is_deterministic_and_varies_by_prompt():
    a1 = await OfflineBackend(seed=0).call("s", "Goal: alpha")
    a2 = await OfflineBackend(seed=0).call("s", "Goal: alpha")
    b = await OfflineBackend(seed=0).call("s", "Goal: beta")
    assert a1 == a2, "same seed + same prompt must replay byte-identically"
    assert a1 != b, "different goals must not collapse to one hypothesis"


async def test_offline_makes_no_network_call(monkeypatch):
    import socket

    def deny(*args, **kwargs):
        raise AssertionError("offline provider attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    text = await OfflineBackend().call("system", "Goal: Explain ALS mechanisms")
    assert text


async def test_timeout_still_surfaces_persisted_hypotheses(tmp_path, capsys, monkeypatch):
    """A wall-clock timeout must not discard work already paid for.

    Live run 143e54a9 (2026-08-09) completed all 12 tasks, then hit the deadline
    during shutdown and reported only a failure line — three real hypotheses were
    in the DB and never shown.
    """
    import coscientist
    from core.state import StateStore
    from core.models import Hypothesis, ResearchPlanConfig

    db = str(tmp_path / "t.db")
    store = StateStore(db)
    await store.init_db()
    config = ResearchPlanConfig(
        run_id="r1", goal="g", preferences="p", attributes=["Novelty"],
        constraints="c", safety_approved=True,
    )
    await store.save_hypothesis(Hypothesis(
        id="h1", run_id="r1", text="Hypothesis: a real mechanism", summary="kept",
        generation_method="debate", source="system", elo_rating=1300.0,
    ))

    await coscientist._print_results(store, config, db)
    out = capsys.readouterr().out
    assert "kept" in out
    assert "1300" in out.replace(".0", "")
