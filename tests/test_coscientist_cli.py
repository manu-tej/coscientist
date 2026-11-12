from unittest.mock import AsyncMock

import pytest

import coscientist
from coscientist import _print_results, build_parser
from core.models import Hypothesis


def test_core_cli_exposes_model_and_worker_scaling_controls():
    args = build_parser().parse_args([
        "find a repurposing mechanism",
        "--workers", "12",
        "--max-tasks", "80",
        "--provider", "openai-compatible",
        "--model-fast", "local:8b",
        "--model-strong", "local:70b",
    ])
    assert args.workers == 12
    assert args.max_tasks == 80
    assert args.provider == "openai-compatible"
    assert args.model_fast == "local:8b"
    assert args.model_strong == "local:70b"


@pytest.mark.parametrize(
    "argv",
    [
        ["goal", "--workers", "0"],
        ["goal", "--max-tasks", "0"],
        ["goal", "--max-time", "0"],
    ],
)
def test_core_cli_rejects_nonpositive_scale_limits(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


@pytest.mark.asyncio
async def test_core_main_closes_owned_backend_when_config_parsing_fails(
    monkeypatch, tmp_path,
):
    backend = object()
    close_backend = AsyncMock()

    class FakeStore:
        def __init__(self, path):
            self.path = path

        async def init_db(self):
            return None

    class BrokenParser:
        def __init__(self, **kwargs):
            pass

        async def parse_and_review(self, **kwargs):
            raise RuntimeError("invalid model response")

    monkeypatch.setattr(coscientist, "StateStore", FakeStore)
    monkeypatch.setattr(coscientist, "ConfigParser", BrokenParser)
    monkeypatch.setattr("tools.llm.make_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr("tools.llm.close_backend", close_backend)

    with pytest.raises(RuntimeError, match="invalid model response"):
        await coscientist.main("test goal", db_path=str(tmp_path / "run.db"))

    close_backend.assert_awaited_once_with(backend)


# --- CRITICAL 2: the CLI must not lie by omission about quarantined hypotheses ---


class _FakeConfig:
    run_id = "run1"


class _FakeResultsStore:
    """Minimal stand-in for StateStore's read side used by _print_results."""

    def __init__(self, active=(), quarantined=()):
        self._by_status = {"active": list(active), "quarantined": list(quarantined)}

    async def list_hypotheses(self, run_id, status="active"):
        return self._by_status.get(status, [])

    async def get_latest_meta_review(self, run_id):
        return None


def _hyp(hid, status="active", elo=1200.0):
    return Hypothesis(
        id=hid, run_id="run1", text=f"Hypothesis text for {hid}.",
        summary=f"summary {hid}", generation_method="literature",
        source="system", elo_rating=elo, status=status,
    )


@pytest.mark.asyncio
async def test_print_results_none_generated_says_so(capsys):
    store = _FakeResultsStore(active=[], quarantined=[])
    await _print_results(store, _FakeConfig(), "db.sqlite")
    out = capsys.readouterr().out
    assert "No hypotheses were produced." in out


@pytest.mark.asyncio
async def test_print_results_all_quarantined_does_not_claim_none_produced(capsys):
    """A run that generated 4 hypotheses and quarantined all 4 must not print
    the flatly false 'No hypotheses were produced.'"""
    store = _FakeResultsStore(
        active=[],
        quarantined=[_hyp("h1", "quarantined"), _hyp("h2", "quarantined"),
                     _hyp("h3", "quarantined")],
    )
    await _print_results(store, _FakeConfig(), "db.sqlite")
    out = capsys.readouterr().out
    assert "No hypotheses were produced." not in out
    assert "3" in out and "quarantined" in out


@pytest.mark.asyncio
async def test_print_results_mixed_surfaces_both_active_and_quarantined(capsys):
    store = _FakeResultsStore(
        active=[_hyp("h1", elo=1300.0)],
        quarantined=[_hyp("h2", "quarantined"), _hyp("h3", "quarantined")],
    )
    await _print_results(store, _FakeConfig(), "db.sqlite")
    out = capsys.readouterr().out
    assert "Top 1 hypotheses by Elo" in out
    assert "2 hypothesis(es) quarantined" in out
    assert "claim_verdicts" in out
