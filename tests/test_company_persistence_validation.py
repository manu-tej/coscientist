from __future__ import annotations

import json
from pathlib import Path

import pytest

from company import engine, store
from company import cli as company_cli
from company.cli import build_parser, cmd_status
from company.models import Company, Program, Stage


def _portfolio():
    portfolio = engine.new_company("Helix", "PAH")
    engine.add_program(portfolio, "PAH-1", "PAH")
    return portfolio


def _temporary_snapshots(path: Path) -> list[Path]:
    return list(path.parent.glob(f".{path.name}.*.tmp"))


def test_save_replaces_snapshot_from_same_directory_after_fsync(
    tmp_path, monkeypatch,
):
    path = tmp_path / "state.json"
    portfolio = _portfolio()
    events: list[tuple] = []
    real_fsync = store.os.fsync
    real_replace = store.os.replace

    def record_fsync(fd):
        events.append(("fsync",))
        return real_fsync(fd)

    def record_replace(source, destination):
        events.append(("replace", Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr(store.os, "fsync", record_fsync)
    monkeypatch.setattr(store.os, "replace", record_replace)

    store.save(portfolio, str(path))

    assert [event[0] for event in events] == ["fsync", "replace"]
    _, temporary, destination = events[1]
    assert temporary.parent == path.parent
    assert destination == path
    assert json.loads(path.read_text())["company"]["name"] == "Helix"
    assert not _temporary_snapshots(path)


def test_save_preserves_old_snapshot_and_cleans_temp_on_sync_failure(
    tmp_path, monkeypatch,
):
    path = tmp_path / "state.json"
    portfolio = _portfolio()
    store.save(portfolio, str(path))
    original = path.read_bytes()
    portfolio.company.name = "Changed"

    def fail_sync(_fd):
        raise OSError("simulated write durability failure")

    monkeypatch.setattr(store.os, "fsync", fail_sync)

    with pytest.raises(OSError, match="durability"):
        store.save(portfolio, str(path))

    assert path.read_bytes() == original
    assert not _temporary_snapshots(path)


def test_save_preserves_old_snapshot_and_cleans_temp_on_replace_failure(
    tmp_path, monkeypatch,
):
    path = tmp_path / "state.json"
    portfolio = _portfolio()
    store.save(portfolio, str(path))
    original = path.read_bytes()
    portfolio.company.name = "Changed"

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(store.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace"):
        store.save(portfolio, str(path))

    assert path.read_bytes() == original
    assert not _temporary_snapshots(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("token_budget", True),
        ("token_budget", -1),
        ("token_budget", 1.5),
        ("token_budget", float("nan")),
        ("token_budget", float("inf")),
        ("credit_budget", True),
        ("credit_budget", -1),
        ("credit_budget", float("nan")),
        ("credit_budget", float("inf")),
    ],
)
def test_new_company_rejects_invalid_budgets(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        engine.new_company("Helix", "PAH", **kwargs)


def test_new_company_accepts_zero_budgets():
    portfolio = engine.new_company(
        "Helix",
        "PAH",
        token_budget=0,
        credit_budget=0,
    )
    assert portfolio.company.token_budget == 0
    assert portfolio.company.credit_budget == 0.0


def test_company_positional_cycle_and_seed_contract_is_preserved():
    company = Company(
        "company",
        "Helix",
        "PAH",
        100,
        20.0,
        3,
        4.0,
        5,
        6,
    )

    assert company.token_spent == 3
    assert company.credit_spent == 4.0
    assert company.cycle == 5
    assert company.seed == 6
    assert company.failed_token_spent == 0
    assert company.failed_credit_spent == 0.0


@pytest.mark.parametrize(
    "value",
    [True, -1, float("nan"), float("inf")],
)
def test_add_program_rejects_invalid_estimated_value_without_mutation(value):
    portfolio = engine.new_company("Helix", "PAH")

    with pytest.raises(ValueError, match="estimated_value"):
        engine.add_program(portfolio, "PAH-1", "PAH", estimated_value=value)

    assert portfolio.programs == []


def test_add_program_rejects_duplicate_name_without_mutation():
    portfolio = engine.new_company("Helix", "PAH")
    original = engine.add_program(portfolio, "PAH-1", "PAH")

    with pytest.raises(ValueError, match="duplicate program name"):
        engine.add_program(portfolio, "PAH-1", "other disease")

    assert portfolio.programs == [original]


def test_portfolio_rejects_duplicate_names_loaded_from_snapshot(tmp_path):
    path = tmp_path / "state.json"
    portfolio = _portfolio()
    store.save(portfolio, str(path))
    payload = json.loads(path.read_text())
    duplicate = dict(payload["programs"][0])
    duplicate["id"] = "different-id"
    payload["programs"].append(duplicate)
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="duplicate program name"):
        store.load(str(path))


@pytest.mark.parametrize(
    "argv",
    [
        ["new", "--name", "Helix", "--disease", "PAH", "--budget", "-1"],
        ["new", "--name", "Helix", "--disease", "PAH", "--budget", "nan"],
        ["new", "--name", "Helix", "--disease", "PAH", "--budget", "inf"],
        ["new", "--name", "Helix", "--disease", "PAH", "--token-budget", "-1"],
        ["new", "--name", "Helix", "--disease", "PAH", "--token-budget", "nan"],
        ["program-new", "--name", "PAH-1", "--value", "-1"],
        ["program-new", "--name", "PAH-1", "--value", "nan"],
        ["program-new", "--name", "PAH-1", "--value", "inf"],
    ],
)
def test_cli_rejects_invalid_budgets_and_estimated_values(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_cli_accepts_zero_budgets_and_estimated_value():
    company_args = build_parser().parse_args(
        [
            "new",
            "--name",
            "Helix",
            "--disease",
            "PAH",
            "--budget",
            "0",
            "--token-budget",
            "0",
        ]
    )
    program_args = build_parser().parse_args(
        ["program-new", "--name", "PAH-1", "--value", "0"]
    )

    assert company_args.budget == 0.0
    assert company_args.token_budget == 0
    assert program_args.value == 0.0


def test_portfolio_boundary_rejects_invalid_direct_models():
    company = Company(
        id="company",
        name="Helix",
        disease_focus="PAH",
        token_budget=True,
        credit_budget=1,
    )
    with pytest.raises(ValueError, match="token_budget"):
        engine.Portfolio(company=company)

    company.token_budget = 1
    program = Program(
        id="program",
        company_id=company.id,
        name="PAH-1",
        disease="PAH",
        estimated_value=float("nan"),
    )
    with pytest.raises(ValueError, match="estimated_value"):
        engine.Portfolio(company=company, programs=[program])


def test_status_reports_committed_and_failed_attempt_usage(
    tmp_path, capsys,
):
    path = tmp_path / "state.json"
    portfolio = _portfolio()
    portfolio.company.credit_spent = 2.0
    portfolio.company.failed_credit_spent = 3.0
    portfolio.company.token_spent = 100
    portfolio.company.failed_token_spent = 50
    store.save(portfolio, str(path))
    args = build_parser().parse_args(["--state", str(path), "status"])

    cmd_status(args)

    output = capsys.readouterr().out
    assert "Credits: 5/500 used (2 committed, 3 failed-attempt, 495 left)" in output
    assert (
        "Portable tokens: 150/5,000,000 used "
        "(100 committed, 50 failed-attempt, 4,999,850 left)"
    ) in output


def test_failed_async_quarter_persists_only_attempt_usage(
    tmp_path, monkeypatch,
):
    path = tmp_path / "state.json"
    portfolio = _portfolio()
    store.save(portfolio, str(path))

    async def failed_quarter(attempted, **_kwargs):
        attempted.company.failed_token_spent += 10
        attempted.company.failed_credit_spent += 2.0
        # Even if a faulty executor leaked scientific mutation into the caller,
        # the failure persistence path must copy only operational usage.
        attempted.company.cycle += 1
        attempted.programs[0].stage = Stage.TRANSLATIONAL
        attempted.programs[0].lead_candidate = "not-durable"
        raise RuntimeError("provider failed after spending")

    monkeypatch.setattr(engine, "run_quarter_async", failed_quarter)

    with pytest.raises(RuntimeError, match="provider failed after spending"):
        company_cli.main([
            "--state",
            str(path),
            "quarter",
            "--cso-allocation",
        ])

    durable = store.load(str(path))
    assert durable.company.failed_token_spent == 10
    assert durable.company.failed_credit_spent == 2.0
    assert durable.company.cycle == 0
    assert durable.programs[0].stage is Stage.DISEASE_DOSSIER
    assert durable.programs[0].lead_candidate is None


def test_failed_async_quarter_without_usage_leaves_snapshot_unchanged(
    tmp_path, monkeypatch,
):
    path = tmp_path / "state.json"
    portfolio = _portfolio()
    store.save(portfolio, str(path))
    original = path.read_bytes()

    async def failed_before_attempt(*_args, **_kwargs):
        raise RuntimeError("failed before provider call")

    monkeypatch.setattr(engine, "run_quarter_async", failed_before_attempt)

    with pytest.raises(RuntimeError, match="failed before provider call"):
        company_cli.main([
            "--state",
            str(path),
            "quarter",
            "--cso-allocation",
        ])

    assert path.read_bytes() == original


def test_successful_hierarchical_quarter_is_saved_before_backend_cleanup(
    tmp_path, monkeypatch,
):
    from tools import llm

    path = tmp_path / "state.json"
    portfolio = _portfolio()
    store.save(portfolio, str(path))
    backend = object()

    async def successful_quarter(attempted, **_kwargs):
        attempted.company.cycle += 1
        attempted.company.token_spent += 20
        attempted.company.credit_spent += 3.0
        attempted.programs[0].stage = Stage.HYPOTHESES
        return []

    async def failed_cleanup(actual):
        assert actual is backend
        durable = store.load(str(path))
        assert durable.company.cycle == 1
        assert durable.company.token_spent == 20
        assert durable.company.credit_spent == 3.0
        assert durable.programs[0].stage is Stage.HYPOTHESES
        raise RuntimeError("backend cleanup failed")

    monkeypatch.setattr(engine, "run_quarter_async", successful_quarter)
    monkeypatch.setattr(llm, "make_backend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(llm, "close_backend", failed_cleanup)

    with pytest.raises(RuntimeError, match="backend cleanup failed"):
        company_cli.main([
            "--state",
            str(path),
            "quarter",
            "--hierarchical",
        ])

    durable = store.load(str(path))
    assert durable.company.cycle == 1
    assert durable.company.token_spent == 20
    assert durable.company.credit_spent == 3.0
    assert durable.programs[0].stage is Stage.HYPOTHESES


def test_failed_hierarchical_quarter_is_debited_before_failing_cleanup(
    tmp_path, monkeypatch,
):
    from tools import llm

    path = tmp_path / "state.json"
    portfolio = _portfolio()
    store.save(portfolio, str(path))
    backend = object()

    async def failed_quarter(attempted, **_kwargs):
        attempted.company.failed_token_spent += 10
        attempted.company.failed_credit_spent += 2.0
        attempted.company.cycle += 1
        attempted.programs[0].stage = Stage.TRANSLATIONAL
        raise RuntimeError("stage failed after spending")

    async def failed_cleanup(actual):
        assert actual is backend
        durable = store.load(str(path))
        # This observation happens before cleanup raises, proving a hanging
        # cleanup cannot delay the operational debit.
        assert durable.company.failed_token_spent == 10
        assert durable.company.failed_credit_spent == 2.0
        assert durable.company.cycle == 0
        assert durable.programs[0].stage is Stage.DISEASE_DOSSIER
        raise RuntimeError("backend cleanup failed")

    monkeypatch.setattr(engine, "run_quarter_async", failed_quarter)
    monkeypatch.setattr(llm, "make_backend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(llm, "close_backend", failed_cleanup)

    with pytest.raises(RuntimeError, match="backend cleanup failed"):
        company_cli.main([
            "--state",
            str(path),
            "quarter",
            "--hierarchical",
        ])

    durable = store.load(str(path))
    assert durable.company.failed_token_spent == 10
    assert durable.company.failed_credit_spent == 2.0
    assert durable.company.cycle == 0
    assert durable.programs[0].stage is Stage.DISEASE_DOSSIER


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("company", "token_spent", -1),
        ("company", "credit_spent", float("nan")),
        ("company", "failed_token_spent", True),
        ("company", "failed_credit_spent", -1),
        ("program", "tokens_spent", -1),
        ("program", "credits_spent", float("inf")),
    ],
)
def test_portfolio_rejects_invalid_persisted_usage(
    owner, field, value,
):
    portfolio = _portfolio()
    target = (
        portfolio.company
        if owner == "company"
        else portfolio.programs[0]
    )
    setattr(target, field, value)

    with pytest.raises(ValueError, match=field):
        engine.Portfolio(
            company=portfolio.company,
            programs=portfolio.programs,
        )
