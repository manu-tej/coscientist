from __future__ import annotations

import json
import socket

import pytest

from company import offline_smoke


@pytest.mark.asyncio
async def test_offline_structural_smoke_is_reproducible_and_complete(tmp_path):
    kwargs = {
        "output_path": tmp_path / "report.json",
        "work_dir": tmp_path / "runs",
        "agent_workers": 2,
        "program_workers": 2,
        "max_tasks": 12,
    }
    first = await offline_smoke.run_structural_smoke(**kwargs)
    first_artifact = json.loads((tmp_path / "report.json").read_text())
    second = await offline_smoke.run_structural_smoke(**kwargs)
    second_artifact = json.loads((tmp_path / "report.json").read_text())

    assert first == first_artifact == second == second_artifact
    assert first["label"] == "STRUCTURAL_OFFLINE_HIERARCHY_SMOKE"
    assert first["status"] == "passed"
    assert first["scientific_quality_benchmark"] is False
    assert first["configuration"]["network_enabled"] is False
    assert first["execution"]["max_concurrent_stage_teams"] == 2
    assert first["execution"]["backend_calls_overlapped"] is True
    assert first["ledger"]["credits_spent"] == 10.0
    assert first["ledger"]["credits_spent"] == first["ledger"]["expected_credits_spent"]
    assert {program["program_id"]: program["selected_drug"]
            for program in first["programs"]} == {
        "offline-alpha": "tadalafil",
        "offline-beta": "sildenafil",
    }
    assert all(
        program["token_accounting"].startswith(
            "hard portable estimated/requested-token envelope"
        )
        for program in first["programs"]
    )
    assert all(
        program["hierarchy"]["all_roster_personas_assigned"]
        for program in first["programs"]
    )
    assert all(
        set(program["hierarchy"]["assigned_personas"])
        == set(program["hierarchy"]["team_roster"])
        for program in first["programs"]
    )
    assert all(first["checks"].values())
    assert first["schema_version"] == 3
    assert first["configuration"]["end_to_end_programs"] == 1
    end_to_end = first["end_to_end"]
    assert end_to_end["passed"] is True
    assert end_to_end["program_status"] == "completed"
    assert end_to_end["final_stage"] == "completed"
    assert end_to_end["state_reloaded"] is True
    assert end_to_end["stages_in_order"] == [
        "disease_dossier",
        "hypotheses",
        "mechanism",
        "translational",
    ]
    assert [stage["stage"] for stage in end_to_end["stage_runs"]] == (
        end_to_end["stages_in_order"]
    )
    assert [stage["gate_status"] for stage in end_to_end["stage_runs"]] == [
        "advanced",
        "advanced",
        "advanced",
        "completed",
    ]
    assert all(
        stage["all_roster_personas_assigned"]
        and stage["all_trace_tasks_succeeded"]
        and stage["trace_tasks"] == 12
        and stage["run_id_present"]
        and stage["database_existed_during_validation"]
        for stage in end_to_end["stage_runs"]
    )
    assert len(end_to_end["gate_decisions"]) == 4
    assert all(
        gate["ceo_decision"] == "advance"
        and gate["survived_roll"] is True
        and gate["run_id_present"]
        for gate in end_to_end["gate_decisions"]
    )
    assert end_to_end["ledger"]["expected_credits_spent"] == 40.0
    assert end_to_end["ledger"]["credits_spent"] == 40.0
    assert end_to_end["ledger"]["reconciled"] is True
    assert end_to_end["package"]["stage_order"] == (
        end_to_end["stages_in_order"]
    )
    assert end_to_end["package"]["stage_run_count"] == 4
    assert end_to_end["package"]["gate_decision_count"] == 4
    assert end_to_end["package"][
        "includes_mechanism_and_translational_evidence"
    ] is True
    assert {
        section["stage"] for section in end_to_end["package"]["stage_sections"]
        if section["experiment_count"] > 0
    } >= {"mechanism", "translational"}
    assert first["checks"]["four_stage_program_completed"] is True
    assert first["checks"]["four_stage_order_preserved"] is True
    assert first["checks"]["four_ceo_gates_persisted"] is True
    assert first["checks"]["completed_portfolio_reloaded"] is True
    assert first["checks"]["final_package_covers_all_stages"] is True
    assert first["checks"]["four_stage_ledger_reconciled"] is True
    assert first["atomicity_probe"]["scientific_state_rolled_back"] is True
    assert first["atomicity_probe"]["failed_attempt_usage_charged"] is True
    assert (
        first["atomicity_probe"]["failed_attempt_usage_survived_reload"]
        is True
    )
    assert first["atomicity_probe"]["state_reloaded"] is True
    assert first["atomicity_probe"]["committed_credits_spent"] == 0.0
    assert first["atomicity_probe"]["committed_tokens_spent"] == 0
    assert first["atomicity_probe"]["failed_attempt_credits_spent"] == 2.0
    assert first["atomicity_probe"]["failed_attempt_tokens_spent"] > 0
    assert (
        first["checks"]["failed_quarter_scientific_state_rolled_back"] is True
    )
    assert first["checks"]["failed_quarter_attempt_usage_charged"] is True
    assert (
        first["checks"]["failed_quarter_attempt_usage_survived_reload"]
        is True
    )
    assert "portfolio_unchanged" not in first["atomicity_probe"]
    assert "failure_was_atomic" not in first["checks"]


def test_offline_smoke_command_emits_json(monkeypatch, tmp_path, capsys):
    expected = {
        "schema_version": 3,
        "label": "STRUCTURAL_OFFLINE_HIERARCHY_SMOKE",
        "status": "passed",
        "scientific_quality_benchmark": False,
    }

    async def fake_run(**kwargs):
        offline_smoke._write_json(kwargs["output_path"], expected)
        return expected

    monkeypatch.setattr(offline_smoke, "run_structural_smoke", fake_run)
    output = tmp_path / "cli.json"
    exit_code = offline_smoke.main(["--output", str(output)])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert json.loads(output.read_text()) == expected


def test_offline_smoke_network_guard_fails_closed():
    with offline_smoke._zero_network_environment():
        with pytest.raises(
            offline_smoke.StructuralSmokeError,
            match="network access attempted",
        ):
            socket.create_connection(("example.com", 443))
