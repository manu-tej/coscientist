"""Tests for the gold-set validation harness — all offline via injected scorers."""
from __future__ import annotations

import math
import runpy
from pathlib import Path

import pytest

from company import validation
from company import cli as company_cli


def test_gold_pairs_full_crossproduct_and_labels():
    pairs = validation.gold_pairs()
    assert len(pairs) == len(validation.GOLD) * len(validation.all_drugs())
    n_gold = sum(1 for *_, g in pairs if g)
    assert n_gold == sum(len(v) for v in validation.GOLD.values())   # sildenafil counts in 2 diseases


def test_report_preserves_original_positional_constructor():
    report = validation.ValidationReport(
        54,
        54,
        10,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
        0.5,
    )

    assert report.auroc == 0.5
    assert report.n_gold_expected == 0
    assert report.complete is False


def test_perfect_scorer_separates_cleanly():
    def scorer(drug, disease):
        return 0.95 if drug in validation.GOLD[disease] else 0.10
    rep = validation.validate(scorer=scorer)
    assert rep.n_scored == rep.n_pairs                  # never abstains
    assert rep.auroc == 1.0
    assert rep.recall_at_1 == 1.0 and rep.recall_at_3 == 1.0
    assert rep.mean_gold > rep.mean_decoy
    assert rep.negative_control_max == 0.10
    assert rep.complete is True
    assert rep.coverage == 1.0
    assert rep.n_gold_expected == 10
    assert rep.n_gold_scored == 10
    assert rep.n_missing == 0


def test_inverted_scorer_is_anticoncordant():
    # a scorer that ranks decoys ABOVE truth → AUROC 0, recall 0 (the failure signature)
    def scorer(drug, disease):
        return 0.1 if drug in validation.GOLD[disease] else 0.9
    rep = validation.validate(scorer=scorer)
    assert rep.auroc == 0.0
    assert rep.recall_at_1 == 0.0


def test_partial_diagnostic_report_requires_explicit_opt_in():
    def scorer(drug, disease):
        if drug == "sildenafil":
            return None                                 # abstain on every sildenafil pair
        return 0.9 if drug in validation.GOLD[disease] else 0.2
    rep = validation.validate(scorer=scorer, require_complete=False)
    assert rep.n_pairs - rep.n_scored == len(validation.GOLD)   # one sildenafil pair per disease
    assert rep.complete is False
    assert rep.n_missing == len(validation.GOLD)
    assert rep.coverage == (rep.n_pairs - len(validation.GOLD)) / rep.n_pairs


def test_strict_validation_rejects_total_outage():
    with pytest.raises(validation.ValidationCoverageError) as failure:
        validation.validate(scorer=lambda *_: None)

    assert failure.value.expected == 54
    assert failure.value.scored == 0
    assert len(failure.value.missing_pairs) == 54


@pytest.mark.parametrize(
    "gold",
    [
        {},
        {"disease": []},
        {"disease": ["drug"]},
        {"disease": ["drug", "drug"]},
    ],
)
def test_malformed_gold_matrix_fails_before_scorer_calls(
    monkeypatch, gold,
):
    calls = 0

    def scorer(*_):
        nonlocal calls
        calls += 1
        return 0.5

    monkeypatch.setattr(validation, "GOLD", gold)

    with pytest.raises(validation.ValidationDataError):
        validation.validate(scorer=scorer)

    assert calls == 0


def test_strict_validation_rejects_one_missing_pair():
    missing = validation.gold_pairs()[0][:2]

    def scorer(drug, disease):
        if (drug, disease) == missing:
            return None
        return 0.5

    with pytest.raises(validation.ValidationCoverageError) as failure:
        validation.validate(scorer=scorer)

    assert failure.value.scored == 53
    assert failure.value.missing_pairs == [missing]


def test_zero_hit_scores_are_valid_observations_not_abstentions():
    report = validation.validate(scorer=lambda *_: 0.0)

    assert report.complete is True
    assert report.n_scored == report.n_pairs == 54
    assert report.n_missing == 0
    assert report.coverage == 1.0
    assert report.auroc == 0.5


def test_falsey_injected_scorer_is_used_without_live_fallback(
    monkeypatch,
):
    class FalseyScorer:
        def __init__(self):
            self.calls = 0

        def __bool__(self):
            return False

        def __call__(self, *_):
            self.calls += 1
            return 0.5

    scorer = FalseyScorer()

    def unexpected_live_scorer(*_):
        raise AssertionError("live scorer factory must not be selected")

    monkeypatch.setattr(
        validation,
        "_literature_scorer",
        unexpected_live_scorer,
    )

    report = validation.validate(scorer=scorer)

    assert scorer.calls == 54
    assert report.mode == "injected_scorer"


def test_falsey_injected_fetch_is_used_without_ncbi_fallback(
    monkeypatch,
):
    from company import literature

    class FalseyFetch:
        def __init__(self):
            self.calls = 0

        def __bool__(self):
            return False

        def __call__(self, _term):
            self.calls += 1
            return 0

    fetch = FalseyFetch()

    def unexpected_ncbi(*_args, **_kwargs):
        raise AssertionError("live NCBI fetch must not be selected")

    monkeypatch.setattr(literature, "_ncbi_count", unexpected_ncbi)

    report = validation.validate(fetch=fetch)

    assert fetch.calls == 54
    assert report.mode == "injected_fetch"


@pytest.mark.parametrize(
    "score",
    [True, float("nan"), float("inf"), -0.01, 1.01, "0.5"],
)
def test_invalid_scores_are_rejected(score):
    with pytest.raises(validation.ValidationDataError, match="finite real"):
        validation.validate(scorer=lambda *_: score)


def test_literature_path_with_stubbed_network():
    # fetch sees the esearch term; return a high count only for true gold pairs
    def fetch(term: str) -> int:
        for disease, drugs in validation.GOLD.items():
            if disease in term and any(f'"{d}"' in term for d in drugs):
                return 1500
        return 4
    rep = validation.validate(fetch=fetch)
    assert rep.auroc == 1.0
    assert rep.mean_gold > rep.mean_decoy
    assert not math.isnan(rep.recall_at_1)
    assert rep.mode == "injected_fetch"
    assert rep.source == "caller-provided PubMed count fetch"
    assert rep.mode != "live_pubmed"


def test_cli_incomplete_validation_exits_two_without_verdict(
    monkeypatch, capsys,
):
    def incomplete():
        raise validation.ValidationCoverageError(
            54,
            0,
            [(drug, disease) for drug, disease, _ in validation.gold_pairs()],
        )

    monkeypatch.setattr(validation, "validate", incomplete)

    assert company_cli.main(["validate"]) == 2
    captured = capsys.readouterr()
    assert "VALIDATION INCOMPLETE: scored 0/54" in captured.err
    assert "no scientific verdict emitted" in captured.err
    assert "REPORT COMPLETE" not in captured.out
    assert "tracks ground truth" not in captured.out


def test_cli_invalid_matrix_exits_two_without_verdict(
    monkeypatch, capsys,
):
    monkeypatch.setattr(validation, "GOLD", {})

    assert company_cli.main(["validate"]) == 2
    captured = capsys.readouterr()
    assert "VALIDATION INVALID" in captured.err
    assert "no scientific verdict emitted" in captured.err
    assert "REPORT COMPLETE" not in captured.out


def test_cli_complete_validation_reports_without_inventing_quality_gate(
    monkeypatch, capsys,
):
    report = validation.validate(
        scorer=lambda drug, disease: (
            0.9 if drug in validation.GOLD[disease] else 0.1
        )
    )
    monkeypatch.setattr(validation, "validate", lambda: report)

    assert company_cli.main(["validate"]) == 0
    captured = capsys.readouterr()
    assert "REPORT COMPLETE: scored 54/54 pairs (10/10 gold)" in captured.out
    assert "No pass/fail quality threshold is configured" in captured.out
    assert "tracks ground truth" not in captured.out
    assert captured.err == ""


def test_direct_company_script_propagates_main_exit_code(monkeypatch):
    monkeypatch.setattr(company_cli, "main", lambda: 2)
    script = Path(__file__).parents[1] / "company.py"

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(script), run_name="__main__")

    assert exit_info.value.code == 2
