import json
import math

import pytest
from bench.report import build_report, render_markdown, to_json


def _sample_results():
    return {
        "concordance": {"spearman_rho": 0.78, "spearman_p": 0.004,
                        "kendall_tau": 0.61, "logistic_coef": 0.4, "logistic_p": 0.01,
                        "logistic_method": "conditional_by_tournament",
                        "logistic_n_rows": 430, "logistic_n_questions": 28,
                        "logistic_separated": False,
                        "n_buckets": 14, "n_rows": 450, "top1_accuracy": 0.66,
                        "blue_minus_red": {"mean_spread": 0.12, "ci_low": 0.05,
                                           "ci_high": 0.19, "n_buckets": 12}},
        "scaling": {"best_elo": {"spearman_rho": 0.95, "no_saturation": True}},
        "baseline": {"full_vs_best_of_32": {"p_value": 0.03, "median_delta": 45.0}},
        "ablation": {"no_evolution": {"p_value": 0.02, "median_delta": 38.0},
                     "no_tournament": {"p_value": 0.01, "median_delta": 60.0}},
        "cost": {"fresh_system_runs": 70, "total_calls": 7960},
        "meta": {"n_questions": 30, "dataset": "gpqa-bio", "system_version": "bdac572"},
    }


def test_build_report_is_json_serializable():
    rep = build_report(_sample_results())
    json.dumps(rep)        # must not raise
    assert rep["concordance"]["spearman_rho"] == 0.78
    assert rep["verdicts"]["concordance_pass"] is True   # rho>=0.7, p<0.05


def test_concordance_fail_verdict():
    res = _sample_results()
    res["concordance"]["logistic_p"] = 0.4
    rep = build_report(res)
    assert rep["verdicts"]["concordance_pass"] is False


def test_concordance_verdict_keys_on_logistic_not_spearman():
    # The bucketed Spearman is descriptive only: a weak rho must not veto a
    # significant within-question logistic, and a strong rho must not rescue one.
    res = _sample_results()
    res["concordance"]["spearman_rho"] = 0.3
    assert build_report(res)["verdicts"]["concordance_pass"] is True


@pytest.mark.parametrize("coef, p", [
    (-0.4, 0.01),                       # wrong sign
    (0.4, 0.3),                         # not significant
    (float("nan"), float("nan")),       # degenerate fit
    (None, None),                       # round-tripped through strict JSON
])
def test_concordance_verdict_fails_without_positive_significant_logistic(coef, p):
    res = _sample_results()
    res["concordance"]["logistic_coef"] = coef
    res["concordance"]["logistic_p"] = p
    assert build_report(res)["verdicts"]["concordance_pass"] is False


def test_render_markdown_includes_caveats_and_verdict():
    md = render_markdown(build_report(_sample_results()))
    assert "Concordance" in md
    assert "ρ" in md or "rho" in md.lower()
    assert "contamination" in md.lower()       # §13 honesty
    assert "0.78" in md


def test_render_markdown_headline_is_within_question_logistic():
    md = render_markdown(build_report(_sample_results()))
    tier1 = md.split("## Tier 1 — Concordance")[1].split("## Tier 1b")[0]
    lines = [l for l in tier1.splitlines() if l.startswith("- ")]
    assert "within question" in lines[0]
    assert "0.400" in lines[0] and "0.0100" in lines[0]
    assert "430 responses" in lines[0] and "28 informative tournaments" in lines[0]
    assert "descriptive" in lines[1].lower() and "0.780" in lines[1]
    assert "separation" not in md.lower()


def test_render_markdown_warns_on_within_question_separation():
    res = _sample_results()
    res["concordance"]["logistic_separated"] = True
    md = render_markdown(build_report(res))
    assert "separation" in md.lower()


def test_render_markdown_says_no_fit_when_logistic_is_nan():
    res = _sample_results()
    res["concordance"].update({"logistic_coef": float("nan"), "logistic_p": float("nan"),
                               "logistic_n_rows": 3, "logistic_n_questions": 1})
    md = render_markdown(build_report(res))
    assert "no fit" in md.lower()
    assert "fitted on" not in md.lower()
    assert "not established" in md


def test_render_markdown_flags_reference_failure():
    res = _sample_results()
    res["concordance"]["blue_minus_red"] = {"mean_spread": 0.0, "ci_low": -0.05,
                                            "ci_high": 0.05, "n_buckets": 10}
    md = render_markdown(build_report(res))
    assert "does not beat" in md.lower() or "fails" in md.lower()


def test_to_json_sanitizes_nan_and_inf():
    rep = {"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": float("-inf")},
           "e": 0.78, "f": "text", "g": True, "h": None}
    s = to_json(rep)
    # strict parse must succeed and nan/inf become null
    parsed = json.loads(s)
    assert parsed["a"] is None
    assert parsed["b"] == [1.0, None]
    assert parsed["c"]["d"] is None
    assert parsed["e"] == 0.78
    assert parsed["g"] is True
    # no NaN/Infinity tokens in the output
    assert "NaN" not in s and "Infinity" not in s
