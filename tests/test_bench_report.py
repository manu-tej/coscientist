import json
import math
from bench.report import build_report, render_markdown, to_json


def _sample_results():
    return {
        "concordance": {"spearman_rho": 0.78, "spearman_p": 0.004,
                        "kendall_tau": 0.61, "logistic_coef": 0.4, "logistic_p": 0.01,
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
    res["concordance"]["spearman_rho"] = 0.3
    rep = build_report(res)
    assert rep["verdicts"]["concordance_pass"] is False


def test_render_markdown_includes_caveats_and_verdict():
    md = render_markdown(build_report(_sample_results()))
    assert "Concordance" in md
    assert "ρ" in md or "rho" in md.lower()
    assert "contamination" in md.lower()       # §13 honesty
    assert "0.78" in md


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
