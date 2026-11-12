from __future__ import annotations

import json
import math
from typing import Any


def _num(c: dict, key: str) -> float:
    """A statistic as a float; None (strict-JSON round trip) or missing → nan."""
    v = c.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")


def _concordance_pass(c: dict) -> bool:
    """Elo tracks correctness when the within-question logistic slope is positive
    and significant. A degenerate fit (nan/None) fails. The bucketed Spearman is
    descriptive only: it correlates ~5 bucket means, not the responses."""
    coef, p = _num(c, "logistic_coef"), _num(c, "logistic_p")
    if math.isnan(coef) or math.isnan(p):
        return False
    return coef > 0 and p < 0.05


def _reference_beats(c: dict) -> bool:
    bmr = c.get("blue_minus_red", {})
    return bmr.get("ci_low", -1.0) > 0.0


def build_report(results: dict) -> dict:
    """Assemble the machine-readable report with derived pass/fail verdicts."""
    c = results.get("concordance", {})
    verdicts = {
        "concordance_pass": _concordance_pass(c),
        "reference_beats_difficulty": _reference_beats(c),
        "scaling_no_saturation": results.get("scaling", {})
            .get("best_elo", {}).get("no_saturation", False),
    }
    return {**results, "verdicts": verdicts}


def _sanitize(obj: Any) -> Any:
    """Recursively replace nan/inf floats with None so the result is strict-JSON
    serializable (json.dumps would otherwise emit non-standard NaN/Infinity tokens)."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def to_json(report: dict, indent: int = 2) -> str:
    """Serialize a report to strict JSON (nan/inf → null), safe for any parser."""
    return json.dumps(_sanitize(report), indent=indent)


def render_markdown(report: dict) -> str:
    c = report.get("concordance", {})
    v = report.get("verdicts", {})
    meta = report.get("meta", {})
    bmr = c.get("blue_minus_red", {})

    lines = [
        "# AI Co-Scientist — Evaluation Report",
        "",
        f"- Dataset: `{meta.get('dataset', '?')}`  ·  questions: {meta.get('n_questions', '?')}"
        f"  ·  system: `{meta.get('system_version', '?')}`",
        "",
        "## Tier 1 — Concordance (Elo ↔ ground truth)",
        "",
        f"- **Conditional logistic (correct ~ Elo, within question):** "
        + (f"coef = **{_num(c, 'logistic_coef'):.3f}** per SD of within-tournament Elo "
           f"(LRT p = {_num(c, 'logistic_p'):.4f}) · fitted on "
           f"{c.get('logistic_n_rows', '?')} responses across "
           f"{c.get('logistic_n_questions', '?')} informative tournaments"
           if not math.isnan(_num(c, "logistic_coef")) else
           f"no fit — only {c.get('logistic_n_rows', '?')} responses across "
           f"{c.get('logistic_n_questions', '?')} informative tournaments "
           f"(a tournament is informative only if its hypotheses differ in both "
           f"correctness and Elo)"),
        *(["- ⚠️ Within-question separation (complete or quasi-complete): the "
           "slope magnitude is not meaningful; read only its sign and the p-value."]
          if c.get("logistic_separated") else []),
        f"- Descriptive (bucketed, pooled across questions): "
        f"Spearman ρ = {_num(c, 'spearman_rho'):.3f} (p = {_num(c, 'spearman_p'):.4f}) · "
        f"Kendall τ-b = {_num(c, 'kendall_tau'):.3f}",
        f"- Top-1 accuracy = {c.get('top1_accuracy', float('nan')):.3f}",
        f"- Buckets = {c.get('n_buckets', '?')}  ·  responses = {c.get('n_rows', '?')}",
        f"- Blue−red spread = {bmr.get('mean_spread', float('nan')):.3f} "
        f"(95% CI [{bmr.get('ci_low', float('nan')):.3f}, {bmr.get('ci_high', float('nan')):.3f}])",
        "",
        f"**Verdict:** {'✅ Elo tracks correctness' if v.get('concordance_pass') else '❌ concordance not established'}",
    ]
    if not v.get("reference_beats_difficulty", False):
        lines.append(
            "- ⚠️ The system **does not beat** difficulty-corrected base sampling "
            "(blue ≈ red): Elo may track question difficulty without adding quality. "
            "This is reported as a failure of the system, not hidden (§13)."
        )
    lines += [
        "",
        "## Tier 1b — Scaling",
        f"- best-Elo Spearman ρ(bucket) = "
        f"{report.get('scaling', {}).get('best_elo', {}).get('spearman_rho', float('nan')):.3f}"
        f"  ·  no-saturation: {v.get('scaling_no_saturation')}",
        "",
        "## Tier 3 — Baseline & Ablations",
    ]
    base = report.get("baseline", {}).get("full_vs_best_of_32", {})
    lines.append(
        f"- full vs best_of_32: ΔElo median = {base.get('median_delta', float('nan')):.1f} "
        f"(p = {base.get('p_value', float('nan')):.3f})"
    )
    for variant, stats in report.get("ablation", {}).items():
        lines.append(
            f"- {variant}: ΔElo median = {stats.get('median_delta', float('nan')):.1f} "
            f"(p = {stats.get('p_value', float('nan')):.3f})"
        )
    cost = report.get("cost", {})
    total_calls = cost.get("total_calls")
    cost_line = (
        f"- fresh system runs: {cost.get('fresh_system_runs', '?')}  ·  "
        f"total LLM calls: ~{total_calls:,}"
        if isinstance(total_calls, int) else
        f"- fresh system runs: {cost.get('fresh_system_runs', '?')}"
    )
    lines += [
        "",
        "## Cost",
        cost_line,
        "",
        "## Caveats (§13)",
        "- **Contamination:** GPQA is widely reposted by 2026; high accuracy is "
        "NOT proof of uncontaminated capability.",
        "- **Self-preference bias:** internal Elo ranking uses the configured strong "
        "tier, which may share a model family with generation. Cross-family "
        "independence is claimed only when a separate judge panel was actually run.",
        "- **Small-N:** results below significance are labeled directional, not proven.",
    ]
    return "\n".join(lines)
