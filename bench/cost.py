from __future__ import annotations


def fresh_system_runs(C: int, a: int, v: int) -> int:
    """§17.2 run accounting: concordance runs plus the extra ablation variants
    (the 'full' variant is reused from the concordance set, so v-1 extra each)."""
    return C + a * (v - 1)


def estimate_cost(
    C: int, a: int, v: int, calls_per_run: int = 100, ref_samples_per_goal: int = 32,
    calls_per_task_upper: int = 50,
) -> dict:
    if min(C, a, calls_per_run, ref_samples_per_goal) < 0 or v < 1:
        raise ValueError("cost inputs must be nonnegative and variants must be >= 1")
    runs = fresh_system_runs(C, a, v)
    system_calls = runs * calls_per_run
    system_calls_upper = system_calls * calls_per_task_upper
    base_samples = C * ref_samples_per_goal   # shared by red line + best_of_32
    return {
        "fresh_system_runs": runs,
        "system_calls": system_calls,
        "system_calls_upper_bound": system_calls_upper,
        "base_model_samples": base_samples,
        "total_calls": system_calls + base_samples,
        "total_calls_upper_bound": system_calls_upper + base_samples,
    }


def format_estimate(est: dict, backend: str = "api") -> str:
    """Human-readable call estimate with a deliberately conservative billing note."""
    lines = [
        "── Pre-run cost estimate ──",
        f"  Fresh system runs:   {est['fresh_system_runs']}",
        f"  System LLM calls:    ~{est['system_calls']:,}–"
        f"{est.get('system_calls_upper_bound', est['system_calls']):,} "
        "(tasks can make multiple calls; workers run concurrently)",
        f"  Base-model samples:  {est['base_model_samples']:,} (batchable, independent)",
        f"  Total LLM calls:     ~{est['total_calls']:,}–"
        f"{est.get('total_calls_upper_bound', est['total_calls']):,}",
        f"  Billing route:       {backend}",
    ]
    if backend == "subscription":
        lines += [
            "  NOTE: subscription-backed providers may enforce plan-specific usage",
            "  limits. This estimate counts calls; it does not predict quota use or fees.",
        ]
    elif backend == "api":
        lines += [
            "  NOTE: metered cost depends on the selected models and token usage.",
            "  This harness does not assume a provider-specific batch discount.",
        ]
    elif backend == "custom":
        lines += [
            "  NOTE: the OpenAI-compatible route may be local or hosted; billing",
            "  and quota behavior cannot be inferred from its protocol.",
        ]
    elif backend == "mixed":
        lines += [
            "  NOTE: this sweep mixes API, subscription, and/or custom routes.",
            "  Review each cell's provider before approving spend.",
        ]
    else:
        lines += [
            "  NOTE: billing and quota behavior are unknown for this route.",
        ]
    return "\n".join(lines)
