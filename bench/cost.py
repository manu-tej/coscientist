from __future__ import annotations


def fresh_system_runs(C: int, a: int, v: int) -> int:
    """§17.2 run accounting: concordance runs plus the extra ablation variants
    (the 'full' variant is reused from the concordance set, so v-1 extra each)."""
    return C + a * (v - 1)


def estimate_cost(
    C: int, a: int, v: int, calls_per_run: int = 100, ref_samples_per_goal: int = 32,
) -> dict:
    runs = fresh_system_runs(C, a, v)
    system_calls = runs * calls_per_run
    base_samples = C * ref_samples_per_goal   # shared by red line + best_of_32
    return {
        "fresh_system_runs": runs,
        "system_calls": system_calls,
        "base_model_samples": base_samples,
        "total_calls": system_calls + base_samples,
    }


def format_estimate(est: dict, backend: str = "api") -> str:
    """Human-readable pre-run estimate. On a subscription, warns that batch (−50%)
    and deep prompt caching are unavailable and the credit pool draws down first."""
    lines = [
        "── Pre-run cost estimate ──",
        f"  Fresh system runs:   {est['fresh_system_runs']}",
        f"  System LLM calls:    ~{est['system_calls']:,} (sequential within each run)",
        f"  Base-model samples:  {est['base_model_samples']:,} (batchable, independent)",
        f"  Total LLM calls:     ~{est['total_calls']:,}",
        f"  Backend:             {backend}",
    ]
    if backend == "subscription":
        lines += [
            "  NOTE: on a subscription, Message Batches (−50%) and deep prompt",
            "  caching (−90% reads) are UNAVAILABLE; the Agent-SDK credit pool",
            "  draws down first, then bills at standard rates. For the full 70-run",
            "  experiment, the metered API + batch path is materially cheaper.",
        ]
    else:
        lines += [
            "  TIP: batch the independent classes (base samples, judge, cross-",
            "  tournament) via the Message Batches API for −50% on those calls.",
        ]
    return "\n".join(lines)
