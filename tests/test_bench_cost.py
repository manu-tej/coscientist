from bench.cost import fresh_system_runs, estimate_cost, format_estimate


def test_fresh_system_runs_formula():
    assert fresh_system_runs(C=30, a=10, v=5) == 70
    assert fresh_system_runs(C=10, a=0, v=5) == 10


def test_estimate_cost_scales_with_runs():
    est = estimate_cost(C=30, a=10, v=5, calls_per_run=100, ref_samples_per_goal=32)
    assert est["fresh_system_runs"] == 70
    assert est["system_calls"] == 70 * 100
    assert est["base_model_samples"] == 30 * 32
    assert est["total_calls"] == 70 * 100 + 30 * 32


def test_format_estimate_flags_subscription_caveat():
    est = estimate_cost(C=5, a=0, v=5, calls_per_run=50, ref_samples_per_goal=32)
    msg = format_estimate(est, backend="subscription")
    assert "subscription" in msg.lower()
    assert "batch" in msg.lower()
    api_msg = format_estimate(est, backend="api")
    assert "batch" in api_msg.lower()


def test_cli_parser_concordance_defaults():
    from bench.cli import build_parser
    args = build_parser().parse_args(["concordance", "--dataset", "gpqa-bio", "--limit", "25"])
    assert args.command == "concordance"
    assert args.dataset == "gpqa-bio"
    assert args.limit == 25
    assert args.full is False
    assert args.yes is False


def test_cli_parser_ablation_and_all():
    from bench.cli import build_parser
    a = build_parser().parse_args(["ablation", "--goals", "comp_bio", "--limit", "10"])
    assert a.command == "ablation" and a.goals == "comp_bio"
    b = build_parser().parse_args(["all", "--yes"])
    assert b.command == "all" and b.yes is True
