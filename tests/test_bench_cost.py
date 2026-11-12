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
    assert est["system_calls_upper_bound"] > est["system_calls"]


def test_format_estimate_flags_subscription_caveat():
    est = estimate_cost(C=5, a=0, v=5, calls_per_run=50, ref_samples_per_goal=32)
    msg = format_estimate(est, backend="subscription")
    assert "subscription" in msg.lower()
    assert "quota" in msg.lower()
    assert "70-run" not in msg
    api_msg = format_estimate(est, backend="api")
    assert "metered cost" in api_msg.lower()
    assert "-50%" not in api_msg
    custom_msg = format_estimate(est, backend="custom")
    assert "cannot be inferred" in custom_msg.lower()
    mixed_msg = format_estimate(est, backend="mixed")
    assert "mixes" in mixed_msg.lower()


def test_cli_parser_concordance_defaults():
    from bench.cli import build_parser
    args = build_parser().parse_args(["concordance", "--dataset", "gpqa-bio", "--limit", "25"])
    assert args.command == "concordance"
    assert args.dataset == "gpqa-bio"
    assert args.limit == 25
    assert args.full is False
    assert args.yes is False
    assert args.backend == "auto"
    assert args.manifest == "bench/manifest.sqlite"
    assert args.no_cache is False


def test_cli_parser_can_select_manifest_or_disable_cache():
    from bench.cli import build_parser

    selected = build_parser().parse_args([
        "rediscover", "--manifest", "artifacts/completed.sqlite",
    ])
    assert selected.manifest == "artifacts/completed.sqlite"
    assert selected.no_cache is False

    disabled = build_parser().parse_args(["rediscover", "--no-cache"])
    assert disabled.no_cache is True


def test_cli_only_advertises_end_to_end_commands():
    from bench.cli import build_parser
    import pytest

    for command in ("scaling", "judge", "baseline", "ablation", "all"):
        with pytest.raises(SystemExit):
            build_parser().parse_args([command])


def test_cli_parser_exposes_model_and_worker_sweep():
    from bench.cli import build_parser
    args = build_parser().parse_args([
        "sweep", "--benchmark", "rediscover",
        "--worker-counts", "1,4",
        "--models", "claude-code:sonnet:opus",
        "--models", "openai-compatible:local",
        "--seeds", "0,1", "--ranking-mode", "absolute", "--yes",
    ])
    assert args.command == "sweep"
    assert args.worker_counts == "1,4"
    assert len(args.models) == 2
    assert args.seeds == "0,1"
    assert args.ranking_mode == "absolute"


def test_cli_rejects_negative_or_zero_cost_dimensions():
    import pytest
    from bench.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["concordance", "--limit", "-1"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["concordance", "--max-tasks", "0"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["concordance", "--workers", "0"])
