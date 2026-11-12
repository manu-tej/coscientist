"""CEO control surface (spec §8). `python company.py <command>`.

Commands: new · program-new · status · cso · quarter [--auto] · gate · advance · kill · hold.
Every command loads the JSON portfolio, mutates, and saves.
"""
from __future__ import annotations

import argparse
import math
import sys

from company import engine, store
from company.cso import recommend_gate
from company.ledger import Ledger
from company.models import GateDecision, ProgramStatus, Stage
from company.science import realized_pos, rnpv_contribution


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be an integer >= 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _nonnegative_finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "must be a finite non-negative number"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _load(args) -> "engine.Portfolio":
    if not store.exists(args.state):
        raise SystemExit(f"no company at {args.state!r} — run `company.py new` first")
    return store.load(args.state)


def _fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def cmd_new(args) -> None:
    # store.save is an atomic os.replace with no backup, so re-running the founding
    # command would irrecoverably wipe programs, gates, stage runs and the spend
    # ledger — state that represents real billed work — while reporting success.
    if store.exists(args.state) and not args.force:
        raise SystemExit(
            f"refusing to overwrite an existing company at {args.state!r}. "
            f"Pass --force to replace it, or use --state PATH for a new one."
        )
    pf = engine.new_company(
        args.name,
        args.disease,
        credit_budget=args.budget,
        token_budget=args.token_budget,
        seed=args.seed,
    )
    store.save(pf, args.state)
    print(f"Founded «{pf.company.name}» — focus: {pf.company.disease_focus}")
    print(f"  budget: {pf.company.credit_budget:.0f} Modal credits, "
          f"{pf.company.token_budget:,} portable token units  (state → {args.state})")


def cmd_program_new(args) -> None:
    pf = _load(args)
    disease = args.disease or pf.company.disease_focus
    p = engine.add_program(pf, args.name, disease, estimated_value=args.value, seed=args.seed)
    store.save(pf, args.state)
    print(f"Program «{p.name}» [{p.id}] founded — {disease}, est. value {_fmt_money(p.estimated_value)}")


def cmd_status(args) -> None:
    pf = _load(args)
    c = pf.company
    led = Ledger(c)
    print(f"\n═══ {c.name} — {c.disease_focus} ═══  (quarter {c.cycle})")
    total_credits = c.credit_spent + c.failed_credit_spent
    total_tokens = c.token_spent + c.failed_token_spent
    print(
        f"  Credits: {total_credits:.0f}/{c.credit_budget:.0f} used "
        f"({c.credit_spent:.0f} committed, "
        f"{c.failed_credit_spent:.0f} failed-attempt, "
        f"{led.credits_remaining:.0f} left) · Portable tokens: "
        f"{total_tokens:,}/{c.token_budget:,} used "
        f"({c.token_spent:,} committed, "
        f"{c.failed_token_spent:,} failed-attempt, "
        f"{led.tokens_remaining:,} left)"
    )
    if not pf.programs:
        print("  (no programs yet)\n")
        return
    print(f"\n  {'PROGRAM':<16}{'STAGE':<16}{'STATUS':<11}{'CONF':>6}{'cumPoS':>8}"
          f"{'SPEND':>8}{'rNPV':>9}  LEAD")
    print("  " + "─" * 96)
    for p in sorted(pf.programs, key=lambda x: rnpv_contribution(x), reverse=True):
        pend = " *gate" if p.id in pf.pending else ""
        print(f"  {p.name:<16}{p.stage.value:<16}{p.status.value+pend:<11}"
              f"{p.confidence:>6.2f}{p.cumulative_pos:>8.3f}{p.credits_spent:>8.1f}"
              f"{rnpv_contribution(p):>9.0f}  {p.lead_candidate or '—'}")
    total = sum(rnpv_contribution(p) for p in pf.programs if p.status is not ProgramStatus.KILLED)
    print(f"\n  Portfolio rNPV (ex-killed): {_fmt_money(total)}\n")


def _print_packet(pf, p) -> None:
    from company.models import next_stage
    result = pf.pending[p.id]
    rec = recommend_gate(p, result)
    pos = realized_pos(p.stage, result.confidence, result.red_flags)
    print(f"\n┌─ GATE  «{p.name}»  {p.stage.value} → {next_stage(p.stage).value}")
    print(f"│  confidence {result.confidence:.2f} · method-agreement {result.method_agreement:.2f} "
          f"· red flags {result.red_flags}")
    if result.top_candidates:
        print("│  top candidates (consensus):")
        for c in result.top_candidates[:5]:
            print(f"│    {c.drug:<16} {c.mean_score:.2f} across {c.n_methods} methods  — {c.rationale}")
        lead = result.top_candidates[0]
        if getattr(lead, "evidence", None) is not None:
            print(f"│  evidence chain for «{lead.drug}»:")
            for line in lead.evidence.explain():
                print(f"│    {line}")
        if result.method_provenance:
            prov = ", ".join(f"{n}×{t}" for t, n in sorted(result.method_provenance.items()))
            print(f"│  method-vote fidelity (via model registry): {prov}  "
                  f"(T1=real KG/literature, T0=estimate)")
    print(f"│  realized PoS this transition: {pos:.2f}   (baseline modulated by science)")
    print(f"│  rNPV contribution: {rnpv_contribution(p):+.0f}")
    tiers: dict[str, int] = {}
    for experiment in result.experiments:
        tiers[experiment.fidelity_tier] = tiers.get(experiment.fidelity_tier, 0) + 1
    tier_summary = ", ".join(f"{n}×{tier}" for tier, n in sorted(tiers.items()))
    print(f"│  experiments: {len(result.experiments)} ({tier_summary or 'none'})")
    print(f"│  CSO recommends: {rec.decision.value.upper()} — {rec.rationale}")
    print(f"└─ decide: company.py advance|hold|kill {p.name}\n")


def cmd_cso(args) -> None:
    import asyncio

    from company.cso import CSOAgent, explore_exploit_beta

    pf = _load(args)
    led = Ledger(pf.company)
    backend = None
    if args.llm:
        from tools.llm import load_backend_environment, make_backend
        from tools.resources import load_yaml_config
        load_backend_environment()
        backend = make_backend(load_yaml_config())

    async def build_plan():
        try:
            return await CSOAgent(backend).propose_plan(
                pf.programs,
                pf.pending,
                led.credits_remaining,
                cycle=pf.company.cycle,
            )
        finally:
            if backend is not None:
                from tools.llm import close_backend
                await close_backend(backend)

    plan = asyncio.run(build_plan())

    print(f"\nCSO portfolio plan  [{plan.source}]  cycle {pf.company.cycle} · "
          f"{led.credits_remaining:.0f} credits left · "
          f"explore→exploit beta {explore_exploit_beta(pf.company.cycle):.1f}")
    print("  " + "─" * 92)
    for i, a in enumerate(plan.actions, 1):
        print(f"  {i}. [{a.kind:<5}] {a.target:<34} {a.detail:<16} {a.rationale}")
    if plan.summary:
        print(f"\n  CSO: {plan.summary}")
    print()


def _persist_failed_attempt_delta(
    state_path: str,
    attempted: "engine.Portfolio",
    *,
    token_before: int,
    credit_before: float,
) -> None:
    """Durably debit only operational usage from a failed async quarter.

    ``run_quarter_async`` is a scientific-state transaction: on failure it rolls
    back programs, gates, experiments, and committed spend, while retaining the
    measured provider-attempt usage on the caller's Company object. The CLI must
    persist that narrow delta before re-raising, or restarting from the prior JSON
    snapshot would reset the lifetime resource wall.

    Reloading the current snapshot and applying only the monotonic delta avoids
    writing any accidental scientific mutations from the failed in-memory object.
    """
    token_delta = attempted.company.failed_token_spent - token_before
    credit_delta = attempted.company.failed_credit_spent - credit_before
    if token_delta < 0 or credit_delta < -1e-9:
        raise RuntimeError(
            "failed-attempt accounting moved backwards during quarter failure"
        )
    if token_delta == 0 and abs(credit_delta) <= 1e-9:
        return

    durable = store.load(state_path)
    durable.company.failed_token_spent += token_delta
    durable.company.failed_credit_spent += max(0.0, credit_delta)
    store.save(durable, state_path)


def cmd_quarter(args) -> None:
    pf = _load(args)
    from company.registry import build_default_registry
    reg = build_default_registry(enable_literature=args.literature,
                                 enable_structure=args.structure,
                                 enable_affinity=args.affinity)
    if args.literature:
        print(
            "  (literature method REQUESTED — live PubMed/T1 when reachable; "
            "fallback provenance is recorded)"
        )
    if args.structure:
        print(
            "  (structure method REQUESTED — Modal ChemBERTa/T2 when deployed "
            "and authenticated; otherwise T0 fallback)"
        )
    if args.affinity:
        print(
            "  (affinity method REQUESTED — Modal Boltz-2/T2 for leaders when "
            "deployed and authenticated; otherwise T0 fallback)"
        )

    if args.hierarchical or args.cso_allocation:
        import asyncio
        from company.cso import propose_allocation
        from company.hierarchy import CoScientistStageExecutor, SeededStageExecutor

        allocation = None
        if args.cso_allocation:
            allocation = propose_allocation(
                pf.programs, Ledger(pf.company).credits_remaining,
                cycle=pf.company.cycle,
                pending_ids=set(pf.pending),
            )
            print("  (CEO-approved deterministic CSO allocation will gate stage launches)")

        if args.hierarchical:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            from tools.llm import configure_backend, make_backend
            from tools.resources import load_yaml_config
            backend_cfg = load_yaml_config()
            selected_provider, model_cfg = configure_backend(
                backend_cfg,
                provider=args.provider,
                model_fast=args.model_fast,
                model_strong=args.model_strong,
            )
            backend = make_backend(
                backend_cfg, provider_override=selected_provider,
            )
            executor = CoScientistStageExecutor(
                backend,
                n_workers=args.workers,
                max_tasks=args.max_agent_tasks,
                max_time_seconds=args.max_agent_time,
                evidence_provider=selected_provider,
            )
            print(f"  (HIERARCHICAL: one co-scientist team/program stage; "
                  f"{args.workers} agents/team, {args.program_workers} concurrent programs; "
                  f"shared portable-token envelope={Ledger(pf.company).tokens_remaining:,}; "
                  f"provider={selected_provider}, fast={model_cfg.get('model_fast', model_cfg.get('model', ''))}, "
                  f"strong={model_cfg.get('model_strong', model_cfg.get('model', ''))})")
        else:
            executor = SeededStageExecutor()

        failed_tokens_before = pf.company.failed_token_spent
        failed_credits_before = pf.company.failed_credit_spent

        async def execute_quarter():
            try:
                try:
                    ran = await engine.run_quarter_async(
                        pf,
                        executor=executor,
                        auto=args.auto,
                        registry=reg,
                        n_program_workers=args.program_workers,
                        approved_allocation=allocation,
                    )
                except BaseException as run_error:
                    try:
                        _persist_failed_attempt_delta(
                            args.state,
                            pf,
                            token_before=failed_tokens_before,
                            credit_before=failed_credits_before,
                        )
                    except BaseException as persistence_error:
                        persistence_error.add_note(
                            "failed while durably recording resource usage from "
                            "an aborted quarter "
                            f"({type(run_error).__name__}: {run_error})"
                        )
                        raise persistence_error from run_error
                    raise

                # A backend close can fail or hang after the scientific
                # transaction commits. Make the completed quarter durable before
                # cleanup starts so that failure cannot erase successful work.
                store.save(pf, args.state)
                return ran
            finally:
                if args.hierarchical:
                    from tools.llm import close_backend
                    await close_backend(backend)

        ran = asyncio.run(execute_quarter())
    else:
        ran = engine.run_quarter(pf, auto=args.auto, registry=reg)
        store.save(pf, args.state)
    print(f"\n── Quarter {pf.company.cycle} ──  ({len(ran)} program(s) ran their stage"
          f"{', auto-resolved by CSO' if args.auto else ''})")
    for p, result, rec in ran:
        verb = "→ " + rec.decision.value if args.auto else "awaiting CEO"
        print(f"  {p.name:<16} {result.stage:<16} conf {result.confidence:.2f}  {verb}")
    if not args.auto and any(p.id in pf.pending for p in pf.programs):
        print("\n  Pending gates — review with `company.py gate <program>`")
    print()


def cmd_package(args) -> None:
    """Emit the clinical-hypothesis package for a program (spec §10 deliverable)."""
    import json as _json

    pf = _load(args)
    p = pf.program(args.program)
    try:
        pkg = engine.build_clinical_hypothesis_package(pf, p)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.json:
        print(_json.dumps(pkg, indent=2))
        return
    print(f"\n═══ CLINICAL-HYPOTHESIS PACKAGE — {p.name} ═══\n")
    print(f"  {pkg['drug']} → {pkg['indication']}")
    print(f"  mechanism:  {pkg['mechanism']}")
    print(f"  confidence: {pkg['confidence']:.2f} (agreement {pkg['method_agreement']:.2f})")
    print(f"  evidence chain ({pkg['real_evidence_count']} computed evidence node(s)):")
    for line in pkg["evidence_chain"]:
        print(f"    {line}")
    print(f"  known safety: {pkg['known_safety']}")
    print("  suggested validation:")
    for s in pkg["suggested_validation"]:
        print(f"    - {s}")
    print()


def cmd_validate(args) -> int:
    """The company's report card: live literature signal vs the curated gold set."""
    from company import validation
    try:
        pairs = validation.gold_pairs()
        drugs = validation.all_drugs()
    except validation.ValidationDataError as exc:
        print(
            f"  VALIDATION INVALID: {exc}; no scientific verdict emitted",
            file=sys.stderr,
        )
        return 2
    print("\n  Validating against the curated gold set (live PubMed co-occurrence)…")
    print(f"  {len(validation.GOLD)} diseases, {len(drugs)} drugs "
          f"→ {len(pairs)} (drug, disease) pairs. This hits the network.\n")
    try:
        rep = validation.validate()
    except validation.ValidationCoverageError as exc:
        print(
            f"  VALIDATION INCOMPLETE: scored {exc.scored}/{exc.expected}; "
            "no scientific verdict emitted",
            file=sys.stderr,
        )
        return 2
    except validation.ValidationDataError as exc:
        print(
            f"  VALIDATION INVALID: {exc}; no scientific verdict emitted",
            file=sys.stderr,
        )
        return 2

    print(
        f"  REPORT COMPLETE: scored {rep.n_scored}/{rep.n_pairs} pairs "
        f"({rep.n_gold_scored}/{rep.n_gold_expected} gold)"
    )
    print(f"  AUROC (gold vs decoy separation):   {rep.auroc:.3f}")
    print(f"  mean gold score / mean decoy score: {rep.mean_gold:.3f} / {rep.mean_decoy:.3f}")
    print(f"  recall@1 / recall@3 (per disease):  {rep.recall_at_1:.2f} / {rep.recall_at_3:.2f}")
    print(f"  negative-control max decoy score:   {rep.negative_control_max:.3f}")
    c = rep.concordance
    print(f"  concordance (score↔gold, within disease): logistic coef "
          f"{c.get('logistic_coef', float('nan')):+.2f}/SD (LRT p={c.get('logistic_p', float('nan')):.3f}; "
          f"{c.get('logistic_n_rows', '?')} pairs, {c.get('logistic_n_questions', '?')} diseases"
          f"{'; SEPARATED' if c.get('logistic_separated') else ''}), "
          f"top-1 acc {c.get('top1_accuracy', float('nan')):.2f}")
    print(
        "\n  Interpretation: this is a complete observed report. "
        "No pass/fail quality threshold is configured.\n"
    )
    return 0


def cmd_gate(args) -> None:
    pf = _load(args)
    p = pf.program(args.program)
    if p.id not in pf.pending:
        raise SystemExit(f"«{p.name}» has no pending gate (run a quarter first)")
    _print_packet(pf, p)


def _decide(args, decision: GateDecision) -> None:
    pf = _load(args)
    p = pf.program(args.program)
    rec = engine.resolve_gate(pf, p, decision)
    store.save(pf, args.state)
    print(f"\n{decision.value.upper()} «{p.name}»: {rec.note}")
    if rec.survived_roll is not None:
        print(f"  stochastic roll vs PoS {rec.realized_pos:.2f}: "
              f"{'SURVIVED' if rec.survived_roll else 'FAILED (attrition)'}")
    print(f"  now: stage={p.stage.value}, status={p.status.value}, "
          f"cumPoS={p.cumulative_pos:.3f}, rNPV={rnpv_contribution(p):+.0f}\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="company.py", description="AI-native drug-repurposing techbio (CEO console)")
    ap.add_argument("--state", default=store.DEFAULT_PATH, help="portfolio JSON path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("new", help="found a company")
    s.add_argument("--name", required=True)
    s.add_argument("--disease", required=True)
    s.add_argument("--budget", type=_nonnegative_finite_float, default=500.0,
                   help="Modal credits")
    s.add_argument("--token-budget", type=_nonnegative_int, default=5_000_000,
                   help=("hard cap on portable chars/4 prompt estimates plus requested "
                         "output tokens (not exact provider billing)"))
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true",
                   help="overwrite an existing company state file (destroys its ledger)")
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("program-new", help="start a program")
    s.add_argument("--name", required=True)
    s.add_argument("--disease", default=None)
    s.add_argument("--value", type=_nonnegative_finite_float, default=400.0)
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(func=cmd_program_new)

    sub.add_parser("status", help="portfolio status").set_defaults(func=cmd_status)
    s = sub.add_parser("cso", help="CSO portfolio plan (ranked action list)")
    s.add_argument("--llm", action="store_true",
                   help="let the CSO agent re-prioritize via LLM (default: deterministic)")
    s.set_defaults(func=cmd_cso)
    sub.add_parser("validate", help="report card vs the curated gold set (live)").set_defaults(func=cmd_validate)

    s = sub.add_parser("quarter", help="advance one simulated quarter")
    s.add_argument("--auto", action="store_true", help="auto-resolve gates via CSO recs")
    s.add_argument("--literature", action="store_true",
                   help="enable the live PubMed co-occurrence method (network; T1)")
    s.add_argument("--structure", action="store_true",
                   help="enable the Modal GPU ChemBERTa analogy method (BILLS Modal credits; T2)")
    s.add_argument("--affinity", action="store_true",
                   help="enable Modal GPU Boltz-2 affinity for portfolio leaders (BILLED/T2)")
    s.add_argument("--hierarchical", action="store_true",
                   help="run each stage through a real co-scientist Supervisor/team")
    s.add_argument("--workers", type=_positive_int, default=4,
                   help="agent workers inside each hierarchical stage team")
    s.add_argument("--program-workers", type=_positive_int, default=2,
                   help="program stage teams allowed to run concurrently")
    s.add_argument("--max-agent-tasks", type=_positive_int, default=24,
                   help="co-scientist agent tasks per stage team")
    s.add_argument("--max-agent-time", type=_positive_int, default=900,
                   help=("async stage-team deadline in seconds; timed-out sync "
                         "adapter work is discarded and conservatively charged"))
    s.add_argument("--provider", default=None,
                   help="hierarchical LLM provider override")
    s.add_argument("--model-fast", default=None,
                   help="fast-tier model override")
    s.add_argument("--model-strong", default=None,
                   help="strong-tier model override")
    s.add_argument("--cso-allocation", action="store_true",
                   help="treat the deterministic CSO credit allocation as CEO-approved envelopes")
    s.set_defaults(func=cmd_quarter)

    s = sub.add_parser("gate", help="show a pending gate packet")
    s.add_argument("program")
    s.set_defaults(func=cmd_gate)

    s = sub.add_parser("package", help="clinical-hypothesis package (the deliverable)")
    s.add_argument("program")
    s.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    s.set_defaults(func=cmd_package)

    for name, dec in [("advance", GateDecision.ADVANCE), ("hold", GateDecision.HOLD), ("kill", GateDecision.KILL)]:
        s = sub.add_parser(name, help=f"{name} a program at its gate")
        s.add_argument("program")
        s.set_defaults(func=lambda a, d=dec: _decide(a, d))

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    return result if isinstance(result, int) else 0
