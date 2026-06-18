"""CEO control surface (spec §8). `python company.py <command>`.

Commands: new · program-new · status · cso · quarter [--auto] · gate · advance · kill · hold.
Every command loads the JSON portfolio, mutates, and saves.
"""
from __future__ import annotations

import argparse

from company import engine, store
from company.cso import propose_allocation, recommend_gate
from company.ledger import Ledger
from company.models import GateDecision, ProgramStatus, Stage
from company.science import realized_pos, rnpv_contribution


def _load(args) -> "engine.Portfolio":
    if not store.exists(args.state):
        raise SystemExit(f"no company at {args.state!r} — run `company.py new` first")
    return store.load(args.state)


def _fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def cmd_new(args) -> None:
    pf = engine.new_company(args.name, args.disease, credit_budget=args.budget, seed=args.seed)
    store.save(pf, args.state)
    print(f"Founded «{pf.company.name}» — focus: {pf.company.disease_focus}")
    print(f"  budget: {pf.company.credit_budget:.0f} Modal credits, "
          f"{pf.company.token_budget:,} tokens  (state → {args.state})")


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
    print(f"  Credits: {c.credit_spent:.0f}/{c.credit_budget:.0f} spent "
          f"({led.credits_remaining:.0f} left) · Tokens: {c.token_spent:,}/{c.token_budget:,}")
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
        print("│  method votes: `network` is a live KG-on-CPU proximity (T1); the rest are T0 fixtures")
    print(f"│  realized PoS this transition: {pos:.2f}   (baseline modulated by science)")
    print(f"│  rNPV contribution: {rnpv_contribution(p):+.0f}")
    print(f"│  experiments: {len(result.experiments)} (all T0 — seeded stub, not real models)")
    print(f"│  CSO recommends: {rec.decision.value.upper()} — {rec.rationale}")
    print(f"└─ decide: company.py advance|hold|kill {p.name}\n")


def cmd_cso(args) -> None:
    pf = _load(args)
    led = Ledger(pf.company)
    alloc = propose_allocation(pf.programs, led.credits_remaining)
    print("\nCSO portfolio proposal:")
    print(f"  credit allocation ({led.credits_remaining:.0f} remaining):")
    for pid, cr in sorted(alloc.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {pf.program(pid).name:<16} {cr:>7.1f} credits")
    pending = [p for p in pf.programs if p.id in pf.pending]
    if pending:
        print("  gate recommendations:")
        for p in pending:
            rec = recommend_gate(p, pf.pending[p.id])
            print(f"    {p.name:<16} {rec.decision.value.upper():<8} {rec.rationale}")
    print()


def cmd_quarter(args) -> None:
    pf = _load(args)
    ran = engine.run_quarter(pf, auto=args.auto)
    store.save(pf, args.state)
    print(f"\n── Quarter {pf.company.cycle} ──  ({len(ran)} program(s) ran their stage"
          f"{', auto-resolved by CSO' if args.auto else ''})")
    for p, result, rec in ran:
        verb = "→ " + rec.decision.value if args.auto else "awaiting CEO"
        print(f"  {p.name:<16} {result.stage:<16} conf {result.confidence:.2f}  {verb}")
    if not args.auto and any(p.id in pf.pending for p in pf.programs):
        print("\n  Pending gates — review with `company.py gate <program>`")
    print()


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
    s.add_argument("--budget", type=float, default=500.0, help="Modal credits")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_new)

    s = sub.add_parser("program-new", help="start a program")
    s.add_argument("--name", required=True)
    s.add_argument("--disease", default=None)
    s.add_argument("--value", type=float, default=400.0)
    s.add_argument("--seed", type=int, default=None)
    s.set_defaults(func=cmd_program_new)

    sub.add_parser("status", help="portfolio status").set_defaults(func=cmd_status)
    sub.add_parser("cso", help="CSO proposal").set_defaults(func=cmd_cso)

    s = sub.add_parser("quarter", help="advance one simulated quarter")
    s.add_argument("--auto", action="store_true", help="auto-resolve gates via CSO recs")
    s.set_defaults(func=cmd_quarter)

    s = sub.add_parser("gate", help="show a pending gate packet")
    s.add_argument("program")
    s.set_defaults(func=cmd_gate)

    for name, dec in [("advance", GateDecision.ADVANCE), ("hold", GateDecision.HOLD), ("kill", GateDecision.KILL)]:
        s = sub.add_parser(name, help=f"{name} a program at its gate")
        s.add_argument("program")
        s.set_defaults(func=lambda a, d=dec: _decide(a, d))

    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
