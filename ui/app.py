import html
import re
import yaml
from pathlib import Path

import gradio as gr

from core.state import StateStore
from ui.data import inject_expert_hypothesis
from ui import explore


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


_CSS = """
.gradio-container { max-width: 1240px !important; }
#stats { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:2px 0; }
#stats .chip { background:#eef2ff; border:1px solid #c7d2fe; border-radius:999px; padding:3px 11px; font-size:13px; color:#3730a3; }
#stats .chip b { color:#1e1b4b; }
#stats .goal { color:#6b7280; font-size:12.5px; }
.scroll { max-height:70vh; overflow-y:auto; padding-right:6px; }
.hyp-card,.match-card { border:1px solid #e7e9ee; border-radius:11px; padding:9px 12px; margin-bottom:8px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.hyp-head,.match-head { display:flex; align-items:center; gap:8px; margin-bottom:5px; font-size:13px; }
.rank { font-weight:700; color:#111827; }
.elo { background:#111827; color:#fff; border-radius:6px; padding:1px 8px; font-size:12px; font-weight:600; }
.method { background:#f3f4f6; border:1px solid #e5e7eb; border-radius:5px; padding:1px 7px; font-size:11px; color:#374151; }
.method.expert { background:#fef3c7; border-color:#fde68a; color:#92400e; }
.meta { margin-left:auto; color:#9ca3af; font-size:11px; white-space:nowrap; }
.summary { color:#374151; font-size:13.5px; line-height:1.5; }
.bar { height:4px; background:#eef0f4; border-radius:3px; margin-top:7px; overflow:hidden; }
.bar > span { display:block; height:100%; background:linear-gradient(90deg,#6366f1,#22c55e); }
details { margin-top:7px; } details > summary { cursor:pointer; color:#4f46e5; font-size:12px; }
.full { white-space:pre-wrap; font-size:12.5px; color:#374151; background:#fafafa; border:1px solid #eee; border-radius:8px; padding:8px 10px; margin-top:6px; max-height:340px; overflow:auto; }
.rev { font-size:12px; color:#444; border-left:3px solid #c7d2fe; padding:2px 8px; margin:5px 0; }
.played { font-size:11.5px; color:#6b7280; margin-top:4px; }
.played .w { color:#16a34a; } .played .l { color:#dc2626; }
.match-body { display:flex; align-items:stretch; gap:10px; }
.side { flex:1; border:1px solid #eee; border-radius:8px; padding:6px 9px; font-size:12.5px; color:#374151; }
.side.win { border-color:#86efac; background:#f0fdf4; }
.side .e { float:right; color:#6b7280; font-size:11px; font-weight:600; }
.side .up { color:#16a34a; } .side .down { color:#dc2626; }
.vs { align-self:center; color:#9ca3af; font-size:11px; }
pre.tx { white-space:pre-wrap; font-size:11.5px; background:#fafafa; border:1px solid #eee; border-radius:8px; padding:8px; max-height:300px; overflow:auto; }
.tl { font-size:12.5px; } .tl .row { display:flex; gap:8px; padding:2px 0; border-bottom:1px dashed #eee; }
.tl .k { width:74px; font-weight:600; } .tl .k.generate{color:#2563eb}.tl .k.match{color:#7c3aed}.tl .k.review{color:#059669}.tl .k.meta{color:#b45309}
.tl .t { color:#9ca3af; width:150px; } .tl .x { color:#374151; }
.eval-card { border:1px solid #e7e9ee; border-radius:11px; padding:12px 14px; margin-bottom:10px; background:#fff; }
.eval-card h4 { margin:0 0 6px; } .eval-kv { font-size:13px; color:#374151; }
.eval-tbl { width:100%; border-collapse:collapse; font-size:12.5px; margin-top:8px; }
.eval-tbl td,.eval-tbl th { border:1px solid #eee; padding:3px 8px; text-align:left; }
.pass { color:#16a34a; font-weight:600; } .fail { color:#dc2626; font-weight:600; }
"""

_BOILER = re.compile(
    r"^(?:\s*(?:novel research hypothesis|hypothesis|introduction|abstract|summary|statement|---)\s*[:\-]*\s*)+",
    re.IGNORECASE)


def _clean(text: str, n: int = 240) -> str:
    t = re.sub(r"[#*_`>\[\]]", "", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    t = _BOILER.sub("", t).strip()
    return (t[: n - 1] + "…") if len(t) > n else t


def _esc(s: str) -> str:
    return html.escape(s or "")


# ── tab renderers (pure: db_path, run_id → HTML) ───────────────────────────

def _stats_html(db_path, run_id) -> str:
    import sqlite3
    c = sqlite3.connect(db_path); c.row_factory = sqlite3.Row
    try:
        goal = (c.execute("SELECT goal FROM configs WHERE run_id=?", (run_id,)).fetchone() or {"goal": ""})["goal"]
        elos = [r[0] for r in c.execute("SELECT elo_rating FROM hypotheses WHERE run_id=? AND status='active'", (run_id,))]
        nm = c.execute("SELECT COUNT(*) FROM tournament_matches WHERE run_id=?", (run_id,)).fetchone()[0]
    finally:
        c.close()
    top = round(max(elos), 1) if elos else 1200.0
    spread = round(max(elos) - min(elos), 1) if elos else 0.0
    return ('<div id="stats">'
            f'<span class="chip"><b>{len(elos)}</b> hypotheses</span>'
            f'<span class="chip"><b>{nm}</b> matches</span>'
            f'<span class="chip">top Elo <b>{top}</b></span>'
            f'<span class="chip">spread <b>{spread}</b></span>'
            f'<span class="goal">{_esc((goal or "")[:150])}</span></div>')


def render_hypotheses(db_path, run_id) -> str:
    rows = explore.hypotheses_detailed(db_path, run_id)
    if not rows:
        return "<p style='color:#9ca3af;padding:8px'>No hypotheses yet…</p>"
    elos = [r["elo"] for r in rows]
    lo, span = min(elos), (max(elos) - min(elos)) or 1.0
    out = ['<div class="scroll">']
    for i, r in enumerate(rows, 1):
        pct = int(100 * (r["elo"] - lo) / span)
        exp = " expert" if r["source"] == "expert" else ""
        wins = sum(1 for m in r["matches"] if m["won"])
        revs = "".join(f'<div class="rev"><b>tier {rv["tier"]}</b> {_esc(_clean(rv["critique"], 400))}</div>'
                       for rv in r["reviews"]) or "<i style='color:#9ca3af'>no reviews</i>"
        played = "".join(
            f'<span class="{"w" if m["won"] else "l"}">{"✓" if m["won"] else "✗"} {_esc(m["opponent"][:32])} ({m["delta"]:+})</span>  '
            for m in r["matches"]) or "<i style='color:#9ca3af'>no matches</i>"
        out.append(
            f'<div class="hyp-card"><div class="hyp-head">'
            f'<span class="rank">#{i}</span><span class="elo">Elo {r["elo"]}</span>'
            f'<span class="method{exp}">{_esc(r["method"])}</span>'
            f'<span class="meta">{len(r["matches"])} matches ({wins}W) · {len(r["reviews"])} reviews</span></div>'
            f'<div class="summary">{_esc(_clean(r["text"], 380))}</div>'
            f'<div class="bar"><span style="width:{pct}%"></span></div>'
            f'<details><summary>full text · reviews · match record</summary>'
            f'<div class="full">{_esc(r["text"])}</div>'
            f'<div class="played">{played}</div>{revs}</details></div>')
    out.append("</div>")
    return "".join(out)


def render_tournament(db_path, run_id) -> str:
    rows = explore.match_history(db_path, run_id)
    if not rows:
        return "<p style='color:#9ca3af;padding:8px'>No tournament matches yet…</p>"
    out = [f'<div style="color:#6b7280;font-size:12px;margin-bottom:6px">{len(rows)} matches, oldest → newest</div><div class="scroll">']
    for m in rows:
        d1, d2 = round(m["e1_after"] - m["e1_before"], 1), round(m["e2_after"] - m["e2_before"], 1)
        s1 = "win" if m["h1_won"] else ""
        s2 = "" if m["h1_won"] else "win"
        def edelta(b, a, d):
            cls = "up" if d >= 0 else "down"
            return f'<span class="e">{b}→{a} <span class="{cls}">({d:+})</span></span>'
        tx = (f'<details><summary>debate transcript</summary><pre class="tx">{_esc(m["transcript"][:30000])}</pre></details>'
              if m["transcript"] else "")
        side1 = (f'<div class="side {s1}">{edelta(m["e1_before"], m["e1_after"], d1)}'
                 f'<details><summary>{_esc(_clean(m["h1"], 200))}</summary>'
                 f'<div class="full">{_esc(m["h1"])}</div></details></div>')
        side2 = (f'<div class="side {s2}">{edelta(m["e2_before"], m["e2_after"], d2)}'
                 f'<details><summary>{_esc(_clean(m["h2"], 200))}</summary>'
                 f'<div class="full">{_esc(m["h2"])}</div></details></div>')
        out.append(
            f'<div class="match-card"><div class="match-head">'
            f'<span class="rank">match #{m["n"]}</span>'
            f'<span class="method">{m["type"]}</span>'
            f'<span class="meta">{_esc(m["created_at"])}</span></div>'
            f'<div class="match-body">{side1}<div class="vs">vs</div>{side2}</div>{tx}</div>')
    out.append("</div>")
    return "".join(out)


def render_timeline(db_path, run_id) -> str:
    ev = explore.timeline(db_path, run_id)
    if not ev:
        return "<p style='color:#9ca3af;padding:8px'>No events yet…</p>"
    rows = "".join(
        f'<div class="row"><span class="k {e["kind"]}">{e["kind"]}</span>'
        f'<span class="t">{_esc(e["t"])}</span><span class="x">{_esc(e["text"])}</span></div>'
        for e in ev)
    return f'<div class="tl scroll">{rows}</div>'


def render_eval(report_path: str) -> str:
    if not report_path:
        return "<p style='color:#9ca3af'>Select a report.</p>"
    rep = explore.load_eval_report(report_path)
    if not rep:
        return f"<p style='color:#dc2626'>Could not read {_esc(report_path)}</p>"
    meta = rep.get("meta", {})
    head = (f'<div class="eval-kv"><b>{_esc(Path(report_path).name)}</b> · '
            f'dataset {_esc(str(meta.get("dataset", "?")))} · '
            f'system {_esc(str(meta.get("system_version", "?")))}</div>')
    if "rediscovery" in rep:
        r = rep["rediscovery"]
        rows = "".join(
            f'<tr><td>{_esc(g["goal_id"])}</td><td>{g["pool_recall"]:.2f}</td>'
            f'<td>{g["top1_recall"]:.2f}</td><td>{g["top3_recall"]:.2f}</td>'
            f'<td>{g["n_hyps"]}</td><td>{g["n_gold"]}</td></tr>'
            for g in r.get("per_goal", []))
        return (f'<div class="eval-card"><h4>Rediscovery — gold-entity recall</h4>{head}'
                f'<table class="eval-tbl"><tr><th>task</th><th>pool</th><th>top1</th><th>top3</th><th>#hyps</th><th>#gold</th></tr>{rows}</table>'
                f'<div class="eval-kv" style="margin-top:8px">'
                f'MEAN pool <b>{r.get("mean_pool_recall",0):.2f}</b> · top1 <b>{r.get("mean_top1_recall",0):.2f}</b> · top3 <b>{r.get("mean_top3_recall",0):.2f}</b><br>'
                f'Elo↔recall Spearman <b>{r.get("elo_recall_spearman",float("nan")):.3f}</b> '
                f'(p={r.get("elo_recall_p",float("nan")):.3f}, n={r.get("n_hyps_scored",0)}) '
                f'— &gt;0 means higher-Elo hypotheses hold more gold biology</div></div>')
    if "concordance" in rep:
        c = rep["concordance"]; v = rep.get("verdicts", {})
        ok = v.get("concordance_pass")
        return (f'<div class="eval-card"><h4>Concordance — Elo vs ground truth</h4>{head}'
                f'<div class="eval-kv">rows <b>{c.get("n_rows","?")}</b> · buckets <b>{c.get("n_buckets","?")}</b> · '
                f'top1 acc <b>{c.get("top1_accuracy",float("nan")):.3f}</b><br>'
                f'Spearman ρ <b>{c.get("spearman_rho",float("nan")):.3f}</b> (p={c.get("spearman_p",float("nan")):.4f}) · '
                f'logistic {c.get("logistic_coef",float("nan")):.3f}<br>'
                f'verdict: <span class="{"pass" if ok else "fail"}">{"PASS — Elo tracks correctness" if ok else "not established"}</span></div></div>')
    return f"<pre class='full'>{_esc(str(rep)[:2000])}</pre>"


# ── app ────────────────────────────────────────────────────────────────────

def build_app(runs: list[dict], reports_dir: str = "bench_runs") -> gr.Blocks:
    """Detailed explorer across runs. `runs` = explore.list_runs(...) output."""
    registry = {r["label"]: r for r in runs}
    labels = list(registry.keys())
    default = labels[0] if labels else ""

    def _resolve(sel):
        r = registry.get(sel) or (runs[0] if runs else None)
        return (r["db_path"], r["run_id"]) if r else (None, None)

    def refresh_all(sel):
        db, rid = _resolve(sel)
        if not db:
            empty = "<p style='color:#9ca3af'>No run selected.</p>"
            return empty, empty, empty, empty
        return (_stats_html(db, rid), render_hypotheses(db, rid),
                render_tournament(db, rid), render_timeline(db, rid))

    async def on_inject(sel, text):
        db, rid = _resolve(sel)
        if db and text.strip():
            await inject_expert_hypothesis(StateStore(db), rid, text.strip())
        return ""

    with gr.Blocks(title="AI Co-Scientist") as app:
        gr.HTML(f"<style>{_CSS}</style>")
        with gr.Row():
            gr.Markdown("### 🔬 AI Co-Scientist — run explorer")
            run_dd = gr.Dropdown(choices=labels, value=default, label="Question / run",
                                 scale=3, min_width=320)
            refresh_btn = gr.Button("↻", scale=0, min_width=60)
            auto_chk = gr.Checkbox(value=True, label="Auto (3s)", scale=0, min_width=100)
        stats_md = gr.HTML()

        with gr.Tabs():
            with gr.Tab("Hypotheses"):
                hyp_html = gr.HTML()
            with gr.Tab("Tournament"):
                tour_html = gr.HTML()
            with gr.Tab("Timeline"):
                tl_html = gr.HTML()
            with gr.Tab("Evals"):
                report_dd = gr.Dropdown(
                    choices=explore.list_eval_reports(reports_dir),
                    label="Eval report (bench_runs/*.json)", min_width=420)
                eval_html = gr.HTML()
                report_dd.change(render_eval, inputs=report_dd, outputs=eval_html)

        with gr.Accordion("Expert input", open=False):
            inject_box = gr.Textbox(label="Inject a hypothesis into the selected run", lines=2)
            inject_btn = gr.Button("Submit", size="sm")
            inject_btn.click(on_inject, inputs=[run_dd, inject_box], outputs=inject_box)

        outs = [stats_md, hyp_html, tour_html, tl_html]
        run_dd.change(refresh_all, inputs=run_dd, outputs=outs)
        refresh_btn.click(refresh_all, inputs=run_dd, outputs=outs)
        app.load(refresh_all, inputs=run_dd, outputs=outs)

        timer = gr.Timer(3.0)
        timer.tick(refresh_all, inputs=run_dd, outputs=outs)
        auto_chk.change(lambda on: gr.Timer(active=on), inputs=auto_chk, outputs=timer)

    return app
