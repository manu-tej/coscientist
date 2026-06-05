import asyncio
import html
import re
import yaml
from pathlib import Path
from core.state import StateStore
from ui.data import (
    get_ranked_hypotheses, get_research_overview, get_run_stats,
    inject_expert_hypothesis, submit_expert_review,
)

import gradio as gr


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


_CSS = """
.gradio-container { max-width: 1180px !important; }
#stats { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:2px 0 0; }
#stats .chip { background:#eef2ff; border:1px solid #c7d2fe; border-radius:999px;
  padding:3px 11px; font-size:13px; color:#3730a3; }
#stats .chip b { color:#1e1b4b; }
#stats .goal { color:#6b7280; font-size:12.5px; margin-left:2px; }
.hyp-scroll { max-height:72vh; overflow-y:auto; padding-right:6px; }
.hyp-card { border:1px solid #e7e9ee; border-radius:11px; padding:9px 12px;
  margin-bottom:8px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.03); }
.hyp-head { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
.hyp-head .rank { font-weight:700; color:#111827; font-size:13px; }
.hyp-head .elo { background:#111827; color:#fff; border-radius:6px; padding:1px 8px;
  font-size:12px; font-weight:600; }
.hyp-head .method { background:#f3f4f6; border:1px solid #e5e7eb; border-radius:5px;
  padding:1px 7px; font-size:11px; color:#374151; }
.hyp-head .method.expert { background:#fef3c7; border-color:#fde68a; color:#92400e; }
.hyp-head .meta { margin-left:auto; color:#9ca3af; font-size:11px; white-space:nowrap; }
.hyp-card .summary { color:#374151; font-size:13.5px; line-height:1.5; }
.hyp-card .bar { height:4px; background:#eef0f4; border-radius:3px; margin-top:7px; overflow:hidden; }
.hyp-card .bar > span { display:block; height:100%;
  background:linear-gradient(90deg,#6366f1,#22c55e); }
#overview-box { background:#fafafa; border:1px solid #eee; border-radius:10px;
  padding:10px 12px; font-size:13px; color:#444; max-height:72vh; overflow-y:auto; }
"""


_BOILER = re.compile(
    r"^(?:\s*(?:novel research hypothesis|hypothesis|introduction|abstract|"
    r"summary|statement|---)\s*[:\-]*\s*)+",
    re.IGNORECASE,
)


def _clean(text: str, n: int = 240) -> str:
    """Strip markdown noise + leading section boilerplate, collapse to one line."""
    t = re.sub(r"[#*_`>\[\]]", "", text or "")
    t = re.sub(r"\s+", " ", t).strip()
    t = _BOILER.sub("", t).strip()
    return (t[: n - 1] + "…") if len(t) > n else t


def build_app(store: StateStore, run_id: str, supervisor_handle: dict) -> gr.Blocks:
    """Compact live explorer for one co-scientist run. Reads the run DB and
    auto-refreshes; supervisor_handle is accepted for API compatibility."""

    async def refresh_hypotheses():
        rows = await get_ranked_hypotheses(store, run_id)
        if not rows:
            return "<p style='color:#9ca3af;padding:8px'>No hypotheses yet — waiting for generation…</p>"
        elos = [r["elo"] for r in rows]
        lo, span = min(elos), (max(elos) - min(elos)) or 1.0
        cards = ['<div class="hyp-scroll">']
        for i, r in enumerate(rows, 1):
            pct = int(100 * (r["elo"] - lo) / span)
            expert = " expert" if r["source"] == "expert" else ""
            cards.append(
                f'<div class="hyp-card"><div class="hyp-head">'
                f'<span class="rank">#{i}</span>'
                f'<span class="elo">Elo {r["elo"]}</span>'
                f'<span class="method{expert}">{html.escape(r["method"])}</span>'
                f'<span class="meta">{r["n_reviews"]} reviews</span></div>'
                f'<div class="summary">{html.escape(_clean(r["summary"]))}</div>'
                f'<div class="bar"><span style="width:{pct}%"></span></div></div>'
            )
        cards.append("</div>")
        return "".join(cards)

    async def refresh_overview():
        ov = await get_research_overview(store, run_id)
        return f'<div id="overview-box">{html.escape(_clean(ov, 4000)).replace(chr(10), "<br>")}</div>'

    async def refresh_stats():
        s = await get_run_stats(store, run_id)
        return (
            '<div id="stats">'
            f'<span class="chip"><b>{s["n_hypotheses"]}</b> hypotheses</span>'
            f'<span class="chip"><b>{s["n_matches"]}</b> matches</span>'
            f'<span class="chip">top Elo <b>{s["top_elo"]}</b></span>'
            f'<span class="chip">spread <b>{s["elo_spread"]}</b></span>'
            f'<span class="goal">{html.escape(s["goal"][:150])}</span></div>'
        )

    async def on_inject(text):
        if text.strip():
            await inject_expert_hypothesis(store, run_id, text.strip())
        return ""

    async def on_review(hyp_id, critique):
        if hyp_id.strip() and critique.strip():
            await submit_expert_review(store, hyp_id.strip(), critique.strip())
        return "", ""

    with gr.Blocks(title="AI Co-Scientist") as app:
        gr.HTML(f"<style>{_CSS}</style>")
        with gr.Row():
            gr.Markdown(f"### 🔬 AI Co-Scientist  ·  run `{run_id}`")
            refresh_btn = gr.Button("↻ Refresh", scale=0, min_width=110)
            auto_chk = gr.Checkbox(value=True, label="Auto (3s)", scale=0, min_width=110)
        stats_md = gr.HTML()

        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                gr.Markdown("#### Hypotheses · ranked by Elo")
                hypotheses_md = gr.HTML()
            with gr.Column(scale=2):
                gr.Markdown("#### Research overview")
                overview_md = gr.HTML()

        with gr.Accordion("Expert input — inject a hypothesis or review one", open=False):
            with gr.Row():
                with gr.Column():
                    inject_box = gr.Textbox(label="Inject a hypothesis", lines=2)
                    inject_btn = gr.Button("Submit hypothesis", size="sm")
                with gr.Column():
                    review_id_box = gr.Textbox(label="Hypothesis ID")
                    review_box = gr.Textbox(label="Your review", lines=2)
                    review_btn = gr.Button("Submit review", size="sm")

        for fn, out in ((refresh_hypotheses, hypotheses_md),
                        (refresh_overview, overview_md),
                        (refresh_stats, stats_md)):
            refresh_btn.click(fn, outputs=out)
            app.load(fn, outputs=out)

        timer = gr.Timer(3.0)
        for fn, out in ((refresh_hypotheses, hypotheses_md),
                        (refresh_overview, overview_md),
                        (refresh_stats, stats_md)):
            timer.tick(fn, outputs=out)
        auto_chk.change(lambda on: gr.Timer(active=on), inputs=auto_chk, outputs=timer)

        inject_btn.click(on_inject, inputs=inject_box, outputs=inject_box)
        review_btn.click(on_review, inputs=[review_id_box, review_box],
                         outputs=[review_id_box, review_box])

    return app
