import asyncio
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


def build_app(store: StateStore, run_id: str, supervisor_handle: dict) -> gr.Blocks:
    """Build the Gradio UI. supervisor_handle is a mutable dict the Start/Stop
    buttons use to launch/cancel the Supervisor's asyncio task."""

    async def refresh_hypotheses():
        rows = await get_ranked_hypotheses(store, run_id)
        if not rows:
            return "No hypotheses yet."
        lines = []
        for i, r in enumerate(rows, 1):
            tag = "expert" if r["source"] == "expert" else "system"
            lines.append(
                f"**#{i}  Elo {r['elo']}**  [{tag}] `{r['method']}`  "
                f"({r['n_reviews']} reviews)\n\n{r['summary']}\n\n---"
            )
        return "\n\n".join(lines)

    async def refresh_overview():
        return await get_research_overview(store, run_id)

    async def refresh_stats():
        s = await get_run_stats(store, run_id)
        return (
            f"**{s['n_hypotheses']}** hypotheses · **{s['n_matches']}** matches · "
            f"top Elo **{s['top_elo']}** · spread **{s['elo_spread']}**\n\n"
            f"_goal:_ {s['goal'][:160]}"
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
        gr.Markdown("# AI Co-Scientist")
        gr.Markdown(f"**Run:** `{run_id}`")
        stats_md = gr.Markdown("Loading…")

        with gr.Row():
            refresh_btn = gr.Button("Refresh", variant="primary")
            auto_chk = gr.Checkbox(value=True, label="Auto-refresh (3s) — live tracking")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Hypothesis Explorer (ranked by Elo)")
                hypotheses_md = gr.Markdown("Click Refresh to load.")
            with gr.Column(scale=1):
                gr.Markdown("## Research Overview")
                overview_md = gr.Markdown("Click Refresh to load.")

        gr.Markdown("## Expert Input")
        with gr.Row():
            with gr.Column():
                inject_box = gr.Textbox(label="Inject a hypothesis", lines=3)
                inject_btn = gr.Button("Submit hypothesis")
            with gr.Column():
                review_id_box = gr.Textbox(label="Hypothesis ID to review")
                review_box = gr.Textbox(label="Your review", lines=3)
                review_btn = gr.Button("Submit review")

        refresh_btn.click(refresh_hypotheses, outputs=hypotheses_md)
        refresh_btn.click(refresh_overview, outputs=overview_md)
        refresh_btn.click(refresh_stats, outputs=stats_md)
        inject_btn.click(on_inject, inputs=inject_box, outputs=inject_box)
        review_btn.click(on_review, inputs=[review_id_box, review_box],
                         outputs=[review_id_box, review_box])

        # Live tracking: tick every 3s; the checkbox toggles it on/off.
        timer = gr.Timer(3.0)
        timer.tick(refresh_hypotheses, outputs=hypotheses_md)
        timer.tick(refresh_overview, outputs=overview_md)
        timer.tick(refresh_stats, outputs=stats_md)
        auto_chk.change(lambda on: gr.Timer(active=on), inputs=auto_chk, outputs=timer)

        # Populate immediately on page load.
        app.load(refresh_hypotheses, outputs=hypotheses_md)
        app.load(refresh_overview, outputs=overview_md)
        app.load(refresh_stats, outputs=stats_md)

    return app
