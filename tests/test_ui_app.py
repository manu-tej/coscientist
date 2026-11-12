import json

from ui import app


def test_markdown_renderer_disables_model_supplied_raw_html():
    rendered = app._md('<img src=x onerror="alert(1)">\n\n**safe markdown**')

    assert "<img" not in rendered
    assert "&lt;img" in rendered
    assert "<strong>safe markdown</strong>" in rendered


def test_render_eval_handles_null_concordance_metrics(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "meta": {"dataset": "tiny", "system_version": "test"},
        "concordance": {
            "n_rows": 1,
            "n_buckets": 1,
            "top1_accuracy": None,
            "spearman_rho": None,
            "spearman_p": None,
            "logistic_coef": None,
            "logistic_p": None,
        },
        "verdicts": {"concordance_pass": False},
    }))

    rendered = app.render_eval(str(report))

    assert rendered.count("n/a") == 5
    assert "within question" in rendered
    assert "not established" in rendered


def test_render_eval_handles_null_rediscovery_metrics(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "meta": {"dataset": "tiny", "system_version": "test"},
        "rediscovery": {
            "per_goal": [{
                "goal_id": "g1",
                "pool_recall": None,
                "top1_recall": None,
                "top3_recall": None,
                "n_hyps": 0,
                "n_gold": 1,
            }],
            "mean_pool_recall": None,
            "mean_top1_recall": None,
            "mean_top3_recall": None,
            "elo_recall_spearman": None,
            "elo_recall_p": None,
            "n_hyps_scored": 0,
        },
    }))

    rendered = app.render_eval(str(report))

    assert rendered.count("n/a") == 8
    assert "g1" in rendered
