from agents.base import BaseAgent
from bench.goalset import load_goalset
from tools.resources import load_yaml_config, resource_path


def test_packaged_assets_fall_back_outside_project_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    config = load_yaml_config()
    assert config["supervisor"]["n_workers"] >= 1
    assert resource_path("prompts/config/parse.txt").is_file()

    agent = BaseAgent(client=object(), prompts_dir="prompts")
    assert agent.render_prompt(
        "test/hello", name="Ada", goal="validate packaging",
    ) == "Hello Ada, goal is validate packaging."

    goals = load_goalset("bench/datasets/comp_bio_goldset.jsonl")
    assert len(goals) == 10


def test_resource_path_prefers_explicit_local_override(monkeypatch, tmp_path):
    local = tmp_path / "config.yaml"
    local.write_text("provider: local-test\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resource_path("config.yaml").resolve() == local
    assert load_yaml_config()["provider"] == "local-test"
