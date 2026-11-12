"""The run explorer is in the README's free, no-network group, so launching it
must not enable Gradio's usage analytics (which call api.gradio.app)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _analytics_setting_after_import(env: dict) -> str:
    out = subprocess.run(
        [sys.executable, "-c",
         "import os, ui.launch; print(os.environ.get('GRADIO_ANALYTICS_ENABLED'))"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True, timeout=120,
    )
    return out.stdout.strip().splitlines()[-1]


def test_importing_launcher_disables_gradio_analytics():
    env = {k: v for k, v in os.environ.items() if k != "GRADIO_ANALYTICS_ENABLED"}
    assert _analytics_setting_after_import(env) == "False"


def test_explicit_environment_setting_is_kept():
    env = dict(os.environ, GRADIO_ANALYTICS_ENABLED="True")
    assert _analytics_setting_after_import(env) == "True"
