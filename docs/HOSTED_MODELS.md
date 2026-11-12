# Multiple models and hosted small language models

How to run CoScientist against more than one model, or against a small language
model (SLM) served from a hosted or remote OpenAI-compatible endpoint.

## What is supported

| Desired experiment | Current support |
| --- | --- |
| Compare different providers/models on the same tasks | CoScientist `bench sweep` with repeated `--models` arguments |
| Use a small model for generation and a larger model for review | Fast/strong model IDs behind one backend and endpoint |
| Use a separate judge | `--judge-provider` / `--judge-model` with absolute-ranking rediscovery |
| Use a remote small model for every role | Existing `openai-compatible` backend; same model ID for both tiers |
| Mix separate vendor backends per agent inside one run | Not implemented; requires explicit role routing and provenance/accounting |

Fast-tier calls include generation, evolution, meta-review, configuration parsing,
and claim extraction. Strong-tier calls include reflection, ranking, claim
verification, and company synthesis. Multiple agent workers are concurrency,
not automatically multiple models.

### Hosted endpoint configuration

In `config.yaml`, change only the existing `openai_compatible` section
to match the chosen endpoint. Keep the top-level default provider unchanged and
select this route explicitly at invocation time:

```yaml
openai_compatible:
  base_url: https://YOUR-HOST/v1
  api_key_env: COSCIENTIST_HOSTED_API_KEY
  model_fast: YOUR-SERVED-MODEL
  model_strong: YOUR-SERVED-MODEL
  timeout_seconds: 180
```

Set `COSCIENTIST_HOSTED_API_KEY` in your local shell or ignored `.env`; the YAML
stores the variable name, not the secret. The endpoint must implement
`POST /v1/chat/completions`. If the endpoint requires no authentication, explicitly
use `api_key_env: null`.

First test a bounded model run (this uses the endpoint and can incur charges):

```bash
uv run python coscientist.py "Identify testable mechanisms for pulmonary arterial hypertension" \
  --provider openai-compatible \
  --model-fast YOUR-SERVED-MODEL --model-strong YOUR-SERVED-MODEL \
  --workers 1 --max-tasks 12 --max-time 600
```

Then compare the SLM with the existing Claude route on a small readiness pilot:

```bash
uv run python -m bench sweep \
  --benchmark rediscover \
  --models "openai-compatible=YOUR-SERVED-MODEL" \
  --models "claude-code=sonnet,opus" \
  --worker-counts 1 --seeds 0 --limit 1 --max-tasks 12 \
  --out artifacts/hosted-pilot
```

The existing spend confirmation is retained. A one-goal pilot tests whether the
model follows the output contracts and completes without failures; it cannot
establish model superiority. Record failures, latency, provider usage, retrieval
failures, and scientific scoring separately before expanding the experiment.
Use the methodology in `docs/ROADMAP.md` for a substantive comparison.

For small-fast/large-strong within one run, set the two model IDs differently.
Both IDs must be served through the same configured endpoint. A remote SLM URL
plus the Claude CLI cannot currently be combined as two tiers inside one run.

### A remote GPU example

One concrete baseline is `Qwen/Qwen3-4B-Instruct-2507`: its model card specifies
4.0B parameters, non-thinking output, and vLLM/SGLang serving. This is an example
to measure, not a claim of biotech capability. [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

On an existing GPU host with a compatible vLLM installation, serve it privately:

```bash
# Set VLLM_API_KEY in the remote environment before starting.
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 --port 8001 --served-model-name slm \
  --max-model-len 16384
```

From the client machine, an SSH tunnel can expose that listener locally:

```bash
ssh -N -L 8001:127.0.0.1:8001 USER@GPU_HOST
```

Use `http://127.0.0.1:8001/v1` as the client base URL, `slm` as both model IDs,
and the matching key in `COSCIENTIST_HOSTED_API_KEY`. vLLM documents its
OpenAI-compatible serving interface and authentication options in the
[official serving guide](https://docs.vllm.ai/en/stable/serving/online_serving/).
This setup is documented but has not been run: no GPU was provisioned, no model
was downloaded, and no paid benchmark was launched.
