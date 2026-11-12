# References

This repository is inspired by, but is not affiliated with, the Google AI co-scientist work.

- Gottweis et al. "Accelerating scientific discovery with Co-Scientist."
  arXiv:2502.18864v2, 2026. https://arxiv.org/abs/2502.18864v2
  Version 1 (2025) was titled "Towards an AI co-scientist":
  https://arxiv.org/abs/2502.18864v1

The paper PDF is not in the current tree. Use the arXiv links above.

Eight prompt templates in `prompts/` reproduce or adapt the agent prompts the
paper publishes (v1 Appendix Figures A.1-A.8; v2 Supplementary Note 9), and
`docs/superpowers/plans/2026-05-28-foundation-and-agents.md` quotes the same
prompts. These files are not covered by the MIT licence; [`NOTICE`](../NOTICE)
lists them and the licence they are used under (CC BY 4.0, the licence of v2).

## Benchmark data

`bench/datasets/biomnibench_goldset.jsonl`, the default goal set for
`bench rediscover`, holds three BiomniBench-DA tasks (`da-1-3`, `da-1-4`,
`da-13-3`):

- Qu, Lu, Tu, Zhang, She, et al. "BiomniBench: Process-level Evaluation of LLM
  Agents for Real-world Biomedical Research." bioRxiv, 2026.
  https://doi.org/10.64898/2026.05.12.724604
  Dataset: https://huggingface.co/datasets/phylobio/BiomniBench-DA (gated
  access; the dataset card lists the benchmark artifacts under CC BY 4.0).

The dataset releases 50 of its 100 tasks and holds the other 50 out. All three
task IDs used here are among the released 50 (checked against the dataset's
public file listing on 2026-11-11). The goal text and gold findings in the file
were curated from those tasks and have not been compared word for word with the
gated task files. The dataset credits each task's source publication in the
task's own files; this repository does not repeat those citations.

The GPQA fixture in `bench/datasets/fixtures/` is hand-written; real GPQA
questions are fetched at run time and are not stored here.

## Related work

Two publications from the Zou lab (Stanford) that landed in the same week,
September 2026. The `company/` portfolio layer here (first commit 2026-06-18)
has a similar CSO -> programs -> gated stages structure. The Science paper's
analysis of clinical-trial outcomes is the kind of external benchmark that layer
has not yet been measured against.

- Zhang, Eckmann, Miao, Mahon, Zou. "The Virtual Biotech: A Multi-Agent AI
  Framework for Therapeutic Discovery and Development." *Science*, 2026-09-17.
  https://doi.org/10.1126/science.aeg6779
  Preprint (bioRxiv, 2026-02-23): https://www.biorxiv.org/content/10.64898/2026.02.23.707551v1
  A Chief Scientific Officer agent delegates to domain-specialized scientist
  agents. More than 37,000 agents annotated outcomes from 55,984 clinical
  trials; drugs targeting cell-type-specific genes were 48% more likely to reach
  market, with 32% fewer adverse events, and (in the preprint) 40% more likely
  to progress from Phase I to Phase II. It also evaluated B7-H3 as a lung cancer
  target and proposed an antibody-drug conjugate strategy.

- Miao, Davis, Zhang, Pritchard, Zou. "Reimagining research papers as
  interactive and reliable AI agents." *Nature*, 2026-09-16.
  https://www.nature.com/articles/s41586-026-11044-y
  Preprint as "Paper2Agent" (arXiv:2509.06917). Converts a paper plus its
  codebase into an MCP server that a chat agent can call. Relevant to the
  evidence layer in `tools/`: an agent that can re-run a method is stronger
  grounding than a quoted abstract.
