# AI-Native Drug-Repurposing Techbio — Design Spec

**Date:** 2026-06-06
**Status:** Living draft (co-designed in session; refine in place)
**Builds on:** the existing AI co-scientist (arXiv:2502.18864 reimplementation in `core/`, `agents/`, `bench/`)

> **One-line:** A simulated, AI-native drug-**repurposing** company. Programs are chains of
> co-scientist runs through gated stages; specialist-persona agents *orchestrate a stack of
> SOTA scientific models* to run in-silico experiments; stochastic, data-realized attrition and
> a dual (token + Modal-GPU) credit budget give the org real teeth; a CSO agent proposes the
> portfolio and **you (CEO) approve gates**; a compounding knowledge base is the moat.

---

## 0. How we got here (decisions, locked)

| Dimension | Decision |
|---|---|
| Company type | Drug-discovery **techbio**, scoped to **drug repurposing** |
| Simulation scope | Scientific work **+ org dynamics** (budget, headcount, gates, timelines) |
| Control | **You = CEO** (approve gates); **CSO agent proposes** |
| Risk model | **Stochastic**, realized as in-silico experimental data (seeded → reproducible) |
| AI-native stance | Agents are **orchestrators of SOTA scientific models**, not the scientists |
| Model access | **Hosted-API + Modal serverless GPU** (500 credits) + CPU-real + **T0 LLM-estimate floor** |
| Moat | **Compounding knowledge base** (warm-starts new programs; doubles as compute cache) |

---

## 1. Goals & non-goals

**Goals**
- Reuse the co-scientist pipeline as the per-stage engine; add an **org layer** above it.
- Generate (drug → new-indication) hypotheses from **multiple independent methods** and let the
  tournament/proximity machinery surface **multi-method consensus** as the confidence signal.
- Make budget *real*: LLM tokens + Modal GPU credits are the literal R&D spend.
- Be **provably useful**: rediscover known repurposings held out as ground truth.
- Keep every prediction **fully traceable** to its evidence chain.

**Non-goals (v1)**
- De-novo molecule design (repurposing reuses approved drugs).
- Wet-lab validation; large human-expert panels.
- Modalities beyond small molecules (antibodies/PROTACs/oligos are additive later — registry is modality-tagged).

---

## 2. The org (what "company" means)

```
  CEO (you) ── approve/override gates & portfolio ──┐
        ▲                                            ▼
   gate packets, rNPV               CSO AGENT  ── proposes portfolio plan,
   report cards                     (strategy)     gate recs, credit/headcount split
        │                                            │
        └──────────── teams run stage co-scientist runs ──────────┘
   Teams: Disease-Biology · Mechanism/Pharmacology · Cheminformatics ·
          Network/KG Data Science · Translational/RWE · Safety-reuse · Critic
   Shared services: Model Registry · Knowledge Base · Ledger · Literature
```

A **team** = today's co-scientist pipeline, specialized with a persona roster + model toolset.
The **CSO** is a higher-order supervisor: it ranks *programs*, not hypotheses.

---

## 3. The repurposing pipeline (stages = co-scientist runs, linked by gates)

| Stage | Run goal | Output | Dominant risk |
|---|---|---|---|
| **Disease dossier** | characterize mechanism, targets, expression signature | target/pathway/signature profile | wrong disease model |
| **Repurposing hypotheses** | propose approved drugs via multiple methods | (drug → indication) pairs + rationale | weak rationale |
| **Mechanistic validation** | does the drug plausibly hit disease biology? | binding + network + pathway evidence | no mechanism |
| **Translational evidence** | RWE, prior trials, feasibility, safety reuse | clinical-hypothesis package | no efficacy signal |

The co-scientist run is the **unit of work inside a stage**; the company layer chains stages
through **go/no-go gates** and arbitrates resources across concurrent programs.

**Economics invert vs de-novo:** an approved drug's safety/PK/tox is largely *known*, so attrition
concentrates in **efficacy/mechanism in the new indication** → lower cost, faster, higher baseline PoS.

---

## 4. Candidate generation = generation strategies (the voting engine)

Each method is a generation strategy (extends `_GENERATION_STRATEGIES`, `core/supervisor.py:29`):

| Method | Proposes (drug → indication) via | Cost tier |
|---|---|---|
| **KG link prediction** (TxGNN-style) | embed PrimeKG/DRKG, predict drug↔disease edges (zero-shot to rare dz) + paths | CPU/GPU |
| **Signature reversal** (CMap/L1000) | drugs whose expression signature anti-correlates with disease signature | CPU |
| **Target/mechanism** | disease targets (OpenTargets) → approved drugs hitting them (ChEMBL/DGIdb) | API |
| **Structure-based** | dock approved-drug library vs disease-target structures | GPU (Modal) |
| **Network proximity** | drug-target network distance to disease module (network medicine) | CPU |
| **Literature/analogy** | PubMed/bioRxiv mining; off-label & adverse-event signals | MCP |

**Consensus = signal.** Candidates proposed by multiple independent methods cluster under the
existing **proximity** agent (`agents/proximity.py`) and rank up via **Elo** — multi-method
agreement is the repurposing confidence the tournament already knows how to surface.

---

## 5. Model layer — registry + adapters

A declarative **`models.yaml`** maps each *facet* to ordered providers with graceful fallback;
a `ModelAdapter` interface mirrors the existing `LLMBackend` factory (`tools/llm.py:22`).

```yaml
complex_structure: { providers: [modal_boltz2, af_server, t0_llm_estimate] }
kg_repurposing:    { providers: [modal_txgnn, cpu_kge, t0_llm_estimate] }
affinity:          { providers: [modal_boltz2, t0_llm_estimate] }
admet:             { providers: [cpu_admet_ai, t0_llm_estimate] }
```

| Facet | SOTA model(s) | Runs on |
|---|---|---|
| KG repurposing | **TxGNN**, DRKG/PrimeKG/Hetionet + KGE | Modal GPU / CPU |
| Signature reversal | LINCS L1000 / CMap | CPU + data |
| Drug→target binding | **Boltz-2**, DiffDock-L, gnina | Modal GPU |
| Target structure | ESMFold / AlphaFold2 / AF3 | Modal GPU / AF Server |
| Network / pathway | network-medicine proximity, GSEA | CPU |
| Cheminformatics / ADMET | RDKit, ADMET-AI, Chemprop | CPU (real, no key) |
| Data | OpenTargets, ChEMBL, DGIdb, PubChem, DisGeNET | API |
| Literature / RWE | PubMed, bioRxiv, Scholar Gateway, Owkin | MCP |
| Reasoning / agents / judging | Claude Opus 4.8 + Sonnet 4.6 (per-role map) | API |

**Fidelity tiers:** T0 (LLM-estimate, labeled) · T1 (real CPU library) · T2 (real SOTA model).
The gate packet flags which readouts were real-computed vs simulated.

**CPU-real boundary (live today, zero keys):** RDKit · ADMET-AI · Chemprop · structural alerts ·
real PubMed/bioRxiv. Everything else is T0 until a Modal/API adapter is activated.

**Persona × model-facet matrix** is the access-control table that makes personas behaviorally
distinct (a Cheminformatician *is* "an LLM wielding RDKit + docking + ADMET").

---

## 6. Org dynamics → existing mechanics

| Org concept | Maps onto |
|---|---|
| **R&D budget** | **dual ledger**: LLM tokens (reasoning) + **Modal GPU credits** (compute). Real money. |
| **Headcount** | shared **worker pool** (`n_workers`, `core/supervisor.py:113`) = elastic agent workforce |
| **Intra-program prioritization** | the weighted scheduler (`compute_weights`/`sample_agent_type`) |
| **Inter-program prioritization** | the **CSO** (§8) — same logic lifted a level |
| **Attrition / risk** | stochastic, data-realized readouts (§7) |

**Cost-tiered cascade:** cheap CPU methods screen *all* candidates; expensive GPU docking runs
**only on tournament leaders** (mirrors the multi-turn-debate threshold, `config.yaml:23`).
KB caches results keyed by (model, inputs) → repeated experiments cost **0** credits.

---

## 7. Risk, scoring & the rNPV objective

**Objective:** maximize risk-adjusted portfolio value
> `score = Σ_programs ( cumulative_PoS × estimated_value ) − total_spend`

**Per-stage baseline PoS** (illustrative, tunable; repurposing-elevated because safety is de-risked):
Target/dossier→Hyp ~0.65 · Hyp→Mechanism ~0.55 · Mechanism→Translational ~0.50.

**Scoring → PoS coupling:** each generation method emits a score; an **evidence-integration** step
(weighted ensemble + LLM-judge sanity) fuses them to a **confidence**. Then:
> `realized_readout ~ Normal(center = integrated_confidence, σ = f(method_agreement))` + seeded noise

More independent agreeing methods → tighter σ → more confident. Stochastic kills happen when
fused readouts cross a threshold — **failures arrive with data and a reason**, not a bare coin flip.

---

## 8. CSO strategy logic (the portfolio brain)

Each cycle the CSO (an LLM agent over portfolio state + a quantitative allocator it can justify):
1. **Allocates** next-cycle credits ∝ **expected marginal rNPV per credit** (information value),
   not just current rank — a bandit-flavored **explore→exploit** curve (spread early, concentrate
   on validated leaders late).
2. **Recommends gate actions**: advance / kill / hold. *Kill rule:* confidence < θ after N
   experiments, or negative rNPV contribution, or credits-to-next-gate > priority-weighted budget.
3. **Proposes new programs**: scans KB + literature for high-value unaddressed diseases.
4. Emits a ranked **action list with rNPV rationale** → the portfolio/gate packet the **CEO approves**.

---

## 9. Time & parallelism model

Discrete **cycles** ("quarters"); wall-clock is decoupled from sim-time (a cycle = a unit of
committed work). Per cycle:
`CSO proposes → CEO approves → committed stage-runs execute in parallel (bounded by worker pool +
Modal credits) → experiments resolve (stochastic) → due gates fire → KB updated → ledger debited.`
Programs run concurrently; a program advances stage only through an approved gate.

---

## 10. Provenance / explainability (the product)

The deliverable is a **clinical-hypothesis package**:
`(drug, indication, mechanism, evidence graph, confidence, known-safety summary, suggested validation)`.

- Every candidate carries an **evidence graph**: nodes = KG paths / signature scores / docking
  results / citations / prior KB artifacts; edges = which method/agent produced what.
- Every readout records: model, fidelity tier, inputs, seed, raw output.
- Every gate records: packet, CSO rec, CEO decision, realized outcome.
- Extends the existing `web_citations` on reviews (`core/orchestrator.py:160`) into a typed evidence model.

A repurposing hypothesis is only as good as its traceable mechanism → provenance is first-class.

---

## 11. Validation — the company's report card (revives the parked bench work)

The repurposing twist makes validation concrete and contamination-resistant:

- **Held-out KG edges:** train KGE on a split, hold out known indications → recall@k / AUROC.
- **Time-split:** repurposings established after a cutoff; check the system proposes them from
  pre-cutoff knowledge only (contamination-resistant).
- **Curated gold set:** famous repurposings (sildenafil→PAH, thalidomide→myeloma, metformin→onc,
  minoxidil→hair-loss, raloxifene…) as named test cases.
- **Concordance:** does internal confidence (Elo / integrated-confidence) track ground-truth
  correctness? — **reuses `bench concordance` directly**.
- **Negative controls:** random drug-disease pairs and decoy diseases must score low.

The parked stress campaign (`bench concordance` / `rediscover`) becomes this suite.

---

## 12. Components — reused vs new

**Reused as-is:** all 6 agents, reflection pipeline, Elo tournament, proximity clustering,
`BaseAgent`, LLM backends, `bench/` harness.

**New (thin):** `core/company.py` (portfolio + resource arbiter) · `core/program.py` (stage/gate
state machine) · `CSOAgent` · dual `Ledger` · CEO CLI (`company new`, `portfolio`, `program new`,
`cso propose`, `gate`, `advance/kill/hold`, `quarter`).

**New (meatier):** `tools/models/` model registry + adapters (Modal, AF-Server, HF, MCP, CPU,
T0-fallback) · `modal_app.py` (GPU functions) · persona rosters + prompts per team · KB schema
(promote `StateStore` to multi-program: company/program/stage_run/experiment/gate/kb_artifact/ledger)
· true tool-use loop for analysis agents.

---

## 13. Build path (vertical-slice first)

1. **Spine:** one repurposing program; `company/program/gate` state machine + dual ledger; KB
   tables; CEO CLI; stochastic gate; **CPU-real generation methods live** (literature + KG-on-CPU +
   signature), Modal/T0 elsewhere. Goal: run one program end-to-end, approve one gate.
2. **Model registry + Modal adapters** (TxGNN, Boltz-2) behind the cascade + KB cache.
3. **Multi-program + CSO** portfolio loop.
4. **Validation suite** (§11) wired through `bench`.
5. **Team dynamics & provenance** depth; additional modalities.

---

## Open questions (refine here)
- Exact PoS baselines & noise σ calibration (deferred to build).
- KB DDL specifics; evidence-graph schema.
- Which curated gold-set repurposings; cutoff date for the time-split.
- CSO allocator math (explicit marginal-rNPV formula).
- First demo disease.
