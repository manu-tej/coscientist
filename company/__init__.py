"""Org layer: a simulated AI-native drug-repurposing techbio.

`core/` + `agents/` are the per-stage research engine (the co-scientist). This
package is the *company* on top: programs that chain co-scientist runs through
gated stages, a dual (token + Modal-credit) budget ledger, stochastic
data-realized attrition, a CSO proposer, and a CEO control surface.

See docs/superpowers/specs/2026-06-06-ai-native-repurposing-techbio-design.md.

The deterministic seeded executor remains available for offline replay. The live
hierarchical executor runs a dedicated co-scientist Supervisor and validated team
synthesis for each program stage, while the model registry can add live T1
literature/KG evidence and billed T2 structure or affinity computation.
"""
