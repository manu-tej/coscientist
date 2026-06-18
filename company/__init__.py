"""Org layer: a simulated AI-native drug-repurposing techbio.

`core/` + `agents/` are the per-stage research engine (the co-scientist). This
package is the *company* on top: programs that chain co-scientist runs through
gated stages, a dual (token + Modal-credit) budget ledger, stochastic
data-realized attrition, a CSO proposer, and a CEO control surface.

See docs/superpowers/specs/2026-06-06-ai-native-repurposing-techbio-design.md.

v1 spine note: the per-stage science is a *seeded stochastic stub* (CPU-only, no
API keys, fully reproducible) so the org loop runs end-to-end today. Swapping the
stub for a live co-scientist run is the next increment — `engine.run_stage` is the
single integration seam.
"""
