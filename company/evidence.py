"""Typed evidence model — provenance as a first-class product (spec §10).

A repurposing hypothesis is only as good as its traceable mechanism, so every
candidate carries an **evidence graph**: nodes are the candidate, the methods that
voted on it, and the concrete evidence each method produced (a KG mechanism path,
a literature co-occurrence, a T0 estimate); edges record *which method produced
what*, tagged with the provider and fidelity tier that generated it.

This is the typed successor to the free-form `web_citations` on reviews: instead of
a bag of URLs, the chain from claim → method → evidence is walkable, serializable,
and renderable as the rationale in a clinical-hypothesis package.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# node kinds
CANDIDATE = "candidate"
METHOD = "method"
KG_PATH = "kg_path"
CITATION = "citation"
ANALOG = "analog"
AFFINITY = "affinity"
ESTIMATE = "estimate"


@dataclass
class EvidenceNode:
    id: str
    kind: str
    label: str
    detail: str = ""


@dataclass
class EvidenceEdge:
    src: str
    dst: str
    relation: str            # "scored_by" | "supported_by"
    produced_by: str = ""    # provider name
    tier: str = ""           # fidelity tier that produced it
    value: float | None = None


@dataclass
class EvidenceGraph:
    nodes: list[EvidenceNode] = field(default_factory=list)
    edges: list[EvidenceEdge] = field(default_factory=list)

    def add_node(self, node: EvidenceNode) -> str:
        if not any(n.id == node.id for n in self.nodes):
            self.nodes.append(node)
        return node.id

    def add_edge(self, edge: EvidenceEdge) -> None:
        self.edges.append(edge)

    def node(self, node_id: str) -> EvidenceNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def of_kind(self, kind: str) -> list[EvidenceNode]:
        return [n for n in self.nodes if n.kind == kind]

    @property
    def real_evidence(self) -> list[EvidenceNode]:
        """Evidence actually computed (paths, citations, analogs, affinities) — not estimates."""
        return [n for n in self.nodes if n.kind in (KG_PATH, CITATION, ANALOG, AFFINITY)]

    def explain(self) -> list[str]:
        """Human-readable evidence chain, strongest-provenance first."""
        lines: list[str] = []
        for e in sorted(self.edges, key=lambda x: x.tier, reverse=True):
            if e.relation != "scored_by":
                continue
            method = self.node(e.dst)
            if method is None:
                continue
            support = [self.node(s.dst) for s in self.edges
                       if s.src == e.dst and s.relation == "supported_by"]
            support = [s for s in support if s is not None]
            val = f"{e.value:.2f}" if e.value is not None else "—"
            why = f" — {support[0].label}" if support else ""
            lines.append(f"[{e.tier}] {method.label} scored {val}{why}")
        return lines

    def to_dict(self) -> dict:
        return {
            "nodes": [vars(n) for n in self.nodes],
            "edges": [vars(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceGraph":
        return cls(
            nodes=[EvidenceNode(**n) for n in d.get("nodes", [])],
            edges=[EvidenceEdge(**e) for e in d.get("edges", [])],
        )


def build_candidate_evidence(drug: str, indication: str, votes: list) -> EvidenceGraph:
    """Assemble the evidence graph for one candidate from its resolved ModelOutputs.

    `votes` are (method_key, ModelOutput) pairs. Each vote becomes a method node
    edged from the candidate; whatever concrete evidence the provider returned
    (KG path, citation count) becomes a supporting node beneath it.
    """
    g = EvidenceGraph()
    root = g.add_node(EvidenceNode(f"cand:{drug}", CANDIDATE,
                                   f"{drug} → {indication}", "repurposing hypothesis"))
    for method_key, out in votes:
        mid = f"method:{method_key}:{out.provider}"
        g.add_node(EvidenceNode(mid, METHOD, f"{method_key} ({out.provider})",
                                f"facet={out.facet} tier={out.tier.value}"))
        g.add_edge(EvidenceEdge(root, mid, "scored_by", out.provider,
                                out.tier.value, out.value))

        ev = out.evidence or {}
        if ev.get("kind") == KG_PATH and ev.get("path"):
            path = " → ".join(ev["path"])
            nid = f"kgpath:{drug}"
            g.add_node(EvidenceNode(nid, KG_PATH, path, "weighted drug→target→disease path"))
            g.add_edge(EvidenceEdge(mid, nid, "supported_by", out.provider, out.tier.value))
        elif ev.get("kind") == AFFINITY and ev.get("target"):
            nid = f"affinity:{drug}"
            g.add_node(EvidenceNode(
                nid, AFFINITY, f"predicted binder of {ev['target']}",
                f"Boltz-2 on {ev.get('region', '?')}; "
                f"pred affinity {ev.get('pred_value')}"))
            g.add_edge(EvidenceEdge(mid, nid, "supported_by", out.provider, out.tier.value))
        elif ev.get("kind") == ANALOG and ev.get("nearest"):
            nid = f"analog:{drug}"
            g.add_node(EvidenceNode(
                nid, ANALOG, f"chemical analog of {ev['nearest']}",
                f"ChemBERTa cosine {ev.get('cosine')} (leave-one-out reference)"))
            g.add_edge(EvidenceEdge(mid, nid, "supported_by", out.provider, out.tier.value))
        elif ev.get("kind") == CITATION and ev.get("count") is not None:
            nid = f"cite:{drug}"
            g.add_node(EvidenceNode(
                nid, CITATION, f"{ev['count']} co-occurring PubMed records",
                ev.get("query", "")))
            g.add_edge(EvidenceEdge(mid, nid, "supported_by", out.provider, out.tier.value))
    return g
