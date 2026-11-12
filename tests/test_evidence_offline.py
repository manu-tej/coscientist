"""The free path must exercise grounding and verification, with zero network."""
import socket

import pytest

from tools.evidence import FixtureEvidenceSource, LiteratureTool
from tools.offline import OfflineBackend
from tools.verify import ClaimVerifier, extract_claims


@pytest.fixture
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("offline path attempted a network connection")

    for target, attr in ((socket.socket, "connect"), (socket.socket, "connect_ex"),
                         (socket, "create_connection"), (socket, "getaddrinfo")):
        monkeypatch.setattr(target, attr, blocked)


async def test_offline_backend_emits_extractable_claims(no_network):
    claims = await extract_claims("Hypothesis: PKG-I phosphorylates RhoA.",
                                  OfflineBackend(), max_claims=4)
    assert claims, "offline backend must emit a numbered claim list"


async def test_full_verification_flow_offline(no_network):
    verdicts = await ClaimVerifier(max_claims=2).verify(
        "Hypothesis: PKG-I phosphorylates RhoA at Ser188.\nSummary: PKG-I acts on RhoA.",
        LiteratureTool(sources=[FixtureEvidenceSource()]), OfflineBackend(),
    )
    assert verdicts
    assert all(v.status in ("supported", "unsupported", "contradicted", "unchecked")
               for v in verdicts)


async def test_some_seed_produces_a_contradiction(no_network):
    """The hard rule must be reachable offline, not only via hand-built dataclasses."""
    seen = set()
    for seed in range(40):
        verdicts = await ClaimVerifier(max_claims=2).verify(
            f"Hypothesis: mechanism {seed} operates in tissue {seed}.\nSummary: s.",
            LiteratureTool(sources=[FixtureEvidenceSource()]), OfflineBackend(seed=seed),
        )
        seen.update(v.status for v in verdicts)
    assert "contradicted" in seen, f"no seed produced a contradiction; saw {seen}"


async def test_fixture_records_are_marked_synthetic(no_network):
    records = (await FixtureEvidenceSource().search_outcome("anything")).records
    assert all(not r.ref.isdigit() for r in records)
    assert all("synthetic" in r.title.lower() for r in records)
