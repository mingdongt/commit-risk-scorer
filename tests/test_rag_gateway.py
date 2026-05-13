"""Tests for the multi-source enterprise RAG gateway.

The gateway is responsible for *dispatch*, not retrieval quality. These
tests pin the routing contract:

  - Layer A retrievers always fire.
  - Layer B retrievers fire only when their declared triggers intersect
    the query triggers; otherwise they're recorded as skipped.
  - Layer C retrievers are never auto-invoked by `dispatch()`; they're
    exposed as agent-callable tools via `dispatch_layer_c()`.
  - Stub retrievers (`NotImplementedError`) degrade to empty results so
    the system runs end-to-end while implementations land incrementally.

End-to-end retrieval quality is the concern of each retriever's own test
file once concrete implementations land.
"""
from __future__ import annotations

import pytest

from src.rag.base import Retriever, RetrieverLayer
from src.rag.gateway import RagGateway
from src.rag.layer_a import LAYER_A_RETRIEVERS
from src.rag.layer_b import LAYER_B_RETRIEVERS, OnCallStatusRetriever
from src.rag.layer_c import LAYER_C_RETRIEVERS, ComplianceCheckRetriever
from src.rag.types import Citation, Document, RetrievalQuery, RetrievalResult


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeRetriever(Retriever):
    """Returns a fixed document and tracks invocation count.

    Used to verify that the gateway actually called this retriever (rather
    than skipping it) without depending on any real backend.
    """

    def __init__(
        self,
        name: str,
        layer: RetrieverLayer,
        triggers: frozenset[str] = frozenset(),
        when_to_invoke: str = "",
    ):
        self._name = name
        self._layer = layer
        self._triggers = triggers
        self._when_to_invoke = when_to_invoke
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def layer(self) -> RetrieverLayer:
        return self._layer

    @property
    def triggers(self) -> frozenset[str]:
        return self._triggers

    @property
    def when_to_invoke(self) -> str:
        return self._when_to_invoke

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        self.calls += 1
        return RetrievalResult(
            layer=self._layer.value,
            retriever_name=self._name,
            documents=(
                Document(
                    content=f"hit-from-{self._name}",
                    citation=Citation(source_type="fake", source_id=self._name),
                    score=0.99,
                ),
            ),
        )


def _query(triggers: set[str] | None = None) -> RetrievalQuery:
    return RetrievalQuery(
        question="Is this PR risky?",
        diff="dummy diff",
        pr_metadata={"author": "alice"},
        triggers=frozenset(triggers or set()),
    )


# ---------------------------------------------------------------------------
# Layer A — always-on
# ---------------------------------------------------------------------------


def test_layer_a_retrievers_always_fire():
    a1 = _FakeRetriever("a1", RetrieverLayer.A)
    a2 = _FakeRetriever("a2", RetrieverLayer.A)
    gateway = RagGateway(retrievers=[a1, a2])

    response = gateway.dispatch(_query())

    assert a1.calls == 1
    assert a2.calls == 1
    fired_names = {r.retriever_name for r in response.results}
    assert fired_names == {"a1", "a2"}


def test_layer_a_fires_regardless_of_triggers():
    """Layer A is unconditional — triggers shouldn't matter."""
    a1 = _FakeRetriever("a1", RetrieverLayer.A)
    gateway = RagGateway(retrievers=[a1])

    gateway.dispatch(_query(triggers={"any_random_trigger"}))
    gateway.dispatch(_query(triggers=set()))

    assert a1.calls == 2


# ---------------------------------------------------------------------------
# Layer B — metadata-triggered
# ---------------------------------------------------------------------------


def test_layer_b_fires_when_trigger_matches():
    b = _FakeRetriever(
        "oncall_status",
        RetrieverLayer.B,
        triggers=frozenset({"touches_oncall_critical"}),
    )
    gateway = RagGateway(retrievers=[b])

    gateway.dispatch(_query(triggers={"touches_oncall_critical"}))

    assert b.calls == 1


def test_layer_b_skipped_when_trigger_absent():
    b = _FakeRetriever(
        "oncall_status",
        RetrieverLayer.B,
        triggers=frozenset({"touches_oncall_critical"}),
    )
    gateway = RagGateway(retrievers=[b])

    response = gateway.dispatch(_query(triggers={"some_other_trigger"}))

    assert b.calls == 0
    assert "oncall_status" in response.skipped


def test_layer_b_skipped_when_no_triggers_at_all():
    b = _FakeRetriever(
        "oncall_status",
        RetrieverLayer.B,
        triggers=frozenset({"touches_oncall_critical"}),
    )
    gateway = RagGateway(retrievers=[b])

    response = gateway.dispatch(_query(triggers=None))

    assert b.calls == 0
    assert "oncall_status" in response.skipped


def test_layer_b_with_multiple_retrievers_each_evaluated_independently():
    b_oncall = _FakeRetriever(
        "oncall_status",
        RetrieverLayer.B,
        triggers=frozenset({"touches_oncall_critical"}),
    )
    b_flag = _FakeRetriever(
        "feature_flag_state",
        RetrieverLayer.B,
        triggers=frozenset({"references_feature_flag"}),
    )
    gateway = RagGateway(retrievers=[b_oncall, b_flag])

    gateway.dispatch(_query(triggers={"references_feature_flag"}))

    assert b_oncall.calls == 0
    assert b_flag.calls == 1


# ---------------------------------------------------------------------------
# Layer C — agent-called, NOT auto-invoked
# ---------------------------------------------------------------------------


def test_layer_c_never_fires_on_dispatch():
    c = _FakeRetriever(
        "compliance_check",
        RetrieverLayer.C,
        when_to_invoke="When PII is touched",
    )
    gateway = RagGateway(retrievers=[c])

    response = gateway.dispatch(_query(triggers={"touches_pii"}))

    # Even with a matching-looking trigger, Layer C must NOT auto-fire.
    assert c.calls == 0
    assert "compliance_check" in response.skipped


def test_layer_c_fires_on_explicit_call():
    c = _FakeRetriever(
        "compliance_check",
        RetrieverLayer.C,
        when_to_invoke="When PII is touched",
    )
    gateway = RagGateway(retrievers=[c])

    result = gateway.dispatch_layer_c("compliance_check", _query())

    assert c.calls == 1
    assert result.retriever_name == "compliance_check"
    assert len(result.documents) == 1


def test_dispatch_layer_c_unknown_name_raises():
    gateway = RagGateway(retrievers=[])
    with pytest.raises(KeyError, match="No Layer-C retriever"):
        gateway.dispatch_layer_c("nonexistent", _query())


def test_available_layer_c_tools_returns_descriptions():
    c = _FakeRetriever(
        "compliance_check",
        RetrieverLayer.C,
        when_to_invoke="When PII is touched",
    )
    gateway = RagGateway(retrievers=[c])

    tools = gateway.available_layer_c_tools()

    assert ("compliance_check", "When PII is touched") in tools


# ---------------------------------------------------------------------------
# Stub graceful degradation
# ---------------------------------------------------------------------------


def test_stub_retrievers_degrade_to_empty_results():
    """Default registry is all stubs; dispatch should still succeed end-to-end.

    This is the contract that lets the project remain runnable while
    individual retriever implementations land incrementally.
    """
    gateway = RagGateway()  # default registry — all stubs
    response = gateway.dispatch(
        _query(triggers={"touches_oncall_critical"})
    )

    # Layer A: every retriever ran (and returned empty)
    layer_a_names = {cls().name for cls in LAYER_A_RETRIEVERS}
    fired_a = {
        r.retriever_name for r in response.results if r.layer == "A"
    }
    assert fired_a == layer_a_names
    for r in response.results:
        if r.layer == "A":
            assert r.documents == ()

    # Layer B: only retrievers whose triggers fired ran; the rest skipped.
    fired_b = {r.retriever_name for r in response.results if r.layer == "B"}
    # The oncall trigger should have fired exactly OnCallStatusRetriever.
    assert OnCallStatusRetriever().name in fired_b

    # Layer C: all skipped.
    layer_c_names = {cls().name for cls in LAYER_C_RETRIEVERS}
    assert layer_c_names.issubset(set(response.skipped))


def test_stub_layer_c_dispatch_returns_empty_not_raises():
    """Explicit Layer-C invocation against a stub returns empty, not raises."""
    gateway = RagGateway()
    result = gateway.dispatch_layer_c("compliance_check", _query())

    assert result.retriever_name == "compliance_check"
    assert result.documents == ()


# ---------------------------------------------------------------------------
# Sanity: default registry covers every declared retriever class
# ---------------------------------------------------------------------------


def test_default_registry_includes_every_declared_retriever():
    gateway = RagGateway()
    declared = (
        {cls().name for cls in LAYER_A_RETRIEVERS}
        | {cls().name for cls in LAYER_B_RETRIEVERS}
        | {cls().name for cls in LAYER_C_RETRIEVERS}
    )
    actual = {r.name for r in gateway.retrievers}
    assert declared == actual
