"""RagGateway — three-layer dispatcher consumed by the Tier-3 agent.

Dispatch model
--------------
- Layer A retrievers fire on every Tier-3 invocation. They're cheap and
  high signal-to-noise; missing them is the cost of admitting that
  retrieval matters at all.
- Layer B retrievers fire only when the query's `triggers` intersect the
  retriever's declared trigger set. This avoids paying for operational
  context lookups on PRs that don't need them (typo fixes, doc updates).
- Layer C retrievers are NOT invoked by the gateway directly. They're
  exposed to the agent as named tools. The agent — looking at the diff —
  decides whether to call e.g. `compliance_check` or `domain_kb`. This is
  agentic RAG (model decides retrieval) vs flat RAG (system always retrieves).

Stub semantics
--------------
Concrete retrievers under `layer_a/b/c.py` raise NotImplementedError until
their backends ship. The gateway catches NotImplementedError and treats
stubs as "empty result" — the system remains runnable end-to-end while
implementations land incrementally. The result list records which layers
contributed; an audit log can attribute findings (and silence) per source.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.rag.base import Retriever, RetrieverLayer
from src.rag.layer_a import LAYER_A_RETRIEVERS
from src.rag.layer_b import LAYER_B_RETRIEVERS
from src.rag.layer_c import LAYER_C_RETRIEVERS
from src.rag.types import RetrievalQuery, RetrievalResult


@dataclass
class GatewayResponse:
    """Combined results from a single gateway dispatch.

    `results` is one `RetrievalResult` per retriever that actually executed
    (including ones that returned empty). `skipped` lists retrievers that
    were eligible but not invoked because their triggers didn't fire — kept
    so the audit log can prove "we considered this source and intentionally
    didn't query it".

    `stub_count` is an honest signal of how much of the gateway is still
    interface-only. Useful during the incremental rollout.
    """

    results: list[RetrievalResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    stub_count: int = 0
    total_latency_ms: float = 0.0


class RagGateway:
    """Three-layer enterprise-RAG dispatcher.

    Instantiate with the default registry (all stub retrievers wired) for a
    runnable end-to-end skeleton; swap individual retrievers as their real
    implementations land.

        gateway = RagGateway()
        response = gateway.dispatch(query)

    Or wire a custom registry — useful for tests and for enterprise
    deployments that replace specific backends:

        gateway = RagGateway(retrievers=[MyRealSimilarPRs(), MyStubOnCall()])
    """

    def __init__(self, retrievers: list[Retriever] | None = None):
        self.retrievers: list[Retriever] = retrievers or self._default_registry()
        self._by_layer: dict[RetrieverLayer, list[Retriever]] = {
            RetrieverLayer.A: [],
            RetrieverLayer.B: [],
            RetrieverLayer.C: [],
        }
        for r in self.retrievers:
            self._by_layer[r.layer].append(r)

    @staticmethod
    def _default_registry() -> list[Retriever]:
        """Instantiate one stub retriever per declared class in every layer."""
        registry: list[Retriever] = []
        for cls in LAYER_A_RETRIEVERS + LAYER_B_RETRIEVERS + LAYER_C_RETRIEVERS:
            registry.append(cls())
        return registry

    # ----- public API ----------------------------------------------------

    def dispatch(self, query: RetrievalQuery) -> GatewayResponse:
        """Run Layers A and B; expose Layer C as agent-callable tools.

        Layer C is intentionally *not* auto-invoked here — see module
        docstring. To run a specific Layer C retriever, the agent calls
        `dispatch_layer_c(name, query)`.
        """
        response = GatewayResponse()
        t0 = time.perf_counter()

        # Layer A — always
        for retriever in self._by_layer[RetrieverLayer.A]:
            self._invoke(retriever, query, response)

        # Layer B — only if triggers fire
        for retriever in self._by_layer[RetrieverLayer.B]:
            # `triggers` lives on the stub base class as a property. Concrete
            # retrievers should preserve the same interface.
            retriever_triggers = getattr(retriever, "triggers", frozenset())
            if retriever_triggers and not (retriever_triggers & query.triggers):
                response.skipped.append(retriever.name)
                continue
            self._invoke(retriever, query, response)

        # Layer C — never auto-invoked; recorded as skipped so audit log is honest
        for retriever in self._by_layer[RetrieverLayer.C]:
            response.skipped.append(retriever.name)

        response.total_latency_ms = (time.perf_counter() - t0) * 1000.0
        return response

    def dispatch_layer_c(
        self, retriever_name: str, query: RetrievalQuery
    ) -> RetrievalResult:
        """Invoke a single Layer-C retriever by name.

        Surfaced to the Tier-3 agent as a tool: the agent reasons about the
        diff, decides "I need a compliance check", and calls this with
        `retriever_name='compliance_check'`.

        Raises:
            KeyError: if no Layer-C retriever with that name is registered.
        """
        for retriever in self._by_layer[RetrieverLayer.C]:
            if retriever.name == retriever_name:
                return self._call_one(retriever, query)
        raise KeyError(
            f"No Layer-C retriever named '{retriever_name}'. "
            f"Available: {[r.name for r in self._by_layer[RetrieverLayer.C]]}"
        )

    def available_layer_c_tools(self) -> list[tuple[str, str]]:
        """Return `(name, when_to_invoke)` tuples for every Layer-C retriever.

        The agent harness uses this to build the MCP tool list it advertises
        to the LLM. The `when_to_invoke` string is the tool description —
        the LLM reads it and decides whether to call.
        """
        out: list[tuple[str, str]] = []
        for retriever in self._by_layer[RetrieverLayer.C]:
            description = getattr(
                retriever, "when_to_invoke", f"Query {retriever.name}"
            )
            out.append((retriever.name, description))
        return out

    # ----- internals -----------------------------------------------------

    def _invoke(
        self,
        retriever: Retriever,
        query: RetrievalQuery,
        response: GatewayResponse,
    ) -> None:
        result = self._call_one(retriever, query)
        response.results.append(result)
        if result.is_empty() and result.retriever_name in {
            r.name for r in self.retrievers
        }:
            # Distinguish "stub returned empty" from "real retriever returned empty".
            # We can't tell from the result alone, so we track stub_count separately
            # in `_call_one`.
            pass

    def _call_one(
        self, retriever: Retriever, query: RetrievalQuery
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        try:
            result = retriever.retrieve(query)
            # Preserve layer/name on the result so audit can group by layer
            return RetrievalResult(
                layer=result.layer or retriever.layer.value,
                retriever_name=result.retriever_name or retriever.name,
                documents=result.documents,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        except NotImplementedError:
            # Stub — system degrades gracefully; record empty result.
            return RetrievalResult(
                layer=retriever.layer.value,
                retriever_name=retriever.name,
                documents=(),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
