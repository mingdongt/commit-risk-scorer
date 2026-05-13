"""Retriever abstract base + the three-layer taxonomy.

Why three layers
----------------
Real enterprise knowledge bases are heterogeneous. A code-aware embedding
model that excels at "find me similar past PRs" performs poorly on long-form
prose like a PRD; a structured KV lookup that answers "is this module frozen
this week?" doesn't need embeddings at all. Forcing all sources through one
retriever yields uniformly mediocre quality and high cost.

The taxonomy:

    Layer A — Code-adjacent operational data (always-on)
        High signal-to-noise, structured, queried on every Tier-3 call.
        Examples: past PRs, incident store, CODEOWNERS, ADRs.
        Retrieval: code-aware embeddings (Voyage-Code / CodeBERT) + BM25 hybrid.

    Layer B — Operational context (metadata-triggered)
        Time-series / tabular signals; only queried when PR metadata flags
        relevance (e.g. diff touches an oncall-critical service).
        Examples: release calendar, on-call rotation, feature-flag state.
        Retrieval: structured queries (SQL / KV) plus light vector search.

    Layer C — Cross-functional enterprise knowledge (LLM-judged)
        Heterogeneous prose; only queried when the agent decides the diff's
        semantics warrant it (e.g. "this touches PII").
        Examples: PRDs, compliance docs, customer support themes, vendor
        contracts, NVIDIA-specific domain KB (hardware spec, ASIL safety).
        Retrieval: general-purpose long-context embeddings + cross-encoder
        re-ranker + strict provenance tracking.

Open-source vs enterprise
-------------------------
Every retriever in this package defines an interface and provides an
OSS-data proxy implementation (or a clearly documented stub). Adopters
swap the backend without changing the agent-facing tool signature — the
selling point of this RAG architecture for enterprise deployment.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from src.rag.types import RetrievalQuery, RetrievalResult


class RetrieverLayer(str, Enum):
    """Which retrieval tier a retriever belongs to.

    Stored as string values so audit logs and config files don't need to
    import the enum to serialize layer membership.
    """

    A = "A"  # code-adjacent, always-on
    B = "B"  # operational, metadata-triggered
    C = "C"  # enterprise knowledge, LLM-judged


class Retriever(ABC):
    """Base class every concrete retriever inherits.

    Implementations live under `src/rag/layer_a.py`, `layer_b.py`, `layer_c.py`.
    The `name` and `layer` properties are used by the gateway for dispatch,
    cost attribution, and audit-log grouping.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, stable identifier (used in citations and audit logs)."""

    @property
    @abstractmethod
    def layer(self) -> RetrieverLayer:
        """Which of the three layers this retriever belongs to."""

    @abstractmethod
    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Run a retrieval against this source.

        Implementations should:
        - Never raise on "no results" — return an empty `RetrievalResult`.
        - Raise `NotImplementedError` if the backend is unconfigured (stubs)
          with a message that names the OSS-proxy and enterprise alternatives.
        - Populate `Citation` for every returned document.
        """
